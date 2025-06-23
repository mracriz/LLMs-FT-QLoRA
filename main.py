import os
import random
import numpy as np
import torch
import asyncio
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from peft import LoraConfig
from trl import SFTConfig
from deepeval import evaluate

from scripts.pre_proc import PreprocessData
from scripts.training import LLM_LoRA_Model
from scripts.evaluate import EvaluateLLM, MMLUEvaluator, BaselineEvaluator, calculate_regression
from custom_metrics.execution_accuracy import ExecutionAccuracy

SEED = 42

# Modelo base de 8B de parâmetros, conforme sugerido no trabalho
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

SPIDER_DB_PATH = "."

# Configurações de QLoRA (quantização)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Configuração do LoRA
lora_config = LoraConfig(
    r=16,                # Rank
    lora_alpha=32,       # Alpha
    lora_dropout=0.05,   # Dropout
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Módulos alvo para Llama 3
    task_type="CAUSAL_LM",
)

# Hiperparâmetros de treinamento para os dois experimentos obrigatórios
TRAINING_CONFIGS = [
    {
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "output_dir": "./results/run_1_lr_2e-4_epochs_1",
    },
    {
        "learning_rate": 1e-4,
        "num_train_epochs": 2,
        "output_dir": "./results/run_2_lr_1e-4_epochs_2",
    }
]

def set_seed(seed: int):
    """Define a semente para todas as bibliotecas relevantes para garantir reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

async def main():
    set_seed(SEED)

    # --- FASE 1: AVALIAÇÃO DE BASELINE DO MODELO BASE ---
    print("--- INICIANDO AVALIAÇÃO DE BASELINE DO MODELO BASE ---")

    # Avaliação de Generalização (MMLU)
    print("\n[MMLU Baseline] Avaliando o modelo base para medir a capacidade de generalização inicial...")
    base_mmlu_evaluator = MMLUEvaluator(model_path=BASE_MODEL_ID, seed=SEED)
    base_mmlu_accuracy, base_mmlu_by_category = await base_mmlu_evaluator.run_evaluation()
    print(f"\n[RESULTADO] Acurácia MMLU (Base): {base_mmlu_accuracy:.2f}%")
    print(f"  - Por Categoria: {base_mmlu_by_category}")

    # Avaliação na Tarefa-Alvo (Text-to-SQL) com Few-Shot
    print("\n[Spider Baseline] Avaliando o modelo base em Text-to-SQL com prompt few-shot...")
    spider_eval_dataset = load_dataset("spider", split="validation")
    # O BaselineEvaluator usa o mesmo modelo carregado pelo MMLUEvaluator para eficiência
    baseline_text2sql_evaluator = BaselineEvaluator(
        model=base_mmlu_evaluator.model,
        tokenizer=base_mmlu_evaluator.tokenizer
    )
    baseline_test_cases = baseline_text2sql_evaluator.run_evaluation(spider_eval_dataset)
    
    print("\n[Spider Baseline] Calculando Acurácia de Execução para o baseline...")
    execution_metric = ExecutionAccuracy(model_db_path=SPIDER_DB_PATH)
    await evaluate(test_cases=baseline_test_cases, metrics=[execution_metric])
    # A acurácia será impressa pelo DeepEval. Anote este valor para o relatório.

    # Limpa a memória antes de iniciar o fine-tuning
    del base_mmlu_evaluator, baseline_text2sql_evaluator, baseline_test_cases
    torch.cuda.empty_cache()

    # --- PRÉ-PROCESSAMENTO DOS DADOS PARA FINE-TUNING ---
    print("\n--- CARREGANDO E PRÉ-PROCESSANDO DADOS PARA FINE-TUNING ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False) # Carrega tokenizer para pré-processamento
    spider_train_dataset = load_dataset("spider", split="train")
    
    preprocessor = PreprocessData(dataset=spider_train_dataset)
    processed_train_dataset = spider_train_dataset.map(
        lambda x: preprocessor.format_example(x, tokenizer, BASE_MODEL_ID),
        num_proc=4,
        remove_columns=spider_train_dataset.column_names
    )
    print("Dataset de treinamento pré-processado com sucesso.")
    del tokenizer
    torch.cuda.empty_cache()


    # --- FASES 2, 3 e 4: LOOP DE EXPERIMENTOS DE FINE-TUNING E AVALIAÇÃO ---
    for i, config in enumerate(TRAINING_CONFIGS):
        run_name = f"Execução {i+1} (LR={config['learning_rate']}, Epochs={config['num_train_epochs']})"
        print(f"\n--- INICIANDO {run_name.upper()} ---")

        # --- FASE 2: Execução do Fine-Tuning ---
        print(f"\n[{run_name}] Iniciando fine-tuning com QLoRA...")
        
        # Configuração do TRL SFTTrainer
        training_args = SFTConfig(
            output_dir=config['output_dir'],
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=config['learning_rate'],
            num_train_epochs=config['num_train_epochs'],
            # Parâmetros para eficiência
            max_seq_length=1024, # Spider pode ter contextos de schema longos
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            # Logging e salvamento
            logging_steps=25,
            save_strategy="epoch",
            seed=SEED,
            report_to="none",
            dataset_text_field="text" # Especifica a coluna com o texto formatado
        )
        
        lora_model_trainer = LLM_LoRA_Model(train_data=processed_train_dataset)
        # O modelo é carregado dentro do método `train` para garantir que esteja limpo
        trained_model = lora_model_trainer.train(
            model_id=BASE_MODEL_ID,
            lora_config=lora_config,
            quantization_config=quantization_config,
            training_arguments=training_args
        )

        adapter_path = os.path.join(config["output_dir"], "final_adapter")
        trained_model.save_pretrained(adapter_path)
        print(f"Adaptador LoRA salvo em: {adapter_path}")
        
        # Libera memória
        del trained_model, lora_model_trainer
        torch.cuda.empty_cache()

        # --- FASE 3: Avaliação na Tarefa-Alvo (Text-to-SQL) ---
        print(f"\n[{run_name}] Avaliando o desempenho em Text-to-SQL...")
        
        # Carrega o modelo fine-tuned para avaliação
        ft_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, 
            quantization_config=quantization_config, 
            device_map="auto"
        )
        ft_model.load_adapter(adapter_path)
        ft_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        
        evaluator_text2sql = EvaluateLLM(
            eval_dataset=spider_eval_dataset,
            model=ft_model,
            tokenizer=ft_tokenizer,
            database_path=SPIDER_DB_PATH
        )
        await evaluator_text2sql.evaluate_accuracy()
        # A acurácia será impressa pelo DeepEval. Anote para o relatório.

        # --- FASE 4: Análise de Regressão de Capacidade (MMLU) ---
        print(f"\n[{run_name}] Avaliando a regressão de capacidade no MMLU...")

        # O MMLUEvaluator carrega o modelo com o adaptador
        ft_mmlu_evaluator = MMLUEvaluator(model_path=BASE_MODEL_ID, adapter_path=adapter_path, seed=SEED)
        ft_mmlu_accuracy, ft_mmlu_by_category = await ft_mmlu_evaluator.run_evaluation()

        print(f"\n[RESULTADO] Acurácia MMLU (Fine-Tuned - {run_name}): {ft_mmlu_accuracy:.2f}%")
        print(f"  - Por Categoria: {ft_mmlu_by_category}")
        
        # Cálculo da Regressão
        regression = calculate_regression(base_mmlu_accuracy, ft_mmlu_accuracy)
        print(f"\n[ANÁLISE DE REGRESSÃO - {run_name}]")
        print(f"  - Variação Percentual na Acurácia Geral do MMLU: {regression:.2f}%")

        for category in base_mmlu_by_category:
            base_acc = base_mmlu_by_category.get(category, 0)
            ft_acc = ft_mmlu_by_category.get(category, 0)
            cat_regression = calculate_regression(base_acc, ft_acc)
            print(f"  - Variação em '{category.upper()}': {cat_regression:.2f}%")

        # Libera memória para a próxima iteração do loop
        del ft_model, ft_tokenizer, evaluator_text2sql, ft_mmlu_evaluator
        torch.cuda.empty_cache()

    print("\n--- EXPERIMENTO CONCLUÍDO ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Ocorreu um erro crítico: {e}")