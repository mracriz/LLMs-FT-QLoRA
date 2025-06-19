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

# Importa as classes dos seus módulos
from scripts.pre_proc import PreprocessData
from scripts.training import LLM_LoRA_Model
from scripts.evaluate import EvaluateLLM, MMLUEvaluator, calculate_regression

# --- Configuração do Experimento ---
# Defina aqui todos os parâmetros para garantir a reprodutibilidade.

# Semente para reprodutibilidade, conforme exigido nas especificações
SEED = 42

# Modelo base a ser utilizado. Ex: "meta-llama/Llama-3-8B-Instruct"
#BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct" 
BASE_MODEL_ID = "openlm-research/open_llama_3b_v2"

# Caminho para a pasta 'database' do Spider, necessária para a ExecutionAccuracy
SPIDER_DB_PATH = "path/to/your/spider/database" 

# Configurações de QLoRA (quantização)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Configuração base do LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"], # Módulos alvo para Llama 3
    task_type="CAUSAL_LM",
)

# Hiperparâmetros de treinamento para as duas execuções obrigatórias
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
    """
    Função principal que orquestra todo o pipeline experimental.
    """
    set_seed(SEED)

    # --- Avaliação de Baseline do Modelo Base ---
    print("--- INICIANDO AVALIAÇÃO DE BASELINE DO MODELO BASE ---")
    
    print("\n[MMLU] Avaliando o modelo base para medir a capacidade de generalização inicial...")
    base_mmlu_evaluator = MMLUEvaluator(model_path=BASE_MODEL_ID, seed=SEED)
    base_mmlu_accuracy, base_mmlu_by_category = base_mmlu_evaluator.run_evaluation()
    print(f"\n[RESULTADO] Acurácia MMLU (Base): {base_mmlu_accuracy:.2f}%")
    print(f"  - Por Categoria: {base_mmlu_by_category}")

    print("\n[Spider] A avaliação de baseline em Text-to-SQL com prompt few-shot precisa ser implementada.")
    # Aqui iria o código para a avaliação de baseline em Text-to-SQL.
    # Ex: base_text2sql_accuracy = evaluate_base_model_on_spider(...)
    # print(f"\n[RESULTADO] Acurácia Text-to-SQL (Base): {base_text2sql_accuracy:.2f}%")

    # --- Carregamento e Pré-processamento dos Dados de Fine-Tuning ---
    print("\n--- CARREGANDO E PRÉ-PROCESSANDO DADOS PARA FINE-TUNING ---")
    
    spider_dataset = load_dataset("spider", split="train")
    spider_eval_dataset = load_dataset("spider", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    
    preprocessor = PreprocessData(dataset=spider_dataset)
    # O método de pré-processamento precisa receber o tokenizer.
    processed_train_dataset = spider_dataset.map(
    lambda x: preprocessor.preprocess_example(x, tokenizer, BASE_MODEL_ID),
    num_proc=4)
    print("Dataset de treinamento pré-processado com sucesso.")

    # --- Loop de Experimentos de Fine-Tuning e Avaliação ---
    for i, config in enumerate(TRAINING_CONFIGS):
        run_name = f"Execução {i+1} (LR={config['learning_rate']}, Epochs={config['num_train_epochs']})"
        print(f"\n--- INICIANDO {run_name.upper()} ---")

        # --- Execução do Fine-Tuning ---
        print(f"\n[{run_name}] Iniciando fine-tuning com QLoRA...")
        
        training_args = TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            learning_rate=config['learning_rate'],
            num_train_epochs=config['num_train_epochs'],
            output_dir=config['output_dir'],
            logging_steps=50,
            save_strategy="epoch",
            seed=SEED,
        )
        
        lora_model = LLM_LoRA_Model(train_data=processed_train_dataset)
        lora_model.set_model(BASE_MODEL_ID, quantization=quantization_config)
        lora_model.tokenizer = tokenizer
        trained_model = lora_model.train(lora_config, training_args)

        adapter_path = os.path.join(config["output_dir"], "final_adapter")
        trained_model.save_pretrained(adapter_path)
        print(f"Adaptador LoRA salvo em: {adapter_path}")

        # --- Avaliação na Tarefa-Alvo (Text-to-SQL) ---
        print(f"\n[{run_name}] Avaliando o desempenho em Text-to-SQL...")
        
        del trained_model
        torch.cuda.empty_cache()

        ft_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, quantization_config=quantization_config, device_map="auto")
        ft_model.load_adapter(adapter_path)
        
        evaluator_text2sql = EvaluateLLM(
            eval_dataset=spider_eval_dataset,
            model=ft_model,
            tokenizer=tokenizer,
            database_path=SPIDER_DB_PATH
        )
        await evaluator_text2sql.evaluate_accuracy()

        # --- Análise de Regressão de Capacidade (MMLU) ---
        print(f"\n[{run_name}] Avaliando a regressão de capacidade no MMLU...")

        ft_mmlu_evaluator = MMLUEvaluator(model_path=BASE_MODEL_ID, adapter_path=adapter_path, seed=SEED)
        ft_mmlu_accuracy, ft_mmlu_by_category = ft_mmlu_evaluator.run_evaluation()

        print(f"\n[RESULTADO] Acurácia MMLU (Fine-Tuned - {run_name}): {ft_mmlu_accuracy:.2f}%")
        print(f"  - Por Categoria: {ft_mmlu_by_category}")
        
        regression = calculate_regression(base_mmlu_accuracy, ft_mmlu_accuracy)
        print(f"\n[ANÁLISE DE REGRESSÃO - {run_name}]")
        print(f"  - Variação Percentual na Acurácia Geral do MMLU: {regression:.2f}%")

        for category in base_mmlu_by_category:
            base_acc = base_mmlu_by_category.get(category, 0)
            ft_acc = ft_mmlu_by_category.get(category, 0)
            cat_regression = calculate_regression(base_acc, ft_acc)
            print(f"  - Variação em '{category.upper()}': {cat_regression:.2f}%")

        del ft_model, ft_mmlu_evaluator
        torch.cuda.empty_cache()

    print("\n--- EXPERIMENTO CONCLUÍDO ---")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Ocorreu um erro crítico: {e}")