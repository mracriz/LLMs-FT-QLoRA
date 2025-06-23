import os
import random
import numpy as np
import torch
import asyncio
import zipfile
import shutil
from pathlib import Path
from datasets import load_dataset
from transformers import (
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

# --- Configuração do Experimento ---
SEED = 42
BASE_MODEL_ID = "openlm-research/open_llama_3b_v2"
SPIDER_DB_PATH = "spider_data/database"

# --- Configuração de Amostragem para Teste Rápido ---
SAMPLE_SIZE_EVAL = None
SAMPLE_SIZE_TRAIN = None

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
)
TRAINING_CONFIGS = [
    {"learning_rate": 2e-4, "num_train_epochs": 1, "output_dir": "./results/run_1_lr_2e-4_epochs_1"},
    {"learning_rate": 1e-4, "num_train_epochs": 2, "output_dir": "./results/run_2_lr_1e-4_epochs_2"}
]

def setup_spider_database():
    """
    Verifica e extrai o banco de dados do Spider a partir de um arquivo zip local.
    """
    db_dir = Path(SPIDER_DB_PATH)
    zip_path = Path("spider_data.zip")

    if db_dir.exists() and any(db_dir.iterdir()):
        print(f"Bancos de dados do Spider já encontrados em '{db_dir}'. Pulando extração.")
        return

    print(f"Diretório de bancos de dados '{db_dir}' não encontrado ou está vazio.")

    if not zip_path.exists():
        print(f"--- ERRO ---")
        print(f"O arquivo '{zip_path}' não foi encontrado na pasta do projeto.")
        raise FileNotFoundError(f"'{zip_path}' não encontrado. O script não pode continuar.")

    print(f"Arquivo '{zip_path}' encontrado. Iniciando descompactação...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print(f"Arquivo '{zip_path}' descompactado com sucesso.")
    except Exception as e:
        print(f"Ocorreu um erro crítico durante a descompactação: {e}")
        raise

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

async def main():
    """
    Função principal que executa o pipeline de avaliação e treinamento.
    Assume que os bancos de dados já foram preparados.
    """
    set_seed(SEED)
    
    print("--- INICIANDO AVALIAÇÃO DE BASELINE DO MODELO BASE ---")

    # Avaliação de Generalização (MMLU)
    print("\n[MMLU Baseline] Avaliando o modelo base...")
    base_mmlu_evaluator = MMLUEvaluator(model_path=BASE_MODEL_ID, seed=SEED)
    base_mmlu_accuracy, base_mmlu_by_category = base_mmlu_evaluator.run_evaluation()
    print(f"\n[RESULTADO] Acurácia MMLU (Base): {base_mmlu_accuracy:.2f}%")
    print(f"  - Por Categoria: {base_mmlu_by_category}")

    # Carrega e amostra o dataset de avaliação
    spider_eval_dataset = load_dataset("spider", split="validation")
    if SAMPLE_SIZE_EVAL is not None:
        print(f"--- ATENÇÃO: Usando uma amostra de {SAMPLE_SIZE_EVAL} exemplos para a avaliação ---")
        spider_eval_dataset = spider_eval_dataset.select(range(SAMPLE_SIZE_EVAL))

    print("\n[Spider Baseline] Avaliando o modelo base em Text-to-SQL...")
    baseline_text2sql_evaluator = BaselineEvaluator(
        model=base_mmlu_evaluator.model, tokenizer=base_mmlu_evaluator.tokenizer
    )
    baseline_test_cases = baseline_text2sql_evaluator.run_evaluation(spider_eval_dataset)
    
    print("\n[Spider Baseline] Calculando Acurácia de Execução para o baseline...")
    execution_metric = ExecutionAccuracy(model_db_path=SPIDER_DB_PATH)
    evaluate(test_cases=baseline_test_cases, metrics=[execution_metric])
    
    del base_mmlu_evaluator, baseline_text2sql_evaluator, baseline_test_cases
    torch.cuda.empty_cache()

    print("\n--- CARREGANDO E PRÉ-PROCESSANDO DADOS PARA FINE-TUNING ---")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
    spider_train_dataset = load_dataset("spider", split="train")
    
    if SAMPLE_SIZE_TRAIN is not None:
        print(f"--- ATENÇÃO: Usando uma amostra de {SAMPLE_SIZE_TRAIN} exemplos para o treinamento ---")
        spider_train_dataset = spider_train_dataset.select(range(SAMPLE_SIZE_TRAIN))

    preprocessor = PreprocessData(dataset=spider_train_dataset)
    processed_train_dataset = spider_train_dataset.map(
        lambda x: preprocessor.format_example(x, tokenizer, BASE_MODEL_ID),
        num_proc=os.cpu_count(), remove_columns=spider_train_dataset.column_names
    )
    print("Dataset de treinamento pré-processado com sucesso.")
    del tokenizer, spider_train_dataset
    torch.cuda.empty_cache()

    for i, config in enumerate(TRAINING_CONFIGS):
        run_name = f"Execução {i+1} (LR={config['learning_rate']}, Epochs={config['num_train_epochs']})"
        print(f"\n--- INICIANDO {run_name.upper()} ---")

        training_args = SFTConfig(
            output_dir=config['output_dir'], per_device_train_batch_size=2,
            gradient_accumulation_steps=4, learning_rate=config['learning_rate'],
            num_train_epochs=config['num_train_epochs'], max_seq_length=1024,
            gradient_checkpointing=True, optim="paged_adamw_8bit",
            logging_steps=10, save_strategy="epoch", seed=SEED,
            report_to="none", dataset_text_field="text"
        )
        
        lora_model_trainer = LLM_LoRA_Model(train_data=processed_train_dataset)
        lora_model_trainer.set_model(BASE_MODEL_ID, quantization=quantization_config)
        
        trained_model = lora_model_trainer.train(
            lora_config=lora_config,
            training_arguments=training_args
        )
        adapter_path = os.path.join(config["output_dir"], "final_adapter")
        trained_model.save_pretrained(adapter_path)
        del trained_model, lora_model_trainer
        torch.cuda.empty_cache()

        print(f"\n[{run_name}] Avaliando o desempenho em Text-to-SQL...")
        ft_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID, quantization_config=quantization_config, device_map="auto"
        )
        ft_model.load_adapter(adapter_path)
        ft_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        if ft_tokenizer.chat_template is None:
            print(f"Definindo chat_template manualmente para o tokenizador de avaliação...")
            ft_tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<s>[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}"
                " {{ message['content'] }}</s>"
                "{% endif %}"
                "{% endfor %}"
            )
        
        evaluator_text2sql = EvaluateLLM(
            eval_dataset=spider_eval_dataset, model=ft_model,
            tokenizer=ft_tokenizer, database_path=SPIDER_DB_PATH
        )
        await evaluator_text2sql.evaluate_accuracy()

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

        del ft_model, ft_tokenizer, evaluator_text2sql, ft_mmlu_evaluator
        torch.cuda.empty_cache()

    print("\n--- EXPERIMENTO CONCLUÍDO ---")

if __name__ == "__main__":
    try:
        # --- ETAPA DE PRÉ-EXECUÇÃO ---
        # Garante que os bancos de dados sejam extraídos ANTES de iniciar o processo principal.
        print("--- VERIFICANDO E PREPARANDO BANCO DE DADOS SPIDER ---")
        setup_spider_database()
        print("--- PREPARAÇÃO CONCLUÍDA. INICIANDO PROCESSO PRINCIPAL ---\n")

        # Inicia a execução principal e assíncrona do script
        asyncio.run(main())

    except Exception as e:
        print(f"Ocorreu um erro crítico: {e}")