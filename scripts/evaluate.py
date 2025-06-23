# --- Bibliotecas Padrão do Python ---
import asyncio
import random
from typing import Dict, List, Tuple

# --- Bibliotecas de Terceiros (Instaladas com Pip) ---
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from tqdm import tqdm # Importa a biblioteca tqdm

# --- Módulos do Seu Próprio Projeto ---
from custom_metrics.execution_accuracy import ExecutionAccuracy

class EvaluateLLM():
    def __init__(self, eval_dataset, model, tokenizer, database_path):
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.custom_metric = ExecutionAccuracy(model_db_path=database_path)

    def prepare_spider_test_cases(self):
        test_cases = []
        print("Gerando previsões do modelo para o Spider dev split...")
        for example in tqdm(self.eval_dataset, desc="Avaliando Text-to-SQL (Fine-Tuned)"):
            question = example['question']
            ground_truth_sql = example['query']
            db_id = example['db_id']

            messages = [{"role": "user", "content": question}]
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=100, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id
                )
            generated_sql = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()
            
            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=generated_sql,
                    expected_output=ground_truth_sql,
                    context=[str(db_id)]
                )
            )
        print(f"Preparados {len(test_cases)} casos de teste.")
        return test_cases

    async def evaluate_accuracy(self):
        print("Iniciando avaliação com DeepEval e métrica customizada...")
        prepared_test_cases = self.prepare_spider_test_cases()
        evaluate(
            test_cases=prepared_test_cases,
            metrics=[self.custom_metric],
            show_progress=False
        )
        print("Avaliação concluída.")

class MMLUEvaluator:
    def __init__(self, model_path: str, seed: int = 42, adapter_path: str | None = None):
        self.model_path = model_path
        self.seed = seed
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Inicializando avaliador para o modelo: {self.model_path}")
        if self.adapter_path:
            print(f"  - Com adaptador: {self.adapter_path}")
        
        self._load_model_and_tokenizer()
        self._prepare_dataset()

    def _load_model_and_tokenizer(self):
        print("Carregando modelo e tokenizador...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        
        if self.tokenizer.chat_template is None:
            print(f"tokenizer.chat_template não encontrado para {self.model_path}. Definindo manualmente...")
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<s>[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}"
                " {{ message['content'] }}</s>"
                "{% endif %}"
                "{% endfor %}"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16
        )

        if self.adapter_path:
            print(f"Carregando adaptador LoRA de: {self.adapter_path}")
            self.model.load_adapter(self.adapter_path)
            print("Adaptador carregado com sucesso.")

        print("Modelo e tokenizador carregados.")

    def _prepare_dataset(self):
        print("Preparando o dataset MMLU...")
        self.categories = {
            "STEM": "college_computer_science", 
            "Humanities": "philosophy",          
            "Social Sciences": "high_school_macroeconomics"        
        }
        
        self.evaluation_suite: Dict[str, List] = {}
        self.few_shot_examples: Dict[str, List] = {}

        for suite_name, hf_subset in self.categories.items():
            dataset = load_dataset("cais/mmlu", hf_subset, trust_remote_code=True)
            random.seed(self.seed)
            test_samples = list(dataset['test'])
            self.evaluation_suite[suite_name] = random.sample(test_samples, 50)
            dev_samples = list(dataset['dev'])
            self.few_shot_examples[suite_name] = random.sample(dev_samples, 4)
        print("Dataset preparado.")

    def _create_prompt(self, question_data: Dict, examples: List[Dict]) -> str:
        choices = ['A', 'B', 'C', 'D']
        prompt = "The following are multiple choice questions (with answers).\n\n"
        
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex['choices']):
                prompt += f"{choices[i]}. {choice}\n"
            prompt += f"Answer: {choices[ex['answer']]}\n\n"
            
        prompt += f"Question: {question_data['question']}\n"
        for i, choice in enumerate(question_data['choices']):
            prompt += f"{choices[i]}. {choice}\n"
        prompt += "Answer:"
        
        return prompt

    def run_evaluation(self) -> Tuple[float, Dict[str, float]]:
        print(f"\nIniciando avaliação para o modelo: {self.model_path}")
        category_results = {s: {"correct": 0, "total": 0} for s in self.categories.keys()}

        for suite_name, questions in self.evaluation_suite.items():
            for question in tqdm(questions, desc=f"Avaliando MMLU - {suite_name}"):
                prompt = self._create_prompt(question, self.few_shot_examples[suite_name])
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=5, pad_token_id=self.tokenizer.eos_token_id)
                
                generated_text = self.tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                predicted_answer = generated_text.strip().upper()
                
                choices = ['A', 'B', 'C', 'D']
                correct_answer = choices[question['answer']]
                
                if predicted_answer.startswith(correct_answer):
                    category_results[suite_name]["correct"] += 1
                
                category_results[suite_name]["total"] += 1

        total_correct = sum(d["correct"] for d in category_results.values())
        total_questions = sum(d["total"] for d in category_results.values())
        
        overall_accuracy = (total_correct / total_questions) * 100 if total_questions > 0 else 0
        category_accuracies = {
            s: (d["correct"] / d["total"]) * 100 if d["total"] > 0 else 0 
            for s, d in category_results.items()
        }
        
        return overall_accuracy, category_accuracies

def calculate_regression(base_acc: float, ft_acc: float) -> float:
    if base_acc == 0:
        return float('inf')
    return ((ft_acc - base_acc) / base_acc) * 100


class BaselineEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.spider_train_dataset = load_dataset("spider", split="train")

    def _get_few_shot_examples(self) -> list:
        indices = [1, 10, 20]
        examples = []
        for i in indices:
            sample = self.spider_train_dataset[i]
            examples.append({
                "question": sample["question"],
                "query": sample["query"]
            })
        return examples

    def _create_prompt(self, question: str, examples: list) -> str:
        prompt = "Gere uma consulta SQL que responda à seguinte pergunta com base no esquema do banco de dados.\n\n"
        for ex in examples:
            prompt += f"-- Pergunta: {ex['question']}\n"
            prompt += f"SQL: {ex['query']}\n\n"
        prompt += f"-- Pergunta: {question}\n"
        prompt += "SQL:"
        return prompt
        
    def run_evaluation(self, eval_dataset) -> list:
        print("--- INICIANDO AVALIAÇÃO DE BASELINE EM TEXT-TO-SQL (FEW-SHOT) ---")
        few_shot_examples = self._get_few_shot_examples()
        print(f"Usando {len(few_shot_examples)} exemplos few-shot fixos para o prompt.")
        test_cases = []
        
        for item in tqdm(eval_dataset, desc="Avaliando baseline Text-to-SQL"):
            question = item['question']
            ground_truth_sql = item['query']
            db_id = item['db_id']

            prompt_text = self._create_prompt(question, few_shot_examples)
            messages = [{"role": "user", "content": prompt_text}]
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=128, num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id, do_sample=False
                )
            
            generated_sql = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            ).strip()

            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=generated_sql,
                    expected_output=ground_truth_sql,
                    context=[str(db_id)]
                )
            )
        
        print(f"Avaliação de baseline concluída. {len(test_cases)} consultas SQL geradas.")
        if test_cases:
            print("\nExemplo de Geração (Baseline):")
            print(f"  Pergunta: {test_cases[0].input}")
            print(f"  SQL Gerado: {test_cases[0].actual_output}")
            print(f"  SQL Correto: {test_cases[0].expected_output}")

        return test_cases