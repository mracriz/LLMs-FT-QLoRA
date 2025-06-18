import pytest
import torch
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.metrics import BaseMetric
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import os
import asyncio # Add this import

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
        for example in self.eval_dataset:
            question = example['question']
            ground_truth_sql = example['query']
            db_id = example['db_id'] # O db_id é crucial para conectar ao banco correto

            # Formatar o prompt para o modelo (o mesmo formato usado no fine-tuning)
            messages = [{"role": "user", "content": question}]
            # Adicionando add_generation_prompt=True para que o modelo comece a gerar a resposta
            input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

            with torch.no_grad(): # Use torch.no_grad() para inferência
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100, # Adjust as needed for SQL length
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_sql = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], # Slice to get only the generated part
                skip_special_tokens=True
            ).strip()

            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=generated_sql,
                    expected_output=ground_truth_sql,
                    # Ensure db_id is a string, as DeepEval context typically expects string values
                    context={"db_id": str(db_id)}
                )
            )
        print(f"Preparados {len(test_cases)} casos de teste.")
        return test_cases # Crucial: return the list of test cases

    async def evaluate_accuracy(self): # Make this method async
        print("Iniciando avaliação com DeepEval e métrica customizada...")
        
        # 1. Prepare the LLMTestCases by generating predictions
        prepared_test_cases = self.prepare_spider_test_cases()

        # 2. Run the evaluation with DeepEval
        await evaluate(
            test_cases=prepared_test_cases, # Pass the generated test cases
            metrics=[self.custom_metric]
        )
        print("Avaliação concluída.")

