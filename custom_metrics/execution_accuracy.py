import sqlite3
from typing import List, Tuple, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import os

class ExecutionAccuracy(BaseMetric):
    """
    Uma métrica customizada para DeepEval que avalia a acurácia de execução de
    queries SQL geradas por um LLM.
    """
    def __init__(self, name: str = "Execution Accuracy", threshold: float = 0.5,
                 model_db_path: str = "database"):
        super().__init__(name, threshold)
        self.model_db_path = model_db_path

    def _execute_sql(self, db_id: str, sql_query: str) -> List[Tuple[Any, ...]]:
        db_path = os.path.join(self.model_db_path, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"Erro: Arquivo do banco de dados não encontrado para '{db_id}' em '{db_path}'")
            raise FileNotFoundError(f"Database file not found for {db_id} at {db_path}")

        conn = None
        results = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Erro ao executar SQL para DB '{db_id}': {e} - Query: '{sql_query}'")
            return []
        finally:
            if conn:
                conn.close()
        return results

    def measure(self, test_case: LLMTestCase) -> float:
        # --- MUDANÇA PRINCIPAL AQUI ---
        # Extrai o db_id do contexto, que agora é uma lista de strings.
        if not test_case.context or not isinstance(test_case.context, list) or len(test_case.context) == 0:
            raise ValueError("LLMTestCase 'context' deve ser uma lista contendo o db_id como primeiro elemento.")
        
        db_id = test_case.context[0]
        # --- FIM DA MUDANÇA ---

        generated_sql = test_case.actual_output
        ground_truth_sql = test_case.expected_output

        print(f"Avaliando para db_id: {db_id}")

        try:
            generated_results = self._execute_sql(db_id, generated_sql)
            ground_truth_results = self._execute_sql(db_id, ground_truth_sql)
        except FileNotFoundError:
            self.score = 0.0
            self.success = False
            self.reason = f"Database file not found for db_id: {db_id}"
            return self.score
        except Exception as e:
            self.score = 0.0
            self.success = False
            self.reason = f"Error executing query: {e}"
            return self.score

        is_match = set(generated_results) == set(ground_truth_results)

        self.score = 1.0 if is_match else 0.0
        self.success = is_match
        self.reason = "Execution results match." if is_match else "Execution results do not match."

        return self.score