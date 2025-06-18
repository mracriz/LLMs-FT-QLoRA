import sqlite3
from typing import List, Tuple, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import os

class ExecutionAccuracy(BaseMetric):
    def __init__(self, name: str = "Execution Accuracy", threshold: float = 0.5,
                 model_db_path: str = "database"): # Caminho para a pasta 'database' do Spider
        super().__init__(name, threshold)
        self.model_db_path = model_db_path

    # Método auxiliar para executar uma query SQL e retornar os resultados
    def _execute_sql(self, db_id: str, sql_query: str) -> List[Tuple[Any, ...]]:
        db_path = os.path.join(self.model_db_path, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found for {db_id} at {db_path}")

        conn = None
        cursor = None
        results = []
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
        except sqlite3.Error as e:
            # Captura erros de sintaxe SQL ou execução
            print(f"Erro ao executar SQL para DB '{db_id}': {e} - Query: '{sql_query}'")
            return [] # Retorna vazio ou levanta exceção, dependendo da sua estratégia de tratamento de erro
        finally:
            if conn:
                conn.close()
        return results

    # Implementação do método principal da métrica
    def measure(self, test_case: LLMTestCase) -> float:
        # test_case.actual_output: a query SQL gerada pelo LLM
        # test_case.expected_output: a query SQL ground truth
        # test_case.context: usaremos para passar o db_id do Spider

        if not test_case.context or 'db_id' not in test_case.context:
            raise ValueError("LLMTestCase context must contain 'db_id' for ExecutionAccuracy.")

        db_id = test_case.context['db_id']
        generated_sql = test_case.actual_output
        ground_truth_sql = test_case.expected_output

        # Execute a query gerada pelo modelo
        try:
            generated_results = self._execute_sql(db_id, generated_sql)
        except Exception as e:
            # Qualquer erro aqui (sintaxe, runtime) significa falha
            print(f"Erro na execução da query gerada: {e}")
            self.score = 0
            self.success = False
            return self.score

        # Execute a query ground truth
        try:
            ground_truth_results = self._execute_sql(db_id, ground_truth_sql)
        except Exception as e:
            print(f"Erro na execução da query ground truth: {e}")
            self.score = 0
            self.success = False
            return self.score

        # Comparar os conjuntos de resultados (insensível à ordem das linhas) 
        # Converter listas de tuplas para conjuntos de tuplas para comparação insensível à ordem
        # Note: A ordem das colunas e tipos de dados ainda importa.
        # Se os resultados contiverem tipos de dados não hashable (como listas), pode ser necessário converter para tuplas recursivamente.
        # Para resultados de SQL, geralmente são tuplas de tipos básicos, então set() deve funcionar.
        is_match = set(generated_results) == set(ground_truth_results)

        self.score = 1.0 if is_match else 0.0
        self.success = is_match

        return self.score