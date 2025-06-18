import sqlite3
from typing import List, Tuple, Any
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
import os

class ExecutionAccuracy(BaseMetric):
    """
    Uma métrica customizada para DeepEval que avalia a acurácia de execução de
    queries SQL geradas por um LLM, comparando os resultados obtidos com a query
    gerada e a query ground truth em um banco de dados SQLite.
    """
    def __init__(self, name: str = "Execution Accuracy", threshold: float = 0.5,
                 model_db_path: str = "database"):
        """
        Inicializa a métrica de Acurácia de Execução.

        Args:
            name (str): O nome da métrica.
            threshold (float): O limite de sucesso para a métrica (geralmente 0.5 para 0/1).
            model_db_path (str): O caminho base para a pasta 'database' do dataset Spider,
                                  contendo os subdiretórios de cada banco de dados.
        """
        super().__init__(name, threshold)
        self.model_db_path = model_db_path

    def _execute_sql(self, db_id: str, sql_query: str) -> List[Tuple[Any, ...]]:
        """
        Método auxiliar para executar uma query SQL em um banco de dados SQLite específico.

        Args:
            db_id (str): O ID do banco de dados (nome do diretório do DB no Spider).
            sql_query (str): A query SQL a ser executada.

        Returns:
            List[Tuple[Any, ...]]: Uma lista de tuplas, representando as linhas do resultado
                                    da query. Retorna uma lista vazia em caso de erro.

        Raises:
            FileNotFoundError: Se o arquivo do banco de dados não for encontrado.
        """
        # Constrói o caminho completo para o arquivo .sqlite
        # Ex: self.model_db_path/employee_db/employee_db.sqlite
        db_path = os.path.join(self.model_db_path, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"Erro: Arquivo do banco de dados não encontrado para '{db_id}' em '{db_path}'")
            raise FileNotFoundError(f"Database file not found for {db_id} at {db_path}")

        conn = None
        cursor = None
        results = []
        try:
            # Conecta ao banco de dados SQLite
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Executa a query SQL
            cursor.execute(sql_query)
            
            # Busca todos os resultados
            results = cursor.fetchall()
        except sqlite3.Error as e:
            # Captura e imprime erros de sintaxe SQL ou outros erros de execução do SQLite
            print(f"Erro ao executar SQL para DB '{db_id}': {e} - Query: '{sql_query}'")
            return [] # Retorna uma lista vazia para indicar falha na execução
        finally:
            # Garante que a conexão com o banco de dados seja fechada
            if conn:
                conn.close()
        return results

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Calcula a acurácia de execução para um dado caso de teste.
        Compara os resultados da execução da query gerada pelo LLM com os resultados
        da execução da query ground truth.

        Args:
            test_case (LLMTestCase): O caso de teste contendo a entrada (pergunta),
                                     saída real (SQL gerado), saída esperada (SQL ground truth)
                                     e o contexto (contendo o db_id).

        Returns:
            float: O score da métrica (1.0 se os resultados de execução coincidirem, 0.0 caso contrário).
        """
        # Extrai o db_id do contexto do test_case
        # test_case.context é um dicionário, e esperamos 'db_id' como uma chave.
        db_id = test_case.context.get("db_id")

        # Verifica se o db_id está presente no contexto
        if not db_id:
            raise ValueError("LLMTestCase context must contain 'db_id' for ExecutionAccuracy to function.")
            # Para robustez, você poderia definir o score como 0 e retornar aqui,
            # mas levantar um erro é mais explícito se o db_id for mandatório.

        # Extrai as queries gerada e ground truth do test_case
        generated_sql = test_case.actual_output
        ground_truth_sql = test_case.expected_output

        # Exemplo de uso de db_id para conexão com o banco de dados
        # O db_id é essencial para localizar o arquivo SQLite correto
        print(f"Avaliando para db_id: {db_id}")

        # 1. Executa a query gerada pelo modelo
        generated_results = []
        try:
            generated_results = self._execute_sql(db_id, generated_sql)
        except FileNotFoundError:
            # Se o arquivo DB não for encontrado, a execução falha.
            self.score = 0.0
            self.success = False
            self.reason = f"Database file not found for db_id: {db_id}"
            return self.score
        except Exception as e:
            # Captura qualquer outro erro durante a execução da query gerada (ex: sintaxe inválida)
            print(f"Erro crítico na execução da query gerada: {e}")
            self.score = 0.0
            self.success = False
            self.reason = f"Error executing generated query: {e}"
            return self.score

        # 2. Executa a query ground truth
        ground_truth_results = []
        try:
            ground_truth_results = self._execute_sql(db_id, ground_truth_sql)
        except FileNotFoundError:
            # Se o arquivo DB não for encontrado, a execução falha.
            self.score = 0.0
            self.success = False
            self.reason = f"Database file not found for db_id: {db_id}"
            return self.score
        except Exception as e:
            # Captura qualquer outro erro durante a execução da query ground truth
            print(f"Erro crítico na execução da query ground truth: {e}")
            self.score = 0.0
            self.success = False
            self.reason = f"Error executing ground truth query: {e}"
            return self.score

        # 3. Compara os conjuntos de resultados
        # Converter listas de tuplas para conjuntos de tuplas para uma comparação insensível à ordem das linhas.
        # Isso é crucial porque a ordem das linhas retornadas por uma query SQL pode variar
        # dependendo do otimizador do banco de dados, a menos que uma cláusula ORDER BY seja usada.
        # NOTA: Esta comparação assume que os tipos e a ordem das colunas nos resultados são consistentes.
        is_match = set(generated_results) == set(ground_truth_results)

        # Define o score e o status de sucesso da métrica
        self.score = 1.0 if is_match else 0.0
        self.success = is_match
        self.reason = "Execution results match." if is_match else "Execution results do not match."

        return self.score