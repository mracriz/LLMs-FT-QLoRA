from transformers import AutoModelForCausalLM, AutoTokenizer

class PreprocessData:
    def __init__(self, dataset):
        self.dataset = dataset

    def format_example(self, example, current_tokenizer, model_id_finetune, instruction_name='question', response_name='query'):
        """
        Função para pré-processar um único exemplo do dataset Spider.
        Mapeia 'question' para 'instruction' e 'query' para 'response' e formata para o chat template.

        Args:
            example (dict): Um dicionário representando um único exemplo do dataset Spider.
            current_tokenizer: O tokenizer carregado (AutoTokenizer).
            model_id_finetune (str): O ID do modelo sendo fine-tuned (ex: "meta-llama/Llama-3-8B-Instruct").

        Returns:
            dict: Um dicionário com uma chave 'text' contendo o prompt formatado.
        """
        # Mapeia as colunas do Spider para instruction e response
        instruction = example[instruction_name]
        response = example[response_name] # A saída desejada é a query SQL

        # Crie a lista de mensagens no formato de chat
        # Use 'model' para Gemma e 'assistant' para Llama/Phi-3
        if "gemma" in model_id_finetune.lower():
            messages = [
                {"role": "user", "content": instruction},
                {"role": "model", "content": response} # Gemma usa 'model'
            ]
        else: # Para Llama, Mistral, Phi-3, e a maioria dos modelos instruct
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response}
            ]

        # Use o tokenizer.apply_chat_template para formatar a string final
        # tokenize=False: retorna a string sem tokenizar
        # add_generation_prompt=False: não adiciona um prompt vazio para a resposta do assistente (já temos a resposta)
        try:
            formatted_text = current_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as e:
            print(f"Erro ao aplicar chat template para {model_id_finetune}: {e}")
            # Fallback para um formato simples se o chat template falhar (não recomendado para modelos instruct)
            formatted_text = f"### Pergunta:\n{instruction}\n\n### SQL:\n{response}"

        return {"text": formatted_text}

    def preprocess_example(self, tokenizer, MODEL_NAME):
        print("Pré-processando o dataset de treinamento para o formato de chat...")
        # Use .map() para aplicar a função a cada exemplo do dataset
        processed_dataset = self.dataset.map(
            lambda x: self.format_example(x, tokenizer, MODEL_NAME),
            num_proc=4, # Opcional: use múltiplos processos para acelerar
            remove_columns= self.dataset.column_names # Remove as colunas originais, mantendo apenas 'text'
        )

        return processed_dataset

