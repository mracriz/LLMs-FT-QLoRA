from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer)

class LoadModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self, quantization=None):
        print(f"Carregando modelo base: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            device_map="auto", # Mapeia automaticamente para GPU se disponível
            trust_remote_code=True, # Necessário para alguns modelos como Phi-3
            # attn_implementation="flash_attention_2" # Opcional: se suportado e instalado, pode acelerar
        )
        print("Modelo carregado.")

        return model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        # 1. Configurar padding token se não estiver definido
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # Alguns modelos (como Llama) não têm pad_token. Usar eos_token é uma prática comum.
            # Para modelos que não são auto-regressivos ou que têm arquiteturas específicas, isso pode precisar de ajuste.

        # 2. Definir o chat_template explicitamente se não estiver configurado
        # Isso é essencial para modelos como o Open Llama 3B V2, que pode não ter um template padrão.
        if tokenizer.chat_template is None:
            print(f"tokenizer.chat_template não encontrado para {self.get_model}. Definindo manualmente...")
            # Este é o template padrão para modelos Llama 2 Instruct (e muitos "Llama-like")
            # Ele inclui <s> e </s> automaticamente quando tokenize=True, mas é bom tê-lo completo aqui.
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<s>[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}"
                " {{ message['content'] }}</s>" # Note o espaço antes da resposta do assistente
                "{% endif %}"
                "{% endfor %}"
            )
        print("Tokenizer carregado e chat_template configurado se necessário.")

        return tokenizer

class PreprocessData():
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
            lambda x: self.preprocess_example(x, tokenizer, MODEL_NAME),
            num_proc=4, # Opcional: use múltiplos processos para acelerar
            remove_columns= self.train_dataset.column_names # Remove as colunas originais, mantendo apenas 'text'
        )

        return processed_dataset

