from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig # Potentially needed if you pass quantization directly
)
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, TrainingArguments # Make sure to import TrainingArguments if it's what you pass as training_arguments

class LLM_LoRA_Model:
    def __init__(self, train_data, eval_data=None):
        self.model_name = None
        self.model = None
        self.tokenizer = None
        self.train_data = train_data
        self.eval_data = eval_data

    def set_model(self, MODEL_NAME, quantization=None):
        print(f"Carregando modelo base: {MODEL_NAME}")
        self.model_name = MODEL_NAME

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization,
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2" # Uncomment if you have Flash Attention 2 installed and want to use it
        )
        print("Modelo carregado.")

        self.model = model
        self.set_tokenizer() # Call set_tokenizer after setting the model

    def set_tokenizer(self):
        if self.model is None:
            print("Modelo não definido. Utilize set_model() primeiro.")
            return

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # 1. Configurar padding token se não estiver definido
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # It's a good practice to also ensure the model can handle the new pad_token_id
            # self.model.config.pad_token_id = tokenizer.eos_token_id # This might be done automatically by TRL/PEFT, but good to be aware

        # 2. Definir o chat_template explicitamente se não estiver configurado
        if tokenizer.chat_template is None:
            print(f"tokenizer.chat_template não encontrado para {self.model_name}. Definindo manualmente...")
            # This template is for Llama 2 Instruct.
            # If targeting Llama 3, its template is different and usually auto-loaded.
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<s>[INST] {{ message['content'] }} [/INST]"
                "{% elif message['role'] == 'assistant' %}"
                " {{ message['content'] }}</s>"
                "{% endif %}"
                "{% endfor %}"
            )
        print("Tokenizer carregado e chat_template configurado se necessário.")
        self.tokenizer = tokenizer

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def train(self, lora_config: LoraConfig, training_arguments: TrainingArguments): # Added type hints for clarity
        torch.cuda.empty_cache() # Clear cache before training

        print("Iniciando a configuração do SFTTrainer...")
        trainer = SFTTrainer(
            model=self.model,
            args=training_arguments,
            train_dataset=self.train_data,
            eval_dataset=self.eval_data,
            tokenizer=self.tokenizer, # CORRECTED: Use 'tokenizer' argument
            peft_config=lora_config,
            dataset_text_field="text", # Specify the column name containing the formatted text
                                       # This is crucial if your dataset only has a 'text' column after preprocessing
        )

        print("Iniciando o treinamento QLoRA...")
        try:
            trainer.train()
            print("Treinamento concluído!")
        except Exception as e:
            print(f"Ocorreu um erro durante o treinamento: {e}")
            # You might want to re-raise the exception or handle it more gracefully
            # depending on your error handling strategy.
            raise # Re-raise the exception if it's critical

        return trainer.model