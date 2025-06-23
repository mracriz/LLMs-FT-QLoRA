
# Fine-Tuning de LLMs com QLoRA para Tarefas Text-to-SQL

Este repositório contém um pipeline completo para o fine-tuning de Modelos de Linguagem Grandes (LLMs) utilizando a técnica de QLoRA, com foco na tarefa de conversão de texto para SQL (Text-to-SQL). O projeto utiliza o modelo `openlm-research/open_llama_3b_v2` e o dataset [Spider](https://yale-lily.github.io/spider).

Além de treinar e avaliar o desempenho na tarefa específica, o projeto também realiza umaálise de regressão de capacidade, medindo o impacto do fine-tuning nas habilidades de conhecimento geral do modelo através do benchmark MMLU.

## ✨ Funcionalidades

* **Avaliação de Baseline**: Mede a performance do modelo base (`openlm-research/open_llama_3b_v2`) tanto na tarefa Text-to-SQL (com a métrica de acurácia de execução) quanto no benchmark de conhecimento geral MMLU.
* **Pré-processamento de Dados**: Formata e prepara o dataset Spider para o treinamento no formato de chat, adequado para modelos de instrução.
* **Treinamento com QLoRA**: Realiza o fine-tuning do LLM de forma eficiente em termos de memória, utilizando a técnica QLoRA e o `SFTTrainer` da biblioteca TRL.
* **Análise Pós-Treinamento**: Avalia o modelo após o fine-tuning para quantificar a melhoria na tarefa de Text-to-SQL.
* **Análise de Regressão**: Compara a performance do modelo treinado com o modelo base no benchmark MMLU para identificar possíveis perdas de capacidade em domínios de conhecimento geral.

## ⚙️ Setup e Instalação

Siga os passos abaixo para configurar o ambiente de execução.

### 1. Clone o Repositório
```bash
git clone [https://github.com/seu-usuario/LLMs-FT-QLoRA.git](https://github.com/seu-usuario/LLMs-FT-QLoRA.git)
cd LLMs-FT-QLoRA
````

### 2\. Baixe o Dataset Spider

**É necessário baixar o dataset Spider, que contém os bancos de dados utilizados para o treinamento e avaliação.**

Acesse o site oficial do [Spider](https://yale-lily.github.io/spider), baixe o arquivo `spider.zip` (que inclui os bancos de dados) e renomeie-o para `spider_data.zip`. Em seguida, **coloque o arquivo `spider_data.zip` na pasta raiz do projeto.** O script `main.py` irá descompactá-lo automaticamente na primeira execução.

### 3\. Instale as Dependências

Este projeto utiliza as dependências listadas no arquivo `requirements.txt`. Para instalá-las, execute o comando:

```bash
pip install -r requirements.txt
```

O arquivo já inclui o link extra para o PyTorch, garantindo a compatibilidade com ambientes CUDA mais recentes, como o do Google Colab.

## 🚀 Como Executar

Com o ambiente configurado e o arquivo `spider_data.zip` na raiz do projeto, você pode executar todo o pipeline de treinamento e avaliação com um único comando:

```bash
python main.py
```

O script cuidará de todas as etapas: avaliação do baseline, pré-processamento, treinamento com diferentes hiperparâmetros e a avaliação final.

## ☁️ Executando no Google Colab

Para facilitar a execução utilizando os recursos de GPU do Google Colab, utilize o notebook **`run_google_colab.ipynb`**. Ele contém os passos necessários para configurar o ambiente e executar o projeto na nuvem.

## 🔧 Configuração

As principais configurações do experimento podem ser ajustadas diretamente no início do arquivo `main.py`:

  * `BASE_MODEL_ID`: O identificador do modelo base no Hugging Face Hub.
  * `SAMPLE_SIZE_EVAL` e `SAMPLE_SIZE_TRAIN`: Permitem definir um número menor de amostras para testes rápidos. Defina-os como `None` para usar os datasets completos.
  * `TRAINING_CONFIGS`: Uma lista de dicionários onde você pode especificar diferentes configurações de hiperparâmetros (como `learning_rate` e `num_train_epochs`) para cada execução do treinamento.

## 📂 Estrutura do Projeto

```
.
├── custom_metrics/
│   └── execution_accuracy.py   # Métrica customizada para acurácia de execução SQL
├── results/
│   └── ...                     # Pasta onde os adaptadores treinados são salvos
├── scripts/
│   ├── evaluate.py             # Módulos de avaliação (MMLU, Text-to-SQL)
│   ├── pre_proc.py             # Módulo de pré-processamento de dados
│   └── training.py             # Classe para o treinamento do modelo com LoRA
├── main.py                     # Script principal para execução do pipeline
├── requirements.txt            # Lista de dependências do projeto
├── run_google_colab.ipynb      # Notebook para execução no Google Colab
└── spider_data.zip             # (Requerido) Dataset Spider a ser baixado pelo usuário
```

```
```
