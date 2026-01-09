# LSTM - Projeto Pós Tech FIAP 

Este projeto foi desenvolvido como parte do curso Pós-Tech da FIAP e tem como objetivo a construção de um modelo de Deep Learning baseado em LSTM (Long Short-Term Memory) para análise e previsão de séries temporais.<br>

- **Dataset**: Os dados utilizados foram obtidos a partir da plataforma Yahoo Finance, amplamente utilizada para análise de ativos financeiros.

- **Deploy**: A aplicação está disponível online e pode ser acessada pelo link: [Streamlit App](https://fivemlet-fase-4.onrender.com/)<br>

## 📁 Estrutura do Projeto

```bash
5mlet_fase_4/
│
├── src/
│   ├── streamlit_app.py
│   └── app/
│       ├── __init__.py
│       ├── dados.py
│       ├── LSTM_predictor.py
│       ├── LSTM_trainer.py
│       └── models/
│
├── run_streamlit.bat
├── README.md
└── requirements.txt

```
- **`src/`**: Diretório principal que contém os códigos-fonte da aplicação e os modelos gerados.
    - **`streamlit_app.py`**: Aplicação desenvolvida em Streamlit para utilização dos modelos LSTM já treinados, permitindo a visualização de previsões.
    - **`app/`**: Diretório que concentra os módulos responsáveis pela manipulação de dados, treinamento e inferência do modelo.
        - **`__init__.py`**: Arquivo que define o diretório como um pacote Python.
        - **`dados.py`**: Responsável pela obtenção, tratamento e preparação dos dados utilizados no modelo, incluindo a coleta via Yahoo Finance.
        - **`LSTM_predictor.py`**: Responsável por definir e utilizar o modelo LSTM para previsão de séries temporais. Este módulo implementa a arquitetura do modelo, os passos de treinamento, validação e teste, além de realizar previsões a partir de sequências de dados, com suporte à normalização e desnormalização dos valores.
        - **`LSTM_trainer.py`**: Script responsável pelo treinamento do modelo LSTM, incluindo definição da arquitetura, treinamento e salvamento do modelo.
        - **`models/`**: Diretório onde ficam armazenados os modelos LSTM treinados.
- **`run_streamlit.bat`**: Script para facilitar a execução da aplicação Streamlit em ambiente Windows.
- **`README.md`**: Documento de descrição do projeto, contendo informações gerais.
- **`requirements.txt`**: Lista de dependências necessárias para executar o projeto corretamente..


## 🛠️ Como Executar o Projeto Localmente

Foi utilizado o Python 3.13

### 1. Clone o Repositório

```bash
git clone https://github.com/PatySutto/5mlet_fase_4.git
```

### 2. Crie um Ambiente Virtual

```bash
python -m venv venv
source .\venv\Scripts\activate   # No Linux: venv/bin/activate
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Como executar localmente
#### Windowns
```
run_streamlit.bat 
```
ou
```
streamlit run src/streamlit_app.py
```

#### Linux
```
streamlit run src/streamlit_app.py
```