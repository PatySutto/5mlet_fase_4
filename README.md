# LSTM - Projeto da Pós Tech FIAP 

Este projeto desenvolvido como parte do curso de Pós-Tech da FIAP e tem como objetivo construir um modelo LSTM.<br>

- **Dataset**: A base de dados utilizada neste projeto foi a Yahoo Finance.

- **Deploy**:<br>
Você pode testar a aplicação online neste link: [Streamlit App](https://fivemlet-f3-streamlit.onrender.com/)<br>

## 📁 Estrutura do Projeto

```bash
5mlet_postech/
├── src/
│   ├── streamlit_app.py
│   └── app/
│       ├── __init__.py
│       ├── dados.py
│       ├── LSTM_predictor.py
│       ├── LSTM_trainer.py
│       └── models/
├── run_streamlit.bat
├── README.md
└── requirements.txt

```
- **`dataset/`**: Diretório que contém as bases de dados usadas.
  - **`heart-failure-tratado.csv`**: Base de dados tratada utilizada para o treinamento do modelo.
  - **`heart-failure.csv`**: Base de dados original.
- **`src/`**: Diretório que reúne os modelos de Machine Learning e os scripts referentes à análise exploratória dos dados.
    - **`analise_exploratoria.ipynb`**: Notebook destinado à realização e visualização da análise exploratória dos dados.
    - **`modelos ML/`**: Diretório que contém os modelos de Machine Learning.
        - **`arvore_decisao.ipynb`**: Notebook com o modelo de Árvore de Decisão.
        - **`knn.ipynb`**: Notebook com o modelo de K-Nearest Neighbors (KNN).
        - **`regressão_logistica.ipynb`**: Notebook com o modelo de Regressão Logística.
        - **`support_vector_machine.ipynb`**: Notebook com o modelo de Support Vector Machine (SVM).
        - **`xgboost.ipynb`**: Notebook com o modelo de XGBoost.
- **`modelo_regressao_logistica.pkl`**: Modelo de Regressão Logística já treinado.
- **`postech_fase_3.pdf`**: Documento com o resumo do que foi feito e os resultados de cada modelo.
- **`README.md`**: Documentação do projeto.
- **`requirements.txt`**: Lista de dependências do projeto.


## 🛠️ Como Executar o Projeto Localmente

### 1. Clone o Repositório

```bash
git clone https://github.com/PatySutto/5mlet_fase_3.git
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

### 4. Execute os modelos

```
streamlit run src/streamlit_app.py
Escolha o modelo desejado e clique em "Run All".
```