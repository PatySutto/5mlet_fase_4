import os
import sys
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import streamlit as st

# Garantir que 'src' esteja no path para importar pacote `app`
SRC_DIR = os.path.abspath(os.path.dirname(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from app.dados import Dados
from app.LSTM_predictor import LSTMPredictor


def iterative_forecast(model, scaler, last_window, days, device='cpu'):
    model.to(device)
    model.eval()
    seq = last_window.copy()
    preds_norm = []

    for _ in range(days):
        x = torch.FloatTensor(seq).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x).cpu().numpy().flatten()[0]

        preds_norm.append(out)

        next_row = seq[-1].copy()
        next_row[0] = out
        seq = np.vstack([seq[1:], next_row])

    preds_norm = np.array(preds_norm)
    if scaler is not None:
        dummy = np.zeros((len(preds_norm), scaler.n_features_in_))
        dummy[:, 0] = preds_norm
        preds = scaler.inverse_transform(dummy)[:, 0]
    else:
        preds = preds_norm

    return preds


def load_last_window_from_prepared(info, empresa):
    dados = info['dados'][empresa]
    scaler = dados.get('scaler')
    seq_length = dados['seq_length']
    features = dados['features']
    df_orig = dados['dados_originais']
    last_rows = df_orig[features].values[-seq_length:]
    if scaler is not None:
        last_norm = scaler.transform(last_rows)
    else:
        last_norm = last_rows
    return last_norm, scaler


def find_model_files():
    here = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(here, '..'))

    # busca recursiva a partir da raiz do repositório (encontra src/app/models/...)
    files = glob.glob(os.path.join(repo_root, '**', '*.ckpt'), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    return sorted(files)


st.set_page_config(page_title='Forecast LSTM', layout='wide')
# st.title('Forecast LSTM — Seleção de modelo em `models/`')

files = find_model_files()
if not files:
    st.warning('Nenhum arquivo .ckpt encontrado. Gere um modelo primeiro ou coloque um .ckpt em models/ ou src/app/models/.')

# Barra lateral: navegação e controles
page = st.sidebar.radio('Página', ['Forecast', 'Treinamento'])

if page == 'Forecast':
    st.header('Forecast')

    # Mostrar apenas o nome do arquivo no selectbox da sidebar, mas manter o caminho completo internamente
    display_map = {}
    counts = {}
    for f in files:
        name = os.path.basename(f)
        cnt = counts.get(name, 0)
        if cnt == 0:
            display = name
        else:
            display = f"{name} ({cnt})"
        counts[name] = cnt + 1
        display_map[display] = f

    display_options = list(display_map.keys())
    model_choice_display = st.selectbox('Selecione o checkpoint (.ckpt)', display_options if display_options else [])
    model_choice = display_map.get(model_choice_display) if model_choice_display else None

    # Preencher automaticamente `Empresa` a partir do nome do checkpoint (ex: AAPL-best-... -> AAPL)
    if model_choice:
        inferred = os.path.basename(model_choice).split('-')[0].upper()
    else:
        inferred = 'AAPL'

    if 'last_model_choice' not in st.session_state:
        st.session_state['last_model_choice'] = None
    if 'empresa' not in st.session_state:
        st.session_state['empresa'] = inferred
    elif st.session_state.get('last_model_choice') != model_choice:
        # modelo trocado — atualiza campo empresa automaticamente
        st.session_state['empresa'] = inferred

    st.session_state['last_model_choice'] = model_choice

    # Empresa / dias na página principal
    empresa = st.text_input('Empresa (símbolo)', key='empresa')
    dias = st.number_input('Dias à frente', min_value=1, max_value=365, value=60)
    device = 'cpu'

    if st.button('Gerar Previsão'):
        if not model_choice:
            st.error('Selecione um checkpoint válido.')
        else:
            with st.spinner('Preparando dados e carregando modelo...'):
                dados_obj = Dados()
                end = datetime.now().strftime('%Y-%m-%d')
                start = (pd.to_datetime(end) - pd.Timedelta(days=800)).strftime('%Y-%m-%d')
                try:
                    df = dados_obj.obter_dados_yfinance([empresa], start, end)
                except Exception as e:
                    st.error(f'Erro ao baixar dados: {e}')
                    st.stop()

                try:
                    prep = dados_obj.preparar_para_lstm(df=df, coluna_alvo='Close', janela=60, incluir_features=['Volume','High','Low','Open'], normalizar=True, batch_size=32)
                except Exception as e:
                    st.error(f'Erro ao preparar dados: {e}')
                    st.stop()

                if empresa not in prep['empresas']:
                    st.error('Empresa não presente nos dados preparados.')
                    st.stop()

                last_window, scaler = load_last_window_from_prepared(prep, empresa)
                n_features = prep['dados'][empresa]['n_features']

                ckpt_path = model_choice
                if not os.path.exists(ckpt_path):
                    ckpt_path = os.path.join(os.path.dirname(__file__), model_choice)

                try:
                    model = LSTMPredictor.load_from_checkpoint(ckpt_path, input_size=n_features)
                except Exception as e:
                    st.error(f'Erro ao carregar o modelo: {e}')
                    st.stop()

                preds = iterative_forecast(model, scaler, last_window, dias, device=device)

                last_date = prep['dados'][empresa]['dados_originais']['Date'].max()
                dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=dias)
                df_out = pd.DataFrame({'Date': dates, 'Predicted': preds})

                st.success('Previsão concluída')
                st.subheader('Previsões geradas')
                st.write(f'Total de linhas: {len(df_out)}')

                # Exibir tabela sem a hora (apenas data)
                df_display = df_out.copy()
                df_display['Date'] = pd.to_datetime(df_display['Date']).dt.date
                st.dataframe(df_display)

                # Para o gráfico, use o índice datetime (mantém granularidade correta)
                st.line_chart(df_out.set_index('Date')['Predicted'])

                csv_bytes = df_out.to_csv(index=False).encode('utf-8')
                st.download_button('Baixar CSV', data=csv_bytes, file_name=f'forecast_{empresa}_{dias}d.csv', mime='text/csv')
elif page == 'Treinamento':
    st.header('Treinamento de Modelo LSTM')
   
