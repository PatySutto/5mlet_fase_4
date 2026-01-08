"""
Script principal para execução completa da solução de predição LSTM.

Este script executa todo o pipeline:
1. Coleta de dados
2. Preparação para LSTM
3. Criação do modelo
4. Treinamento
5. Avaliação
6. Visualização
7. Salvamento

Autor: Sistema de Predição de Ações
Data: 2024
"""

# ============================================================================
# IMPORTS
# ============================================================================

from .dados import Dados
from .LSTM_trainer import LSTMTrainer


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

# Configurações de dados
EMPRESAS = ['AAPL', 'MSFT', 'GOOGL']
DATA_INICIO = '2020-01-01'
DATA_FIM = '2023-12-31'

# Configurações do LSTM
JANELA = 60  # Dias anteriores para prever
BATCH_SIZE = 32
FEATURES = ['Volume', 'High', 'Low', 'Open']

# Configurações do modelo
HIDDEN_SIZE = 50
NUM_LAYERS = 2
DROPOUT = 0.2
LEARNING_RATE = 0.001
BIDIRECTIONAL = False

# Configurações de treinamento
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """
    Executa o pipeline completo de predição LSTM.
    """
    
    print("\n" + "="*80)
    print("PIPELINE DE PREDIÇÃO LSTM - INICIANDO")
    print("="*80)
    
    # ------------------------------------------------------------------------
    # ETAPA 1: COLETAR DADOS
    # ------------------------------------------------------------------------
    print("\n[1/7] COLETANDO DADOS...")
    
    dados_obj = Dados()
    df = dados_obj.obter_dados_yfinance(
        empresas=EMPRESAS,
        data_inicio=DATA_INICIO,
        data_fim=DATA_FIM
    )
    
    # Visualiza primeiros registros
    print("\nPrimeiros registros:")
    print(df.head())
    
    # ------------------------------------------------------------------------
    # ETAPA 2: PREPARAR DADOS PARA LSTM
    # ------------------------------------------------------------------------
    print("\n[2/7] PREPARANDO DADOS PARA LSTM...")
    
    dados_lstm = dados_obj.preparar_para_lstm(
        df=df,
        coluna_alvo='Close',
        janela=JANELA,
        incluir_features=FEATURES,
        batch_size=BATCH_SIZE,
        normalizar=True,
        usar_pytorch=True
    )
    
    # ------------------------------------------------------------------------
    # ETAPA 3-7: TREINAR PARA CADA EMPRESA
    # ------------------------------------------------------------------------
    modelos_treinados = {}
    
    for empresa in dados_lstm['empresas']:
        print("\n" + "="*80)
        print(f"PROCESSANDO EMPRESA: {empresa}")
        print("="*80)
        
        # ETAPA 3: CRIAR TREINADOR
        print(f"\n[3/7] CRIANDO TREINADOR PARA {empresa}...")
        trainer = LSTMTrainer(dados_lstm, empresa=empresa)
        
        # ETAPA 4: CRIAR MODELO
        print(f"\n[4/7] CRIANDO MODELO LSTM PARA {empresa}...")
        trainer.criar_modelo(
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            learning_rate=LEARNING_RATE,
            bidirectional=BIDIRECTIONAL
        )
        
        # ETAPA 5: TREINAR
        print(f"\n[5/7] TREINANDO MODELO PARA {empresa}...")
        trainer.treinar(
            max_epochs=MAX_EPOCHS,
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
        
        # ETAPA 6: AVALIAR
        print(f"\n[6/7] AVALIANDO MODELO PARA {empresa}...")
        metricas = trainer.avaliar()
        
        # ETAPA 7:  SALVAR
               
        # Salvar modelo
        nome_arquivo = f'modelo_{empresa}.ckpt'
        trainer.salvar_modelo(nome_arquivo)
        
        # Armazena informações
        modelos_treinados[empresa] = {
            'trainer': trainer,
            'metricas': metricas,
            'arquivo': nome_arquivo
        }
    
    # ------------------------------------------------------------------------
    # RESUMO FINAL
    # ------------------------------------------------------------------------
    print("\n" + "="*80)
    print("RESUMO FINAL - TODAS AS EMPRESAS")
    print("="*80)
    
    for empresa, info in modelos_treinados.items():
        metricas = info['metricas']
        print(f"\n{empresa}:")
        print(f"  RMSE: ${metricas['RMSE']:.2f}")
        print(f"  MAE:  ${metricas['MAE']:.2f}")
        print(f"  MAPE: {metricas['MAPE']:.2f}%")
        print(f"  Arquivo: {info['arquivo']}")
    
    print("\n" + "="*80)
    print("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*80 + "\n")
    
    return modelos_treinados


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def treinar_empresa_unica(empresa: str, data_inicio: str, data_fim: str):
    """
    Treina modelo para uma única empresa.
    
    Args:
        empresa: Símbolo da empresa (ex: 'AAPL')
        data_inicio: Data inicial (formato: 'YYYY-MM-DD')
        data_fim: Data final (formato: 'YYYY-MM-DD')
    
    Returns:
        trainer: Objeto LSTMTrainer treinado
    """
    print(f"\n{'='*80}")
    print(f"TREINAMENTO INDIVIDUAL: {empresa}")
    print(f"{'='*80}")
    
    # Coleta dados
    dados_obj = Dados()
    df = dados_obj.obter_dados_yfinance(
        empresas=[empresa],
        data_inicio=data_inicio,
        data_fim=data_fim
    )
    
    # Prepara para LSTM
    dados_lstm = dados_obj.preparar_para_lstm(
        df=df,
        coluna_alvo='Close',
        janela=JANELA,
        incluir_features=FEATURES,
        batch_size=BATCH_SIZE,
        normalizar=True
    )
    
    # Cria e treina
    trainer = LSTMTrainer(dados_lstm, empresa=empresa)
    trainer.criar_modelo(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE
    )
    
    trainer.treinar(max_epochs=MAX_EPOCHS)
    metricas = trainer.avaliar()
    trainer.plotar_resultados()
    trainer.salvar_modelo(f'modelo_{empresa}.ckpt')
    
    print(f"\n✓ Treinamento concluído para {empresa}")
    print(f"RMSE: ${metricas['RMSE']:.2f} | MAPE: {metricas['MAPE']:.2f}%")
    
    return trainer


def comparar_modelos(modelos_treinados: dict):
    """
    Compara performance de múltiplos modelos.
    
    Args:
        modelos_treinados: Dicionário com modelos treinados
    """
    print("\n" + "="*80)
    print("COMPARAÇÃO DE MODELOS")
    print("="*80)
    
    import pandas as pd
    
    # Cria DataFrame de comparação
    comparacao = []
    for empresa, info in modelos_treinados.items():
        metricas = info['metricas']
        comparacao.append({
            'Empresa': empresa,
            'RMSE': metricas['RMSE'],
            'MAE': metricas['MAE'],
            'MAPE': metricas['MAPE']
        })
    
    df_comp = pd.DataFrame(comparacao)
    df_comp = df_comp.sort_values('MAPE')
    
    print("\nRanking por MAPE (melhor modelo primeiro):")
    print(df_comp.to_string(index=False))
    
    # Melhor modelo
    melhor = df_comp.iloc[0]
    print(f"\n🏆 Melhor Modelo: {melhor['Empresa']}")
    print(f"   MAPE: {melhor['MAPE']:.2f}%")


# ============================================================================
# EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    # Executa pipeline completo
    modelos = main()
    
    # Comparar modelos (opcional)
    comparar_modelos(modelos)
    
    # Exemplo: Treinar apenas uma empresa
    # trainer_apple = treinar_empresa_unica('AAPL', '2020-01-01', '2023-12-31')
    