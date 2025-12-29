import yfinance as yf
import pandas as pd
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class Dados:
    def __init__(self):
        """
        Inicializa a classe Dados para gerenciar dados financeiros.
        """
        self.dados_cache = None

    def obter_dados_yfinance(
        self, 
        empresas: List[str], 
        data_inicio: str, 
        data_fim: str
    ) -> pd.DataFrame:
        """
        Obtém os dados históricos das empresas usando yfinance com estrutura organizada.
        
        :param empresas: Lista de símbolos das empresas (ex: ['AAPL', 'MSFT'])
        :param data_inicio: Data de início no formato 'YYYY-MM-DD'
        :param data_fim: Data de fim no formato 'YYYY-MM-DD'
        :param colunas: Lista de colunas desejadas. Se None, retorna todas.
                       Opções: ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        :return: DataFrame com os dados históricos organizados
        """
        dados_lista = []
        
        print(f"Baixando dados para {len(empresas)} empresa(s)...\n")
        
        for empresa in empresas:
            try:
                # Baixa os dados da empresa (silencia o output)
                df = yf.download(empresa, start=data_inicio, end=data_fim, progress=False)
                
                if df.empty:
                    print(f"⚠️  Aviso: Nenhum dado encontrado para {empresa}")
                    continue
                
                # Remove MultiIndex das colunas se existir
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Remove o nome do índice das colunas (geralmente 'Price')
                df.columns.name = None
                
                # Reseta o índice para transformar Date em coluna
                df = df.reset_index()
                
                # Adiciona a coluna Empresa no início
                df.insert(0, 'Empresa', empresa)
                
                dados_lista.append(df)
                print(f"✓ {empresa}: {len(df)} registros baixados")
                
            except Exception as e:
                print(f"✗ Erro ao baixar dados de {empresa}: {str(e)}")
        
        if not dados_lista:
            print("\n⚠️  Nenhum dado foi baixado com sucesso.")
            return pd.DataFrame()
        
        # Concatena todos os DataFrames
        dados_completos = pd.concat(dados_lista, ignore_index=True)
                
        # Ordena por Empresa e Data
        dados_completos = dados_completos.sort_values(['Empresa', 'Date']).reset_index(drop=True)
        
        # Armazena em cache
        self.dados_cache = dados_completos
         
        return dados_completos

    def preparar_para_lstm(
        self,
        df: Optional[pd.DataFrame] = None,
        coluna_alvo: str = 'Close',
        janela: int = 60,
        incluir_features: Optional[List[str]] = None,
        normalizar: bool = True,
        batch_size: int = 32,
        usar_pytorch: bool = True
    ) -> dict:
        """
        Prepara os dados para treinamento de modelos LSTM com PyTorch Lightning.
        
        :param df: DataFrame com os dados. Se None, usa o cache.
        :param coluna_alvo: Coluna que será predita (padrão: 'Close')
        :param janela: Número de dias anteriores para usar como entrada (padrão: 60)
        :param incluir_features: Lista de features adicionais (ex: ['Volume', 'High', 'Low'])
        :param normalizar: Se True, normaliza os dados entre 0 e 1
        :param batch_size: Tamanho do batch para DataLoader (padrão: 32)
        :param usar_pytorch: Se True, retorna DataLoaders do PyTorch (padrão: True)
        :return: Dicionário com dados preparados, scaler e DataLoaders
        :raises ValueError: Se parâmetros forem inválidos
        """
                
        # Usa cache se df não for fornecido
        if df is None:
            df = self.dados_cache
        
        if df is None or df.empty:
            raise ValueError("❌ Nenhum dado disponível. Execute obter_dados_yfinance() primeiro.")
        
        # Validações
        if coluna_alvo not in df.columns:
            raise ValueError(f"❌ Coluna '{coluna_alvo}' não encontrada. Colunas disponíveis: {list(df.columns)}")
        
        if janela < 1:
            raise ValueError("❌ O tamanho da janela deve ser no mínimo 1")
        
        if janela >= len(df):
            raise ValueError(f"❌ Janela ({janela}) maior que o número de registros ({len(df)})")
        
        print(f"\n{'='*80}")
        print(f"PREPARANDO DADOS PARA LSTM (PyTorch Lightning)")
        print(f"{'='*80}")
        
        # Determina quais features usar
        if incluir_features is None:
            features = [coluna_alvo]
        else:
            features = [coluna_alvo] + [f for f in incluir_features if f in df.columns and f != coluna_alvo]
            features_faltantes = [f for f in incluir_features if f not in df.columns]
            if features_faltantes:
                print(f"⚠️  Features não encontradas: {features_faltantes}")
        
        print(f"✓ Features selecionadas: {features}")
        print(f"✓ Tamanho da janela: {janela} dias")
        print(f"✓ Coluna alvo: {coluna_alvo}")
        print(f"✓ Batch size: {batch_size}")
        
        # Agrupa por empresa se houver múltiplas
        empresas = df['Empresa'].unique() if 'Empresa' in df.columns else ['Dados']
        
        dados_preparados = {}
        
        for empresa in empresas:
            print(f"\n--- Processando: {empresa} ---")
            
            # Filtra dados da empresa
            if 'Empresa' in df.columns:
                df_empresa = df[df['Empresa'] == empresa].copy()
            else:
                df_empresa = df.copy()
            
            # Ordena por data
            if 'Date' in df_empresa.columns:
                df_empresa = df_empresa.sort_values('Date').reset_index(drop=True)
            
            # Extrai features
            dados = df_empresa[features].values
            
            # Verifica NaN
            if np.isnan(dados).any():
                print(f"⚠️  Aviso: Dados contêm valores NaN. Removendo linhas com NaN...")
                df_empresa = df_empresa.dropna(subset=features)
                dados = df_empresa[features].values
            
            if len(dados) < janela + 1:
                print(f"✗ Erro: Dados insuficientes para empresa {empresa} ({len(dados)} < {janela + 1})")
                continue
            
            # Normalização
            scaler = None
            if normalizar:
                scaler = MinMaxScaler(feature_range=(0, 1))
                dados_normalizados = scaler.fit_transform(dados)
            else:
                dados_normalizados = dados
            
            # Cria sequências para LSTM
            X, y = [], []
            
            for i in range(janela, len(dados_normalizados)):
                X.append(dados_normalizados[i-janela:i])  # Janela de entrada
                y.append(dados_normalizados[i, 0])         # Valor alvo (primeiro feature)
            
            X = np.array(X)
            y = np.array(y)
            
            # Divide em treino e teste (80/20)
            split_idx = int(len(X) * 0.8)
            
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_test = X[split_idx:]
            y_test = y[split_idx:]
            
            # Converte para PyTorch se solicitado
            if usar_pytorch:
                # Converte para tensores
                X_train_tensor = torch.FloatTensor(X_train)
                y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
                X_test_tensor = torch.FloatTensor(X_test)
                y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
                
                # Cria DataLoaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0
                )
                
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0
                )
                
                dados_preparados[empresa] = {
                    'train_loader': train_loader,
                    'test_loader': test_loader,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'X_train_tensor': X_train_tensor,
                    'y_train_tensor': y_train_tensor,
                    'X_test_tensor': X_test_tensor,
                    'y_test_tensor': y_test_tensor,
                    'scaler': scaler,
                    'features': features,
                    'dados_originais': df_empresa,
                    'n_features': len(features),
                    'seq_length': janela,
                    'batch_size': batch_size
                }
                
                print(f"✓ Train DataLoader: {len(train_loader)} batches")
                print(f"✓ Test DataLoader: {len(test_loader)} batches")
            else:
                dados_preparados[empresa] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test,
                    'scaler': scaler,
                    'features': features,
                    'dados_originais': df_empresa
                }
            
            print(f"✓ Treino: X={X_train.shape}, y={y_train.shape}")
            print(f"✓ Teste:  X={X_test.shape}, y={y_test.shape}")
            print(f"✓ Split: {split_idx}/{len(X)-split_idx} (treino/teste)")
        
        if not dados_preparados:
            raise ValueError("❌ Nenhum dado foi preparado com sucesso")
        
        print(f"\n{'='*80}")
        print(f"✓ DADOS PREPARADOS COM SUCESSO PARA {len(dados_preparados)} EMPRESA(S)")
        print(f"{'='*80}\n")
        
        # Informações adicionais
        info = {
            'janela': janela,
            'coluna_alvo': coluna_alvo,
            'features': features,
            'normalizado': normalizar,
            'empresas': list(dados_preparados.keys()),
            'n_features': len(features),
            'batch_size': batch_size,
            'usar_pytorch': usar_pytorch,
            'dados': dados_preparados
        }
        
        return info

    
# Exemplo de uso
if __name__ == "__main__":
    # Inicializa a classe
    dados_obj = Dados()

    #Viveo - VVEO3.SA
    
    # Define as empresas e período
    empresas = ['VVEO3.SA']
    
    # Obtém os dados
    df = dados_obj.obter_dados_yfinance(
        empresas=empresas,
        data_inicio='2000-01-01',
        data_fim='2025-12-31'
    )
    
    # Visualiza os primeiros registros
    print("\n" + "="*80)
    print("PRIMEIROS REGISTROS:")
    print("="*80)
    print(df.head(10))
    print("="*80)
    print(df.shape)

    dados_lstm = dados_obj.preparar_para_lstm(
        df=df,
        coluna_alvo='Close',
        janela=60,
        incluir_features=['Volume', 'High', 'Low', 'Open'],
        normalizar=True,
        batch_size=32
    )
    
    for empresa in dados_lstm['empresas']:
        print(f"\n--- Dados de {empresa} prontos para PyTorch Lightning ---")
        print(f"Train DataLoader: {len(dados_lstm['dados'][empresa]['train_loader'])} batches")
        print(f"Test DataLoader: {len(dados_lstm['dados'][empresa]['test_loader'])} batches")
        print(f"Input shape: (batch_size, {dados_lstm['dados'][empresa]['seq_length']}, {dados_lstm['dados'][empresa]['n_features']})")
        print(f"Output shape: (batch_size, 1)")