from typing import Optional, List, Dict, Any
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from LSTM_predictor import LSTMPredictor

class LSTMTrainer:
    """
    Classe para gerenciar o treinamento e avaliação de modelos LSTM.
    
    Esta classe simplifica o processo de:
    - Criação do modelo
    - Treinamento com early stopping
    - Avaliação com múltiplas métricas
    - Visualização de resultados
    - Salvamento e carregamento de modelos
    
    Attributes:
        empresa (str): Nome da empresa sendo modelada
        n_features (int): Número de features de entrada
        model (LSTMPredictor): Modelo LSTM
        trainer (pl.Trainer): Trainer do PyTorch Lightning
    """
    
    def __init__(self, dados_lstm: Dict[str, Any], empresa: str):
        """
        Inicializa o treinador LSTM.
        
        Args:
            dados_lstm: Dicionário retornado por preparar_para_lstm()
            empresa: Nome da empresa a treinar
        
        Raises:
            ValueError: Se a empresa não for encontrada nos dados
        """
        if empresa not in dados_lstm['empresas']:
            raise ValueError(
                f"❌ Empresa '{empresa}' não encontrada. "
                f"Disponíveis: {dados_lstm['empresas']}"
            )
        
        # Atributos principais
        self.empresa = empresa
        self.dados_lstm = dados_lstm
        self.dados_empresa = dados_lstm['dados'][empresa]
        
        # DataLoaders
        self.train_loader = self.dados_empresa['train_loader']
        self.test_loader = self.dados_empresa['test_loader']
        
        # Configurações
        self.scaler = self.dados_empresa['scaler']
        self.n_features = self.dados_empresa['n_features']
        self.seq_length = self.dados_empresa['seq_length']
        
        # Modelo e resultados
        self.model = None
        self.trainer = None
        self.predictions = None
        self.metricas = None
        
        self._print_inicializacao()
    
    def _print_inicializacao(self) -> None:
        """Imprime informações de inicialização."""
        print(f"\n{'='*80}")
        print(f"TREINADOR LSTM INICIALIZADO PARA: {self.empresa}")
        print(f"{'='*80}")
        print(f"✓ Features: {self.n_features}")
        print(f"✓ Sequência: {self.seq_length} dias")
        print(f"✓ Batches treino: {len(self.train_loader)}")
        print(f"✓ Batches teste: {len(self.test_loader)}")
    
    def criar_modelo(
        self,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = False
    ) -> LSTMPredictor:
        """
            Cria um novo modelo LSTM.
            
            Args:
                hidden_size: Número de unidades LSTM ocultas (padrão: 50)
                num_layers: Número de camadas LSTM (padrão: 2)
                dropout: Taxa de dropout (padrão: 0.2)
                learning_rate: Taxa de aprendizado (padrão: 0.001)
                bidirectional: Se True, usa LSTM bidirecional (padrão: False)
            
            Returns:
                Modelo LSTM criado
        """
        self.model = LSTMPredictor(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            bidirectional=bidirectional
        )
        
        print(f"\n{'='*80}")
        print("MODELO CRIADO")
        print(f"{'='*80}")
        print(f"✓ Input size: {self.n_features}")
        print(f"✓ Hidden size: {hidden_size}")
        print(f"✓ Num layers: {num_layers}")
        print(f"✓ Dropout: {dropout}")
        print(f"✓ Learning rate: {learning_rate}")
        print(f"✓ Bidirectional: {bidirectional}")
        
        # Calcula parâmetros totais
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✓ Parâmetros totais: {total_params:,}")
        print(f"✓ Parâmetros treináveis: {trainable_params:,}")
        
        return self.model
    
    def treinar(
        self,
        max_epochs: int = 50,
        accelerator: str = 'auto',
        devices: int = 1,
        early_stopping_patience: int = 10,
        callbacks: Optional[List] = None
    ) -> None:
        """
        Treina o modelo LSTM.
        
        Args:
            max_epochs: Número máximo de épocas (padrão: 50)
            accelerator: Tipo de acelerador ('auto', 'gpu', 'cpu') (padrão: 'auto')
            devices: Número de dispositivos a usar (padrão: 1)
            early_stopping_patience: Paciência para early stopping (padrão: 10)
            callbacks: Lista de callbacks adicionais (opcional)
        
        Raises:
            ValueError: Se o modelo não foi criado
        """
        if self.model is None:
            raise ValueError("❌ Crie um modelo primeiro usando criar_modelo()")
        
        # Callbacks padrão
        default_callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath='checkpoints/',
                filename=f'{self.empresa}-{{epoch:02d}}-{{val_loss:.4f}}',
                save_top_k=3,
                mode='min'
            )
        ]
        
        if callbacks:
            default_callbacks.extend(callbacks)
        
        # Cria o trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=default_callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
            deterministic=True
        )
        
        print(f"\n{'='*80}")
        print("INICIANDO TREINAMENTO")
        print(f"{'='*80}")
        print(f"✓ Épocas máximas: {max_epochs}")
        print(f"✓ Early stopping patience: {early_stopping_patience}")
        print(f"✓ Accelerator: {accelerator}")
        
        # Treina o modelo
        self.trainer.fit(self.model, self.train_loader, self.test_loader)
        
        print(f"\n{'='*80}")
        print("✓ TREINAMENTO CONCLUÍDO")
        print(f"{'='*80}\n")
    
    def avaliar(self) -> Dict[str, float]:
        """
        Avalia o modelo no conjunto de teste.
        
        Returns:
            Dicionário com métricas de avaliação:
            - test_loss: Loss no conjunto de teste
            - MSE: Mean Squared Error
            - RMSE: Root Mean Squared Error
            - MAE: Mean Absolute Error
            - MAPE: Mean Absolute Percentage Error
        
        Raises:
            ValueError: Se o modelo não foi treinado
        """
        if self.model is None or self.trainer is None:
            raise ValueError("❌ Treine o modelo primeiro usando treinar()")
        
        print(f"\n{'='*80}")
        print("AVALIANDO MODELO")
        print(f"{'='*80}")
        
        # Testa o modelo
        test_results = self.trainer.test(self.model, self.test_loader)
        
        # Faz predições
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                pred = self.model(batch_x)
                predictions.extend(pred.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        # Denormaliza se houver scaler
        if self.scaler is not None:
            predictions = self._denormalizar(predictions)
            actuals = self._denormalizar(actuals)
        
        self.predictions = {'predictions': predictions, 'actuals': actuals}
        
        # Calcula métricas
        self.metricas = self._calcular_metricas(predictions, actuals, test_results[0])
        self._print_metricas()
        
        return self.metricas
    
    def _denormalizar(self, valores: np.ndarray) -> np.ndarray:
        """
        Denormaliza valores usando o scaler.
        
        Args:
            valores: Array de valores normalizados
        
        Returns:
            Array de valores denormalizados
        """
        dummy = np.zeros((len(valores), self.scaler.n_features_in_))
        dummy[:, 0] = valores
        return self.scaler.inverse_transform(dummy)[:, 0]
    
    def _calcular_metricas(
        self, 
        predictions: np.ndarray, 
        actuals: np.ndarray,
        test_result: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calcula métricas de avaliação.
        
        Args:
            predictions: Array de predições
            actuals: Array de valores reais
            test_result: Resultado do teste do trainer
        
        Returns:
            Dicionário com todas as métricas
        """
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'test_loss': test_result['test_loss'],
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape
        }
    
    def _print_metricas(self) -> None:
        """Imprime métricas de avaliação."""
        print(f"\n{'='*80}")
        print("MÉTRICAS DE AVALIAÇÃO")
        print(f"{'='*80}")
        print(f"✓ Test Loss: {self.metricas['test_loss']:.6f}")
        print(f"✓ MSE: {self.metricas['MSE']:.4f}")
        print(f"✓ RMSE: ${self.metricas['RMSE']:.2f}")
        print(f"✓ MAE: ${self.metricas['MAE']:.2f}")
        print(f"✓ MAPE: {self.metricas['MAPE']:.2f}%")
        print(f"{'='*80}\n")
    
    
    def salvar_modelo(self, caminho: str = 'modelo_lstm.ckpt') -> None:
        """
            Salva o modelo treinado.
            
            Args:
                caminho: Caminho para salvar o modelo (padrão: 'modelo_lstm.ckpt')
            
            Raises:
                ValueError: Se não há modelo para salvar
        """
        if self.model is None:
            raise ValueError("❌ Nenhum modelo para salvar")
        
        self.trainer.save_checkpoint(caminho)
        print(f"✓ Modelo salvo em: {caminho}")
    
    def carregar_modelo(self, caminho: str) -> None:
        """
        Carrega um modelo salvo.
        
        Args:
            caminho: Caminho do modelo salvo
        """
        self.model = LSTMPredictor.load_from_checkpoint(
            caminho,
            input_size=self.n_features
        )
        print(f"✓ Modelo carregado de: {caminho}")