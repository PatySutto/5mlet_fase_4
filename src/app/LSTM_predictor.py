from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler


class LSTMPredictor(pl.LightningModule):
    """
    Modelo LSTM para previsão de séries temporais usando PyTorch Lightning.
    
    Attributes:
        input_size (int): Número de features de entrada
        hidden_size (int): Número de unidades LSTM ocultas
        num_layers (int): Número de camadas LSTM
        learning_rate (float): Taxa de aprendizado
        bidirectional (bool): Se True, usa LSTM bidirecional
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 50,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        bidirectional: bool = False
    ):
        """
        Inicializa o modelo LSTM.
        
        Args:
            input_size: Número de features de entrada
            hidden_size: Número de unidades LSTM ocultas (padrão: 50)
            num_layers: Número de camadas LSTM (padrão: 2)
            dropout: Taxa de dropout entre camadas (padrão: 0.2)
            learning_rate: Taxa de aprendizado (padrão: 0.001)
            bidirectional: Se True, usa LSTM bidirecional (padrão: False)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Hiperparâmetros
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.direction_multiplier = 2 if bidirectional else 1
        
        # Arquitetura do modelo
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.fc = nn.Linear(hidden_size * self.direction_multiplier, 1)
        self.criterion = nn.MSELoss()
        
        # Histórico de perdas
        self.train_losses = []
        self.val_losses = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada com shape (batch_size, seq_length, input_size)
        
        Returns:
            Predições com shape (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions
    
    def training_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Executa um step de treinamento.
        
        Args:
            batch: Tupla (x, y) com dados de entrada e target
            batch_idx: Índice do batch
        
        Returns:
            Valor da loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Executa um step de validação.
        
        Args:
            batch: Tupla (x, y) com dados de entrada e target
            batch_idx: Índice do batch
        
        Returns:
            Valor da loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def test_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Executa um step de teste.
        
        Args:
            batch: Tupla (x, y) com dados de entrada e target
            batch_idx: Índice do batch
        
        Returns:
            Valor da loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Configura o otimizador e scheduler.
        
        Returns:
            Dicionário com optimizer e lr_scheduler
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_sequence(
        self,
        x: torch.Tensor,
        scaler: Optional[MinMaxScaler] = None
    ) -> np.ndarray:
        """
        Faz predições em uma sequência de dados.
        
        Args:
            x: Tensor de entrada
            scaler: Scaler para denormalizar as predições (opcional)
        
        Returns:
            Array numpy com predições (denormalizadas se scaler fornecido)
        """
        self.eval()
        with torch.no_grad():
            predictions = self(x).cpu().numpy()
            
            if scaler is not None:
                dummy = np.zeros((predictions.shape[0], scaler.n_features_in_))
                dummy[:, 0] = predictions.flatten()
                predictions = scaler.inverse_transform(dummy)[:, 0]
            
            return predictions