import os
from abc import ABC, abstractmethod
import torch
import torch.optim
from pytorch_forecasting import TemporalFusionTransformer, QuantileLoss

#The utility functions to train and test temporal fusion transformer

class TFTModel(ABC):
    """utility functions for temporal fusion transformer
    """
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get('hidden_size', 16)
        self.hidden_continuous_size = kwargs.get('hidden_continuous_size', 16)
        self.learning_rate = kwargs.get('learning_rate', 0.002)
        
    def create_tft_model(self, dataset):
        """create the tft model
        """
        #create the model
        self.model = TemporalFusionTransformer.from_dataset(
            # dataset
            dataset,
            # architecture hyperparameters
            hidden_size=self.hidden_size,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=16,
            # loss metric to optimize
            loss=QuantileLoss(),
            # logging frequency
            log_interval=2,
            # optimizer parameters
            learning_rate=self.learning_rate,
            reduce_on_plateau_patience=4
        )


