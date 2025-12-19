import os
from abc import ABC
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper


class VARModel(ABC):
    """utility functions for var
    """
    def __init__(self, **kwargs):
        self.max_lag = kwargs.get('var_lag', 5)
        self.forecast_length = kwargs.get('forecast_length', 7)
        
    def train(self,  train_data, filename):
        """train the model
        """
        model = VAR(train_data)
        self.model = model.fit(self.max_lag)
        self.model.save(filename)
        
    def load_model(self, filename):
        self.model = VARResultsWrapper.load(filename)
        
    def predict(self, x):
        predictions = self.model.forecast(x.values, self.forecast_length)
        return predictions

    

