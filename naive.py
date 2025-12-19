import numpy as np
from abc import ABC, abstractmethod

class NaiveModel(ABC):
    def __init__(self, **kwargs):
        self.history = kwargs.get('history',None)
        self.forecast_length = kwargs.get('forecast_length')
        self.forecast = None

    @abstractmethod
    def fit(self):
        pass
    

class PersistenceModel(NaiveModel):
    """
    Return persistence model forecasts, i.e. y_t+h = y_t
    """
    
    def __init_(self):
        super().__init__()
        
    def fit(self): 
        
        # create outer product of ones and last element in history
        base = np.ones((1,self.forecast_length,1))
        forecast = self.history[:,-1,:]
        self.forecast = forecast[:,np.newaxis,:] * base
        
        
class DriftModel(NaiveModel):
    """
    Return drift model forecasts, i.e. y_t+h = y_t + h * delta_y/delta_t
    """
    def __init_(self):
        super().__init__()
        
    def fit(self):
        ### Fill this in
        
        y_t = self.history[:, -1, :]
        slope = (self.history[:,-1,:] - self.history[:,0,:]) / self.history.shape[1]
        timesteps = np.arange(self.forecast_length)+1
        timesteps = np.reshape(timesteps, (1, self.forecast_length, 1))
        
        base_constant = np.ones((1, self.forecast_length, 1))
        base_linear = slope[:, np.newaxis, :]
        
        b = (base_linear * timesteps)
        c = base_constant * y_t[:, np.newaxis, :]
        self.forecast = b + c
        
    