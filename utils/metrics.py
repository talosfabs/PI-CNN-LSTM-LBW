import numpy as np
from sklearn.metrics import mean_squared_error
from typing import Dict, Union, List

def calculate_metrics(labels: Union[np.ndarray, List], 
                     outputs: Union[np.ndarray, List]) -> Dict[str, float]:
    labels = np.asarray(labels)
    outputs = np.asarray(outputs)
    
    mae = np.mean(np.abs(outputs - labels))
    mse = mean_squared_error(labels, outputs)
    rmse = np.sqrt(mse)
    
    return {
        "MAE": mae, 
        "MSE": mse, 
        "RMSE": rmse
    }