import numpy as np
from typing import Union

def solve_heat_transfer(alpha: Union[float, np.ndarray], 
                       thermal_conductivity: Union[float, np.ndarray], 
                       radius: Union[float, np.ndarray], 
                       efficiency: Union[float, np.ndarray], 
                       power: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    # Constants
    EMPIRICAL_CONSTANT = 2.556e-2
    SPEED_CONSTANT = 2.0 / 60
    TEMPERATURE_DELTA = 923 - 298
    DECAY_FACTOR = 1 - np.exp(-3)
    
    # Calculate heat transfer coefficient
    numerator = efficiency * power * alpha * EMPIRICAL_CONSTANT
    denominator = SPEED_CONSTANT * thermal_conductivity * radius * DECAY_FACTOR * TEMPERATURE_DELTA
    
    # Convert to mm units
    return numerator / denominator * 1e3