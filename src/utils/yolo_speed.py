import numpy as np
from typing import Dict

class SpeedMetrics:
    """Track inference speed metrics"""
    
    def __init__(self):
        self.times = []
    
    def update(self, time: float):
        self.times.append(time)
    
    def compute(self) -> Dict[str, float]:
        if not self.times:
            return {'fps': 0.0, 'latency_ms': 0.0}
        
        times = np.array(self.times)
        mean_time = times.mean()
        
        return {
            'fps': 1.0 / mean_time if mean_time > 0 else 0.0,
            'latency_ms': mean_time * 1000,
            'latency_std_ms': times.std() * 1000
        }