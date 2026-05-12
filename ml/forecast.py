import pandas as pd
import numpy as np
from typing import Optional, Tuple

def calculate_time_to_breach(values: list, threshold: float = 95.0) -> Optional[float]:
    """
    Calculates the estimated seconds until a metric hits a critical threshold.
    Uses simple Linear Regression: y = mx + b
    
    Args:
        values: List of recent metric values (e.g., last 10 records)
        threshold: The critical limit (e.g., 95% CPU)
        
    Returns:
        Estimated seconds (float), or None if stable/decreasing
    """
    if len(values) < 5:
        return None # Not enough data for a reliable trend
        
    # Assume data points are 5 seconds apart (dashboard refresh interval)
    x = np.array(range(len(values))) * 5.0
    y = np.array(values)
    
    # Calculate slope (m) and intercept (b)
    try:
        n = len(x)
        m = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
        b = (np.sum(y) - m * np.sum(x)) / n
        
        # If slope is positive, calculate time to reach threshold
        if m > 0.05: # Only predict if it's actually climbing significantly
            current_value = values[-1]
            if current_value >= threshold:
                return 0.0
                
            seconds_remaining = (threshold - current_value) / m
            return max(0.0, seconds_remaining)
            
    except ZeroDivisionError:
        return None
        
    return None
