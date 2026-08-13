"""
NASA C-MAPS Predictive Maintenance Package
Turbofan engine RUL prediction using XGBoost
"""
from .data_loader import load_data
from .feature_engineering import FeatureEngineer

__version__ = "1.0.0"
__author__ = "Rayhane Nouri"
__all__ = [
    'load_data',
    'FeatureEngineer',
]
