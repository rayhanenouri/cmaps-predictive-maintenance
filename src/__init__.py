"""
NASA C-MAPS Predictive Maintenance Package
Turbofan engine RUL prediction using XGBoost
"""

from .data_loader import CMAPSDataLoader
from .feature_engineering import FeatureEngineer
from .model import RULPredictor

__version__ = "1.0.0"
__author__ = "Rayhane Nouri"

__all__ = [
    'CMAPSDataLoader',
    'FeatureEngineer',
    'RULPredictor'
]
