"""
Module de structures de données personnalisées pour le projet IRP DDoS Detection.
"""

from .base import IRPBaseStructure, IRPDataFrame, IRPDaskFrame
from .flow import NetworkFlow

__all__ = [
    'IRPBaseStructure',
    'IRPDataFrame',
    'IRPDaskFrame',
    'NetworkFlow',
]
