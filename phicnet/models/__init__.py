"""
PhICNet model architectures and baselines.
"""
from .physics_model import PhysicsWave2D
from .source_net import REDNet
from .recurrent_source import ConvLSTMCell, RecurrentSourceNetwork
from .phicnet import PhICNet, PhysicsOnlyBaseline, CNNBaseline, ConvLSTMBaseline

__all__ = [
    "PhysicsWave2D",
    "REDNet",
    "ConvLSTMCell",
    "RecurrentSourceNetwork",
    "PhICNet",
    "PhysicsOnlyBaseline",
    "CNNBaseline",
    "ConvLSTMBaseline"
]
