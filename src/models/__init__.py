"""
NILM Deep Learning Neural Network Models Package
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .nilm_net import MultiTaskConvBiGRUNet

__all__ = ["MultiTaskConvBiGRUNet"]
