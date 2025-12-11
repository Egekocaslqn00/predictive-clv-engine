"""
E-Commerce Customer Lifetime Value (CLV) Prediction and Segmentation

This package provides tools for:
- Data processing and cleaning
- RFM analysis
- Customer segmentation
- Advanced CLV modeling using probabilistic models
- Visualization and reporting
"""

__version__ = "1.0.0"
__author__ = "Data Scientist"
__email__ = "your.email@example.com"

from . import data_processing
from . import rfm_analysis
from . import segmentation
from . import clv_modeling
from . import visualization
from . import utils

__all__ = [
    "data_processing",
    "rfm_analysis",
    "segmentation",
    "clv_modeling",
    "visualization",
    "utils",
]
