"""Boston-area daily weather forecasting from NOAA station data.

v0.1.x: Ridge regression + Seasonal Naive baseline (stdlib only).
v0.2.x: Pure-TensorFlow LSTM and Transformer multi-step forecasters.
"""

__all__ = [
    "__version__",
    "LSTMForecaster",
    "TransformerForecaster",
    "build_sequence_dataset",
    "SequenceScaler",
]

__version__ = "0.2.0"

try:
    from climate_modeling.sequence_dataset import SequenceScaler, build_sequence_dataset
    from climate_modeling.tf_models import LSTMForecaster, TransformerForecaster
except ImportError:
    # TensorFlow not installed — stdlib-only pipeline still usable
    pass
