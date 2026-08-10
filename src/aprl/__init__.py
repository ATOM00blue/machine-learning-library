"""Adaptive Prototype Residual Learning.

APRL combines a stable global model with relevance-weighted prototypes and
locally fitted residual experts. See ``algorithms/aprl/README.md`` for the
algorithm specification and its intended scope.
"""

from .classifier import APRLClassifier
from .regressor import APRLRegressor

__all__ = ["APRLClassifier", "APRLRegressor"]
__version__ = "0.1.0"
