"""Standard GP-BO baseline: handcrafted descriptors + GP, no graph encoder.

This package is the descriptor-based comparator for the DKL-BO benchmark. It
deliberately reuses the *same* GP surrogate (`models.surrogate.ExactGPSurrogate`)
as DKL-BO so the only difference between the two methods is the feature source:
handcrafted composition + structure descriptors here, learned CGCNN embeddings there.
"""

from .descriptors import build_descriptors, FEATURE_NOTES
from .feature_bo_loop import FeatureBOLoop

__all__ = ["build_descriptors", "FEATURE_NOTES", "FeatureBOLoop"]
