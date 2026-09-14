"""Client-side dataset owner and optional post-training eval collection."""

from .inference_client import (
    ClassificationResult,
    InferenceResult,
    SentraInferenceClient,
    SentraInferenceClientConfig,
)

__all__ = [
    "ClassificationResult",
    "InferenceResult",
    "SentraInferenceClient",
    "SentraInferenceClientConfig",
]
