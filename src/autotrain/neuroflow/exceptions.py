class NeuroFlowError(Exception):
    """Base exception for NeuroFlow integration errors."""


class NeuroFlowValidationError(NeuroFlowError):
    """Raised when the provided graph cannot be translated into an AutoTrain job."""


class NeuroFlowConfigError(NeuroFlowError):
    """Raised when required configuration bits are missing."""
