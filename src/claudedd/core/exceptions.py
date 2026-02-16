"""Custom exception hierarchy for ClaudeDD."""


class ClaudeDiscoveryError(Exception):
    """Base exception for all ClaudeDD errors."""


class MoleculeParsingError(ClaudeDiscoveryError):
    """Raised when a molecule cannot be parsed from input."""

    def __init__(self, input_repr: str, format: str, reason: str = ""):
        self.input_repr = input_repr
        self.format = format
        super().__init__(
            f"Failed to parse molecule from {format}: "
            f"'{input_repr[:80]}' - {reason}"
        )


class FileFormatError(ClaudeDiscoveryError):
    """Raised when a file format is unsupported or corrupted."""


class ValidationError(ClaudeDiscoveryError):
    """Raised when molecular validation fails."""


class ConfigurationError(ClaudeDiscoveryError):
    """Raised when pipeline configuration is invalid."""


class DatabaseError(ClaudeDiscoveryError):
    """Raised when a database fetch operation fails."""


class VisualizationError(ClaudeDiscoveryError):
    """Raised when a plot cannot be generated."""


# Phase 2 exceptions

class ScreeningError(ClaudeDiscoveryError):
    """Raised when a screening operation fails."""


class ModelError(ClaudeDiscoveryError):
    """Raised when model training, loading, or prediction fails."""


class ScoringError(ClaudeDiscoveryError):
    """Raised when scoring or ranking computation fails."""


# Phase 3 exceptions

class GenerationError(ClaudeDiscoveryError):
    """Raised when molecular generation or design fails."""


# Phase 4 exceptions

class DockingError(ClaudeDiscoveryError):
    """Raised when docking, ligand prep, or structure-based analysis fails."""
