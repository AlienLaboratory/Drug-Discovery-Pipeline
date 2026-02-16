"""Pipeline configuration system with YAML loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from claudedd.core.exceptions import ConfigurationError


@dataclass
class FilterConfig:
    lipinski: bool = True
    lipinski_max_violations: int = 1
    veber: bool = False
    pains: bool = True
    brenk: bool = False
    custom_mw_range: Optional[Tuple[float, float]] = None
    custom_logp_range: Optional[Tuple[float, float]] = None


@dataclass
class DescriptorConfig:
    compute_all: bool = False
    descriptor_names: List[str] = field(default_factory=lambda: [
        "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors",
        "TPSA", "NumRotatableBonds", "RingCount",
        "FractionCSP3", "HeavyAtomCount",
    ])


@dataclass
class FingerprintConfig:
    morgan: bool = True
    morgan_radius: int = 2
    morgan_nbits: int = 2048
    maccs: bool = True
    rdkit_fp: bool = False
    rdkit_fp_nbits: int = 2048


@dataclass
class VisualizationConfig:
    output_dir: str = "./output/plots"
    format: str = "png"
    dpi: int = 300
    figsize: Tuple[int, int] = (10, 6)


# Phase 2 configuration

@dataclass
class ScreeningConfig:
    """Configuration for virtual screening operations."""
    similarity_threshold: float = 0.7
    similarity_metric: str = "tanimoto"
    similarity_fp_type: str = "morgan_r2_2048"
    similarity_aggregation: str = "max"
    substructure_count_matches: bool = False


@dataclass
class QSARConfig:
    """Configuration for QSAR modeling."""
    model_type: str = "random_forest"
    task: str = "regression"
    activity_column: str = "activity"
    feature_source: str = "descriptors"
    fp_type: str = "morgan_r2_2048"
    test_size: float = 0.2
    split_method: str = "random"
    scale_features: bool = True
    cv_folds: int = 5
    y_randomization: bool = True
    y_randomization_iterations: int = 10
    random_state: int = 42


@dataclass
class ScoringConfig:
    """Configuration for scoring and ranking."""
    activity_weight: float = 0.4
    drug_likeness_weight: float = 0.3
    sa_score_weight: float = 0.3
    normalize_activity: bool = True
    top_n: int = 50


@dataclass
class PipelineConfig:
    name: str = "default_run"
    output_dir: str = "./output"
    log_level: str = "INFO"
    n_jobs: int = 1
    random_seed: int = 42

    input_file: str = ""
    input_format: str = "auto"
    smiles_column: str = "smiles"
    id_column: str = "id"

    standardize: bool = True
    strip_salts: bool = True
    neutralize: bool = True
    remove_stereo: bool = False

    filters: FilterConfig = field(default_factory=FilterConfig)
    descriptors: DescriptorConfig = field(default_factory=DescriptorConfig)
    fingerprints: FingerprintConfig = field(default_factory=FingerprintConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Phase 2
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    qsar: QSARConfig = field(default_factory=QSARConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {path}: {e}")

        config = cls()
        sub_config_map = {
            "filters": FilterConfig,
            "descriptors": DescriptorConfig,
            "fingerprints": FingerprintConfig,
            "visualization": VisualizationConfig,
            "screening": ScreeningConfig,
            "qsar": QSARConfig,
            "scoring": ScoringConfig,
        }

        for key, value in data.items():
            if key in sub_config_map and isinstance(value, dict):
                attr = getattr(config, key)
                for sub_key, sub_value in value.items():
                    if hasattr(attr, sub_key):
                        setattr(attr, sub_key, sub_value)
            elif hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_yaml(self, path: str) -> None:
        data = dataclasses.asdict(self)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
