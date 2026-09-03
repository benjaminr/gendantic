from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from .distributions import (
    Beta,
    Binomial,
    Categorical,
    Conditional,
    Constraints,
    CopulaType,
    Correlations,
    DistributionSpec,
    Exponential,
    LogNormal,
    Normal,
    Ordering,
    Poisson,
    Range,
    Uniform,
)
from .export import to_dataframe
from .fidelity import (
    CorrelationFidelity,
    FidelityReport,
    FieldFidelity,
    fidelity_report,
)
from .generator import (
    generate_synthetic_data,
    generate_synthetic_data_batch,
    generate_synthetic_data_batch_sync,
    generate_synthetic_data_sync,
)
from .llm_driven_analyser import LLMDrivenModelAnalyser
from .model_generator import (
    CodeValidationError,
    extend_model_with_correlations,
    extend_model_with_distributions,
    generate_model_from_description,
)
from .relational import (
    Dataset,
    ForeignKey,
    ForeignKeySpec,
    PrimaryKey,
    generate_dataset,
    generate_dataset_sync,
)
from .sampler import DistributionSampler

try:
    __version__ = _package_version("gendantic")
except PackageNotFoundError:  # pragma: no cover - running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    # Generation
    "generate_synthetic_data",
    "generate_synthetic_data_batch",
    "generate_synthetic_data_sync",
    "generate_synthetic_data_batch_sync",
    "generate_model_from_description",
    "extend_model_with_distributions",
    "extend_model_with_correlations",
    # Relational / multi-model
    "generate_dataset",
    "generate_dataset_sync",
    "PrimaryKey",
    "ForeignKey",
    "ForeignKeySpec",
    "Dataset",
    # Export
    "to_dataframe",
    # Fidelity validation
    "fidelity_report",
    "FidelityReport",
    "FieldFidelity",
    "CorrelationFidelity",
    # Distribution specs
    "DistributionSpec",
    "Normal",
    "Uniform",
    "Categorical",
    "LogNormal",
    "Exponential",
    "Poisson",
    "Beta",
    "Binomial",
    # Correlations and Copulas
    "Correlations",
    "CopulaType",
    # Conditional distributions and cross-field constraints
    "Conditional",
    "Range",
    "Constraints",
    "Ordering",
    # Utilities
    "DistributionSampler",
    "LLMDrivenModelAnalyser",
    # Exceptions
    "CodeValidationError",
]
