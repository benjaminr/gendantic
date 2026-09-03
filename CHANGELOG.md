# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Inherited fields are now honoured.** Distribution specs, `Conditional`,
  `PrimaryKey` and `ForeignKey` markers declared on a parent model were
  silently ignored on subclasses (the fields fell through to LLM generation and
  keys went unassigned). Field introspection now reads Pydantic's
  `model_fields`, which includes inherited fields.
- **`from __future__ import annotations` no longer disables gendantic.** Under
  postponed evaluation the class annotations are strings, so no markers were
  found at all; the same `model_fields`-based introspection fixes this.
- `Annotated[Optional[int], ...]` (and `Optional[Annotated[int, ...]]`)
  distribution fields are now sampled and rounded as integers instead of
  emitting floats that failed Pydantic validation on every record.
- `fidelity_report` no longer raises on categorical fields. Weights that sum
  only approximately to 1.0 (within `Categorical`'s tolerance) are renormalised
  before the chi-square test, and a value outside the declared categories is
  reported as a failed field rather than crashing scipy's chi-square.
- LLM-generated enum and nested-model fields no longer produce dangling
  `$ref`s: the model's `$defs` travel with the partial schema and are hoisted to
  the root of the structured-output schema where the references resolve.
- `extend_model_with_correlations` now works on models that use `Conditional`,
  `Range`, `Constraints` or `Ordering`: those names are available (and
  allow-listed) in the code sandbox, and the model source round-trip keeps
  numeric `Field(ge=..., le=..., gt=..., lt=...)` bounds and field defaults,
  which were previously dropped from the regenerated model.
- `generate_synthetic_data` raises a clear `ValueError` for a negative `count`
  instead of silently returning an empty list.

### Added

- `gendantic.__version__` (from installed package metadata).
- `generate_dataset` / `generate_dataset_sync` accept `max_concurrency`, one
  budget shared across every model in the dataset, matching the other entry
  points.

### Changed

- The sdist no longer bundles notebooks, the lockfile, CI workflows or dotfiles
  (`MANIFEST.in`); it shrinks from ~480 KB to ~145 KB.
- CI (and the release gate) now run `ruff format --check`; the tree has been
  reformatted so the check passes.
- The model-analysis prompt no longer asks the LLM for a
  `field_generation_strategies` block and non-existent distribution types that
  the strict response schema could never carry.

## [0.1.0] - 2026-09-03

Initial release.

### Added

- **Statistical distributions** via `Annotated` types: `Normal`, `Uniform`,
  `Categorical`, `LogNormal`, `Exponential`, `Poisson`, `Beta`, `Binomial` —
  sampled deterministically with numpy/scipy from a `seed`.
- **Correlated fields** using copulas (`Correlations`, `CopulaType`): Gaussian,
  Student's t, Clayton, Gumbel, and Frank, preserving each field's marginal.
- **Conditional distributions** (`Conditional`): switch a field's distribution
  on another field's value, keyed by category or numeric `Range`, resolved in
  dependency order with cycle detection.
- **Cross-field ordering constraints** (`Constraints`, `Ordering`) with two
  strategies: `method="sort"` (default) and marginal-preserving
  `method="resample"`.
- **Relational generation** (`generate_dataset`, `PrimaryKey`, `ForeignKey`):
  multi-model datasets with referential integrity, composite and self/mutual
  keys, nullable foreign keys, and join tables.
- **Database binding** (`gendantic.db`): `reflect_schema`, `infer_distributions`,
  and `load_dataset` to reflect a live schema, generate data, and load it back.
- **LLM-driven authoring**: `generate_model_from_description`,
  `extend_model_with_distributions`, `extend_model_with_correlations`, with
  AST-allowlist sandboxing of generated code.
- **Fidelity validation** (`fidelity_report`): KS / chi-square goodness-of-fit
  and correlation checks, with per-branch checks for conditional fields.
- **Export**: `to_dataframe` and `Dataset.to_dataframes` (pandas extra).
- **Async-first API** with `*_sync` wrappers and batch generation.

### Changed

- **Per-pair copulas** are now modelled as a vine (1-truncated R-vine / Markov
  tree): each correlation pair keeps its own family and strength instead of
  collapsing to a single averaged copula. Mixed families are honoured per pair;
  the pairs must form a forest (cycles and duplicate pairs raise), and a
  negative correlation on a positive-only Archimedean family (Clayton, Gumbel)
  raises. Homogeneous Gaussian / Student-t specs still use the exact full
  correlation matrix.
- **Bounded fields** (`ge` / `le` / `gt` / `lt`) are now **truncated** via the
  inverse CDF instead of clamped, so the boundary no longer accumulates a spike
  of mass and the field stays faithful to its declared shape. `fidelity_report`
  is truncation-aware and compares against the truncated distribution.
- Correlation fidelity now uses the rank statistic the copula family targets:
  Kendall's τ for Archimedean families, Spearman's ρ otherwise.
- Generation now returns **exactly** the requested count: records that fail
  Pydantic validation are regenerated (bounded top-up rounds) instead of being
  silently dropped, so `generate_synthetic_data` no longer returns a short
  batch. If records keep failing validation it raises a clear error. Relational
  generation raises on any validation failure (rows carry engine-assigned keys
  and cannot be dropped without breaking referential integrity).
- Dropped the direct `httpx` runtime dependency: it was never imported
  directly and is still available transitively via litellm.
- LLM field-generation calls are now concurrency-bounded instead of firing
  every record batch at once, so generating large batches no longer floods the
  provider or triggers rate-limit rejections. The cap defaults to 8 and is
  configurable per call via the `max_concurrency` argument on
  `generate_synthetic_data` (and its batch/sync variants), or deployment-wide
  via the `GENDANTIC_MAX_CONCURRENCY` environment variable.
- `generate_synthetic_data_batch` now shares a single concurrency budget across
  all contexts instead of giving each context its own, so the total in-flight
  LLM calls stays within `max_concurrency` rather than reaching
  `len(contexts) * max_concurrency`.
- Ship a `py.typed` marker so downstream type checkers pick up the package's
  inline type hints (the `Typing :: Typed` classifier was previously unbacked).

### Fixed

- Student-t conditional (copula h-inverse) now uses `df+1` for the inner
  quantile, fixing the conditional-sampling round-trip.
- `Categorical` weights that sum to only ~1.0 (within the accepted 0.01
  tolerance) are now renormalised internally, so they no longer crash numpy's
  sampler, which requires probabilities summing to exactly 1.0.
- Clayton/Gumbel copulas with a target correlation of exactly 1.0 no longer
  divide by zero: the parameter is clamped just below the singularity to a
  large-but-finite value (approaching comonotonicity).

[Unreleased]: https://github.com/benjaminr/gendantic/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/benjaminr/gendantic/releases/tag/v0.1.0
