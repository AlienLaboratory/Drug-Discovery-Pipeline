# ClaudeDD

A modular command-line drug discovery pipeline built with RDKit and scikit-learn. From data loading to molecular docking, ClaudeDD provides a complete computational chemistry toolkit accessible through a single CLI.

## Features

**Phase 1 - Data Foundation**
- Load molecules from CSV, SDF, SMILES files
- Validate, standardize, and deduplicate structures
- Compute 200+ molecular descriptors and fingerprints (Morgan, MACCS, RDKit)
- Apply drug-likeness filters (Lipinski, Veber, PAINS, Brenk)
- Visualize chemical space (PCA, t-SNE), property distributions, structure grids

**Phase 2 - Virtual Screening & QSAR**
- Substructure (SMARTS), pharmacophore, and similarity-based screening
- Train QSAR models (Random Forest, Gradient Boosted Trees, XGBoost)
- Cross-validation, Y-randomization, applicability domain assessment
- SA score, drug-likeness scoring, multi-objective composite ranking

**Phase 3 - De Novo Molecular Design**
- BRICS fragment recombination
- Scaffold decoration with R-group libraries
- Random chemical mutations (atom swap, bond change, fragment add/remove)
- QSAR-guided genetic algorithm
- Active learning acquisition (greedy, uncertainty, UCB, diversity, balanced)
- Strategy benchmarking (validity, uniqueness, novelty, diversity metrics)

**Phase 4 - Structure-Based Drug Design**
- 3D conformer generation (ETKDGv3) and MMFF/UFF optimization
- Shape-based virtual screening (O3A alignment, Shape Tanimoto, combo scores)
- Protein-ligand interaction analysis (H-bonds, hydrophobic, pi-stacking, salt bridges)
- Interaction fingerprints (PLIF) with Tanimoto comparison
- Optional AutoDock Vina docking wrapper

**Phase 5 - Integration Workflows**
- Hit-to-Lead: profile, score, generate analogs, rank
- De Novo Design: generate, validate, score, prepare 3D
- Lead Optimization: mutate lead, score, shape-screen vs original
- JSON/CSV reporting with provenance tracking

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ClaudeDD.git
cd ClaudeDD

# Install in editable mode
pip install -e .

# Optional: docking support
pip install -e ".[docking]"

# Optional: development tools
pip install -e ".[dev]"
```

**Requirements:** Python >= 3.9, RDKit >= 2024.3.1

## Quick Start

### Profile a molecule library
```bash
claudedd pipeline quick-profile \
    --input molecules.csv \
    --output-dir output/profile
```

### Full hit-to-lead workflow
```bash
claudedd workflow hit-to-lead \
    --input molecules.csv \
    --output-dir output/hit2lead \
    --n-generate 100 --top-n 30
```

### De novo molecular design
```bash
claudedd workflow denovo \
    --input seeds.csv \
    --output-dir output/denovo \
    --strategy brics -n 200 --top-n 50
```

### Lead optimization
```bash
claudedd workflow optimize \
    --lead "CC(=O)Oc1ccccc1C(=O)O" \
    --output-dir output/optimize \
    --n-analogs 100 --top-n 20
```

## CLI Reference

```
claudedd
  data          Load, standardize, export molecular data
  analyze       Compute properties, descriptors, fingerprints, filters
  viz           Structure grids, property plots, chemical space
  pipeline      Predefined pipeline workflows (quick-profile, screen-and-score)
  screen        Substructure, pharmacophore, similarity screening
  model         Train, evaluate, predict with QSAR models
  generate      De novo generation (BRICS, scaffold, mutate, GA, active learning)
  benchmark     Compare generation strategies
  dock          Ligand prep, shape screening, interactions, Vina docking
  workflow      End-to-end workflows (hit-to-lead, denovo, optimize)
```

## Architecture

```
src/claudedd/
  core/           Models, config, constants, exceptions, logging
  phase1/
    data/         Loaders, validators, standardizer, writers
    analysis/     Properties, descriptors, fingerprints, filters, similarity
    visualization/  Structure rendering, distributions, chemical space
  phase2/
    screening/    Substructure, pharmacophore, similarity
    qsar/         Data prep, models (RF/GBT/XGBoost), evaluation, prediction
    scoring/      SA score, drug-likeness, multi-objective, ranking
  phase3/
    generation/   BRICS, scaffold decoration, mutations, genetic algorithm
    active_learning/  Uncertainty, diversity, acquisition strategies
    benchmarking/ Metrics, benchmark runner, comparison
  phase4/
    ligand_prep/  Conformers, optimization, protonation, preparation
    shape_screening/  Alignment (O3A), scoring, virtual screening
    interactions/ Geometry, contacts (H-bond/hydrophobic/pi/salt), PLIF
    docking/      Grid, protein prep, Vina wrapper
  phase5/
    workflows/    Hit-to-lead, de novo design, lead optimization
    reporting/    Summary, export, provenance
  cli/            Click command groups (10 groups, 40+ commands)
  utils/          Chemistry, I/O, parallelization utilities
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/test_phase1/ -v
pytest tests/test_phase5/ -v

# With coverage
pytest tests/ --cov=claudedd --cov-report=html
```

**348 tests** covering all 5 phases.

## Sample Data

The `data/sample/` directory contains 50 real drug molecules for testing:

```bash
# Quick test with sample data
claudedd pipeline quick-profile -i data/sample/sample_molecules.csv -o output/test
```

## License

MIT License. See [LICENSE](LICENSE) for details.
