"""Chemical constants, default thresholds, and enumeration values."""

# Lipinski Rule of Five
LIPINSKI_MW_MAX = 500.0
LIPINSKI_LOGP_MAX = 5.0
LIPINSKI_HBD_MAX = 5
LIPINSKI_HBA_MAX = 10

# Veber filter (oral bioavailability)
VEBER_TPSA_MAX = 140.0
VEBER_ROTATABLE_BONDS_MAX = 10

# Ghose filter
GHOSE_MW_RANGE = (160.0, 480.0)
GHOSE_LOGP_RANGE = (-0.4, 5.6)
GHOSE_ATOM_COUNT_RANGE = (20, 70)
GHOSE_REFRACTIVITY_RANGE = (40.0, 130.0)

# Supported file formats
SUPPORTED_INPUT_FORMATS = {"sdf", "smi", "smiles", "csv", "tsv", "pdb", "mol", "mol2"}
SUPPORTED_OUTPUT_FORMATS = {"sdf", "csv", "smi", "json"}

# Fingerprint type defaults
FINGERPRINT_TYPES = {
    "morgan": {"radius": 2, "nbits": 2048},
    "maccs": {},
    "rdkit": {"minPath": 1, "maxPath": 7, "fpSize": 2048},
    "atom_pair": {"nBits": 2048},
    "topological_torsion": {"nBits": 2048},
}

# Default descriptors for quick computation
DEFAULT_DESCRIPTORS = [
    "MolWt", "ExactMolWt", "MolLogP", "MolMR",
    "NumHDonors", "NumHAcceptors", "TPSA",
    "NumRotatableBonds", "RingCount", "NumAromaticRings",
    "FractionCSP3", "HeavyAtomCount", "NumHeteroatoms",
    "LabuteASA",
]

# Similarity metrics
SIMILARITY_METRICS = [
    "tanimoto", "dice", "cosine", "sokal", "russel",
    "kulczynski", "mcconnaughey",
]

# ── Bioactivity Data Constants ────────────────────────────────

# Supported ChEMBL activity types
BIOACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]
BIOACTIVITY_DEFAULT_TYPE = "IC50"

# ChEMBL API pagination
CHEMBL_PAGE_SIZE = 1000
CHEMBL_REQUEST_DELAY = 0.5  # seconds between paginated requests

# Unit normalization target
BIOACTIVITY_TARGET_UNIT = "nM"

# Relation filter defaults
BIOACTIVITY_EXACT_RELATIONS = {"="}
BIOACTIVITY_PERMISSIVE_RELATIONS = {"=", "<", "<="}

# Outlier removal IQR multiplier
BIOACTIVITY_IQR_MULTIPLIER = 1.5

# Default IC50 thresholds (nM) for activity labeling
BIOACTIVITY_ACTIVE_THRESHOLD_NM = 1000.0     # IC50 < 1 uM = active
BIOACTIVITY_INACTIVE_THRESHOLD_NM = 10000.0  # IC50 >= 10 uM = inactive

# Confidence scoring weights
BIOACTIVITY_CONFIDENCE_WEIGHTS = {
    "exact_measurement": 0.4,
    "pchembl_available": 0.3,
    "trusted_assay": 0.2,
    "unit_consistent": 0.1,
}

# Metadata fields stored from ChEMBL activity records
CHEMBL_ACTIVITY_FIELDS = [
    "pchembl_value",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "target_chembl_id",
    "document_chembl_id",
    "molecule_chembl_id",
    "standard_type",
    "standard_value",
    "standard_units",
    "standard_relation",
]

# ── Phase 2 Constants ──────────────────────────────────────────

# Screening defaults
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_SIMILARITY_FP_TYPE = "morgan_r2_2048"
DEFAULT_SIMILARITY_METRIC = "tanimoto"

# Pharmacophore feature types
PHARMACOPHORE_FEATURE_FAMILIES = [
    "Donor", "Acceptor", "NegIonizable", "PosIonizable",
    "Aromatic", "Hydrophobe", "LumpedHydrophobe",
]

# QSAR model types
QSAR_MODEL_TYPES = ["random_forest", "gradient_boosting", "xgboost", "svr"]
QSAR_TASK_TYPES = ["regression", "classification"]

# Default model hyperparameters
QSAR_DEFAULT_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
        "n_jobs": -1,
    },
    "gradient_boosting": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
    },
    "svr": {
        "kernel": "rbf",
        "C": 1.0,
        "epsilon": 0.1,
        "gamma": "scale",
    },
}

# QSAR evaluation
QSAR_DEFAULT_CV_FOLDS = 5
QSAR_Y_RANDOMIZATION_ITERATIONS = 10

# Scoring weights
SCORING_DEFAULT_WEIGHTS = {
    "predicted_activity": 0.4,
    "drug_likeness": 0.3,
    "sa_score": 0.3,
}

# SA Score range (1 = easy, 10 = hard)
SA_SCORE_MIN = 1.0
SA_SCORE_MAX = 10.0

# Drug-likeness score components
DRUG_LIKENESS_COMPONENTS = ["qed", "lipinski", "veber", "pains"]

# ── Phase 3 Constants ──────────────────────────────────────────

# BRICS generation
BRICS_MAX_FRAGMENTS = 500
BRICS_MAX_BUILD_MOLECULES = 1000
BRICS_DEFAULT_MAX_DEPTH = 2
BRICS_MIN_FRAGMENT_FREQUENCY = 2

# Genetic algorithm
GA_POPULATION_SIZE = 100
GA_NUM_GENERATIONS = 50
GA_MUTATION_RATE = 0.3
GA_CROSSOVER_RATE = 0.7
GA_ELITE_FRACTION = 0.1
GA_TOURNAMENT_SIZE = 3

# Mutation types for molecular mutations
MUTATION_TYPES = [
    "atom_swap",       # Change atom element type
    "bond_change",     # Change bond order
    "add_fragment",    # Attach small fragment
    "remove_fragment", # Remove substituent
    "ring_mutation",   # Open/close ring
]

# Common medicinal chemistry R-groups (SMILES)
R_GROUP_LIBRARY = [
    "C",       # methyl
    "CC",      # ethyl
    "C(C)C",   # isopropyl
    "O",       # hydroxyl
    "N",       # amino
    "F",       # fluorine
    "Cl",      # chlorine
    "Br",      # bromine
    "OC",      # methoxy
    "C(F)(F)F",  # trifluoromethyl
    "C(=O)O",   # carboxylic acid
    "C(=O)N",   # amide
    "C#N",       # nitrile
    "S(=O)(=O)N",  # sulfonamide
    "c1ccccc1",    # phenyl
    "C1CC1",       # cyclopropyl
    "C(=O)",       # carbonyl
    "S",           # thiol
    "OCC",         # ethoxy
    "NC",          # methylamino
]

# Atoms for atom-swap mutations
MUTATION_ATOM_TYPES = [6, 7, 8, 9, 16, 17]  # C, N, O, F, S, Cl

# Benchmarking metrics
BENCHMARK_METRICS = [
    "validity", "uniqueness", "novelty",
    "internal_diversity", "drug_likeness_rate",
    "mean_sa_score", "mean_qed",
]

# Active learning
AL_DEFAULT_BATCH_SIZE = 20
AL_STRATEGIES = ["greedy", "uncertainty", "ucb", "diversity", "balanced"]
AL_DEFAULT_KAPPA = 1.0  # UCB exploration parameter

# ── Phase 4 Constants ──────────────────────────────────────────

# Conformer generation
CONFORMER_NUM_CONFS = 50
CONFORMER_MAX_ATTEMPTS = 500
CONFORMER_PRUNE_RMSD = 0.5  # Angstroms

# Force field optimization
FORCE_FIELD_TYPES = ["MMFF", "UFF"]
OPTIMIZATION_MAX_ITERS = 500

# Shape-based screening
SHAPE_SCORE_TYPES = ["tanimoto", "protrusion", "combo"]
SHAPE_DEFAULT_METRIC = "tanimoto"
SHAPE_DEFAULT_THRESHOLD = 0.5

# Docking (AutoDock Vina)
VINA_EXHAUSTIVENESS = 8
VINA_NUM_MODES = 9
VINA_ENERGY_RANGE = 3.0
GRID_PADDING = 10.0  # Angstroms around ligand

# Interaction distance cutoffs (Angstroms)
INTERACTION_HBOND_DISTANCE = 3.5
INTERACTION_HBOND_ANGLE = 120.0  # degrees
INTERACTION_HYDROPHOBIC_DISTANCE = 4.5
INTERACTION_PI_STACKING_DISTANCE = 5.5
INTERACTION_SALT_BRIDGE_DISTANCE = 4.0

# Hydrophobic atom types
HYDROPHOBIC_ELEMENTS = {6}  # Carbon (non-polar)
# Donor/acceptor atoms
HBOND_DONORS = {7, 8}    # N, O
HBOND_ACCEPTORS = {7, 8, 9, 16}  # N, O, F, S
# Aromatic check for pi-stacking
PI_STACKING_ELEMENTS = {6, 7}  # C, N in aromatic rings
