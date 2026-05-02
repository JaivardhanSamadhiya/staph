"""Global configuration: paths, constants, dataset sizing, hyperparameters."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
AUX_DIR = DATA_DIR / "aux"

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
METRIC_DIR = RESULTS_DIR / "metrics"

for _d in (RAW_DIR, PROC_DIR, AUX_DIR, FIG_DIR, MODEL_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# NCBI E-utilities etiquette (override with ENTREZ_EMAIL in the environment).
ENTREZ_EMAIL = "phage-host-pipeline@user.local"
ENTREZ_TOOL = "staph_phage_pipeline"

# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------
# Most recent INPHARED snapshot we verified is publicly available.
# Older snapshots will continue to work; newer ones can be set here when they
# are published.
INPHARED_DATE = "31August2025"
INPHARED_BASE = "https://millardlab-inphared.s3.climb.ac.uk"
INPHARED_META_URL = f"{INPHARED_BASE}/{INPHARED_DATE}_data.tsv.gz"
# INPHARED splits its FASTA distribution into a RefSeq subset (~5k phages,
# ~110 MB compressed) and the rest of GenBank (~30k phages, ~530 MB
# compressed). We need both to cover the full manifest.
INPHARED_REFSEQ_URL = f"{INPHARED_BASE}/{INPHARED_DATE}_genomes.fa.gz"
INPHARED_GENBANK_URL = f"{INPHARED_BASE}/{INPHARED_DATE}_genomes_excluding_refseq.fa.gz"

META_PATH = RAW_DIR / "inphared_data.tsv.gz"
REFSEQ_FASTA_PATH = RAW_DIR / "inphared_refseq_genomes.fa.gz"
GENBANK_FASTA_PATH = RAW_DIR / "inphared_genbank_genomes.fa.gz"
GENOME_FASTA_PATHS = (REFSEQ_FASTA_PATH, GENBANK_FASTA_PATH)

# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
TARGET_HOST = "Staphylococcus"

# Quality filters applied to candidate phages.
MIN_GENOME_LEN = 5_000
MAX_GENOME_LEN = 500_000
EXCLUDE_HOST_VALUES = {"", "Unknown"}

# We use *all* qualifying Staphylococcus phages and a **matched-count** negative
# set for training, but negatives are quota-sampled across phylogenetic strata
# (NCBI lineage distance to Staphylococcus) so the classifier is not dominated
# by the most phylogenetically distant out-groups.
NEG_PER_GENUS_CAP = 80          # per-genus cap within each stratum
MIN_GENERA_PER_NEG = 12         # unique genera required in final negative set

# Target fraction of negatives drawn from each tertile bucket (near | mid | far
# defined from intra-negative distance quantiles). Residual mass is allocated
# to ``far`` if a stratum is exhausted.
NEG_STRATUM_FRACTIONS = {"near": 0.32, "mid": 0.32, "far": 0.32, "unresolved": 0.04}

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
KMER_SIZES = (2, 3, 4)

# ---------------------------------------------------------------------------
# Evaluation: prevalence scenarios + screening threshold
# ---------------------------------------------------------------------------
# Artificial prevalence grid for deployment-style reporting (fraction positive).
PREVALENCE_GRID = (0.01, 0.02, 0.05, 0.10, 0.50)

# Tuning prevalence for recall-target threshold selection: use empirical INPHARED
# Staphylococcus / all filtered phages (written to ``corpus_stats.json``).
# Default when stats file missing:
CALIBRATION_PREVALENCE_FLOOR = 0.02

# OOF recall target for the primary screening operating point (primary model).
TARGET_RECALL_OOF = 0.95

# Cluster bootstrap resamples for 95% confidence intervals on held-out metrics.
# Override at train time with ``python -m src.train_models --bootstrap-draws N``.
BOOTSTRAP_CLUSTER_DRAWS = 2000

# When a target prevalence π needs more held-out negatives than exist, we draw
# negatives **with replacement** this many times and report mean ± std of metrics.
PREVALENCE_MONTE_CARLO_NEGATIVE_DRAWS = 512

# Optional multi-seed robustness (``python -m src.train_models --seed N``).
ROBUSTNESS_SEEDS = tuple(range(42, 47))  # 42–46 inclusive

# ---------------------------------------------------------------------------
# Leakage control
# ---------------------------------------------------------------------------
# Cosine-distance threshold for collapsing near-duplicate phages into a single
# group during clustering. Single-linkage agglomerative clustering at this
# distance reproduces the connected components of the "near-duplicate graph",
# which is exactly the leakage-safe equivalence relation. Empirically, a value
# of ~0.001 isolates outbreak/strain-level near-duplicates while leaving
# distinct phage species in different clusters.
CLUSTER_DISTANCE = 0.001
CLUSTER_LINKAGE = "single"

# Held-out test fraction (computed at the cluster level, not the genome level).
TEST_CLUSTER_FRACTION = 0.20

CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIG_DPI = 300
FIG_FORMATS = ("pdf", "png")
