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

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODEL_DIR = RESULTS_DIR / "models"
METRIC_DIR = RESULTS_DIR / "metrics"

for _d in (RAW_DIR, PROC_DIR, FIG_DIR, MODEL_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

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

# Per-class budget. We use *all* qualifying Staphylococcus phages and sample a
# matched number of non-Staphylococcus phages, stratified across host genera so
# the model is not exposed to one dominant out-group.
NEG_PER_GENUS_CAP = 80          # cap to encourage host diversity
MIN_GENERA_PER_NEG = 12         # require at least this many distinct host genera

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
KMER_SIZES = (2, 3, 4)

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
