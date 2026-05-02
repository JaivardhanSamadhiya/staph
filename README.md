# Predicting *Staphylococcus aureus* Phage Candidates with Machine Learning

End-to-end, reproducible pipeline that predicts whether a bacteriophage is likely
to infect *Staphylococcus aureus* from genomic features alone.

The pipeline is fully automated: download `requirements`, run `pipeline.py`, and
publication-ready figures are produced in `results/figures/`.

## Background

Antibiotic resistance kills millions of people each year and the WHO classifies
it as a top global health threat. Phage therapy — using bacteriophages to kill
specific bacterial strains — is a promising alternative for treating multidrug
resistant infections such as those caused by *Staphylococcus aureus*. However,
identifying suitable phage–host matches in the lab is slow and expensive.

This project tests the hypothesis that a machine-learning model trained on a
broad genomic feature set can predict whether a given phage is likely to infect
*S. aureus*, providing a fast computational pre-filter that focuses lab effort
on the most promising candidates.

## How to run

```bash
pip install -r requirements.txt
python pipeline.py
```

Run a single step with the `--only` flag (for example,
`python pipeline.py --only train`), or rebuild a step with `--force fetch`.

## Pipeline stages

1. **Fetch** — Downloads the INPHARED phage metadata table and bulk genome FASTA
   from the Millard Lab S3 bucket. INPHARED aggregates and curates every phage
   genome in GenBank monthly.
2. **Build dataset** — Parses metadata, identifies *Staphylococcus* phages
   (positives) and a stratified, host-diverse set of non-*Staphylococcus* phages
   (negatives). Quality filters: complete genomes, plausible length, known host.
3. **Features** — Streams the genome FASTA and extracts purely sequence-derived
   features for each phage: GC content, GC skew, log genome length, canonical
   k-mer frequencies (k = 2, 3, 4) collapsed by reverse complement.
4. **Train** — Clusters phages by k-mer similarity to define non-redundant
   groups, then performs `GroupKFold` cross-validation. Compares Logistic
   Regression, Random Forest, Gradient Boosting and an MLP.
5. **Evaluate** — Holds out a group-disjoint test set, generates ROC, PR,
   confusion-matrix, feature-importance, UMAP and dataset-summary figures, and
   writes a metrics report.

## Why this avoids data leakage

Random splits on phage genomes leak information because many phages are nearly
identical (same genus / cluster). We build a similarity graph on canonical
k-mer frequencies, perform agglomerative clustering with a cosine-distance
threshold, and split *by cluster* using `GroupKFold`. Training and test sets
therefore contain no closely-related phages. We additionally hold out an
entirely separate test set of clusters before any model selection.

## Data sources

* **INPHARED** — Cook *et al.*, PHAGE 3 (2021), 214–223
  ([Millard Lab](https://millardlab.org/bacteriophage-genomics/inphared/),
  [GitHub](https://github.com/RyanCook94/inphared)).
* Underlying genomes originate from NCBI GenBank.

## Outputs

* `data/raw/` — downloaded INPHARED files (cached).
* `data/processed/manifest.csv` — selected phages with labels.
* `data/processed/features.npz` — feature matrix.
* `data/processed/clusters.csv` — non-redundant cluster assignments.
* `results/models/*.joblib` — trained models.
* `results/metrics/cv_results.csv`, `results/metrics/test_metrics.json` —
  cross-validation and test metrics.
* `results/figures/` — publication-ready PDFs and PNGs at 300 dpi.

## Reproducibility

A single random seed (`SEED = 42` in `src/config.py`) is propagated through
sampling, splits and model initialisation. Re-running `pipeline.py` produces
the same outputs.
