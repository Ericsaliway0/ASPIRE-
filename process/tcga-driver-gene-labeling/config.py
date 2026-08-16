"""Central configuration for the TCGA driver-gene labeling pipeline.

All paths are relative to the repository root unless overridden via
environment variables or CLI flags in the scripts under `scripts/`.
"""

from pathlib import Path

# --- Directories -----------------------------------------------------------
DATA_DIR = Path("../data")
TCGA_MAF_DIR = DATA_DIR / "tcga_maf"
PROCESSED_DIR = DATA_DIR / "processed"
REACTOME_DIR = DATA_DIR / "reactome"

MUTATION_FREQUENCY_FILE = DATA_DIR / "mutation_frequency_tcga.csv"
GENCODE_GENES_FILE = DATA_DIR / "gencode_genes.csv"
DISEASE_PATHWAY_GENES_FILE = DATA_DIR / "cancer_pathway_genes.csv"
GENE_LABELS_FILE = PROCESSED_DIR / "gene_labels_driver_vs_nondriver.csv"

# --- External reference files (must be provided by the user) ---------------
DATA_DIR_ = Path("../../data")
GENCODE_GTF_FILE = DATA_DIR_ / "gencode.v49.basic.annotation.gtf"
CGC_CENSUS_FILE = DATA_DIR_ / "Census_allWed.tsv"
REACTOME_GMT_FILE = DATA_DIR_ / "ReactomePathways.gmt"
REACTOME_RELATIONS_FILE = DATA_DIR_ / "reactome_relations.csv"

# Additional driver-gene reference databases used to build the expanded
# exclusion set for negative sampling (see src/negative_sampling.py).
# Set to None (or leave the file absent) to skip a source you don't have.
NCG_FILE = DATA_DIR / "NCG_cancergenes.tsv"
INTOGEN_FILE = DATA_DIR / "IntOGen_drivers.tsv"
BAILEY_FILE = DATA_DIR / "Bailey2018_drivers.tsv"

# Gene expression reference (e.g. median TPM per gene across GTEx/TCGA
# normal samples), used for covariate-matched negative sampling.
GENE_EXPRESSION_FILE = DATA_DIR / "gene_median_expression.csv"

# --- GDC API -----------------------------------------------------------------
GDC_API_FILES = "https://api.gdc.cancer.gov/files"
GDC_API_DATA = "https://api.gdc.cancer.gov/data"

CANCER_TYPES = [
    "BRCA", "LUAD", "LUSC", "HNSC",
    "KIRC", "KIRP", "PRAD", "THCA",
    "STAD", "ESCA", "COAD", "READ",
    "LIHC", "UCEC", "BLCA", "CESC",
    "SKCM", "OV", "LAML", "GBM",
    "LGG", "SARC", "PAAD", "PCPG",
]

# --- Reactome pathway hierarchy scope --------------------------------------
# Top-level "Disease" pathway. Genes are kept only if they belong to a
# pathway that is a descendant of this node, rather than to any pathway
# in the full Reactome catalog.
REACTOME_DISEASE_ROOT = "R-HSA-1643685"

# --- Gene labeling filters ---------------------------------------------------
USE_MUTATION_FILTER = True
USE_PATHWAY_FILTER = True
MUTATION_FREQUENCY_THRESHOLD = 0.01  # exclude genes at/above this frequency
RANDOM_SEED = 42

# --- Negative (non-driver) sampling -----------------------------------------
# If True, run covariate-matched negative sampling (length + expression,
# optionally + mutation rate) after the mutation/pathway filters, instead of
# treating every surviving candidate as an equally valid negative.
USE_MATCHED_NEGATIVE_SAMPLING = True
NEGATIVES_PER_POSITIVE = 1
# Mutation frequency is available only for genes with observed mutations.
# Requiring it drops most otherwise eligible negatives, so retain the full
# length/expression-matched candidate pool by default.
MATCH_ON_MUTATION_RATE = False
