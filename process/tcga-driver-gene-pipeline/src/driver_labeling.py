"""Build the driver vs. non-driver gene label set.

Pipeline:
    1. Start from the protein-coding gene universe (GENCODE).
    2. Remove known driver genes (COSMIC Cancer Gene Census) -> candidate
       non-drivers.
    3. Optionally remove genes with mutation frequency >= threshold in any
       TCGA cancer type.
    4. Optionally remove genes belonging to a disease-related Reactome
       pathway.
"""

from pathlib import Path
from typing import Set

import pandas as pd


def load_driver_genes(cgc_file: Path) -> Set[str]:
    """Load the COSMIC Cancer Gene Census and return the set of driver gene
    symbols.
    """
    cgc = pd.read_csv(cgc_file, sep="\t")
    return set(cgc["Symbol"].dropna())


def apply_mutation_frequency_filter(
    non_drivers: Set[str],
    mutation_frequency_file: Path,
    threshold: float = 0.01,
) -> Set[str]:
    """Remove genes whose mutation frequency is >= `threshold` in any TCGA
    cancer type.

    Genes that never appear in the mutation-frequency table (i.e. were never
    observed mutated in any sample) are treated as having a frequency of 0
    and are kept, rather than being dropped by omission. `mutation_frequency
    _tcga.csv` has one row per (gene, cancer_type) pair, so the maximum
    frequency across cancer types is used to decide whether a gene is
    "frequently mutated in any cancer type" — a gene rare in one cancer but
    common in another should still be excluded.
    """
    mutation_df = pd.read_csv(mutation_frequency_file)
    max_freq_per_gene = mutation_df.groupby("gene")["mutation_frequency"].max()

    return {
        gene
        for gene in non_drivers
        if max_freq_per_gene.get(gene, 0.0) < threshold
    }


def apply_pathway_filter(
    non_drivers: Set[str], pathway_genes: Set[str]
) -> Set[str]:
    """Remove genes that belong to a disease-related Reactome pathway."""
    return non_drivers - pathway_genes


def build_gene_labels(
    driver_genes: Set[str], non_driver_genes: Set[str]
) -> pd.DataFrame:
    """Combine driver (label=1) and non-driver (label=0) genes into a single
    labeled DataFrame.
    """
    labels = [(gene, 1) for gene in driver_genes]
    labels += [(gene, 0) for gene in non_driver_genes]
    return pd.DataFrame(labels, columns=["gene", "label"])
