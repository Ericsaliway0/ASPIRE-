"""Parse MAF files and compute per-gene somatic mutation frequency."""

from pathlib import Path
from collections import defaultdict
from typing import Set, Tuple

import pandas as pd


def process_maf(cancer_dir: Path) -> Tuple[Set[Tuple[str, str]], Set[str]]:
    """Read every MAF file under `cancer_dir` and collect unique
    (gene, sample) mutation pairs plus the set of sample barcodes seen.

    A gene mutated multiple times in the same sample is only counted once;
    downstream frequency is "fraction of samples with >=1 mutation in this
    gene", not total mutation count.
    """
    gene_sample_pairs: Set[Tuple[str, str]] = set()
    sample_set: Set[str] = set()

    for maf_file in cancer_dir.rglob("*.maf*"):
        try:
            print(f"[PROCESS] {maf_file.name}")

            df = pd.read_csv(
                maf_file,
                sep="\t",
                comment="#",
                usecols=["Hugo_Symbol", "Tumor_Sample_Barcode"],
                dtype={"Hugo_Symbol": str, "Tumor_Sample_Barcode": str},
                compression="infer",
            )

            df = df.dropna().drop_duplicates()

            pairs = set(zip(df["Hugo_Symbol"], df["Tumor_Sample_Barcode"]))
            gene_sample_pairs.update(pairs)
            sample_set.update(df["Tumor_Sample_Barcode"])

        except Exception as e:
            print(f"[ERROR] {maf_file}: {e}")
            continue

    print(
        f"[SUMMARY] {len(gene_sample_pairs)} gene-sample pairs, "
        f"{len(sample_set)} samples"
    )

    return gene_sample_pairs, sample_set


def compute_mutation_frequency(cancer_dir: Path) -> pd.DataFrame:
    """Compute per-gene mutation frequency (fraction of samples mutated)
    for a single cancer type's MAF directory.
    """
    gene_sample_pairs, sample_set = process_maf(cancer_dir)
    gene_counts = defaultdict(int)

    for gene, _sample in gene_sample_pairs:
        gene_counts[gene] += 1

    num_samples = len(sample_set)
    results = [
        {
            "gene": gene,
            "mutation_frequency": count / num_samples if num_samples > 0 else 0,
        }
        for gene, count in gene_counts.items()
    ]

    return pd.DataFrame(results)
