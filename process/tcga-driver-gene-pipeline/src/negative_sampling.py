"""Expanded driver-gene exclusion and covariate-matched negative sampling.

Two upgrades over the earlier non-driver construction:

1. **Broader exclusion set.** A candidate gene is only eligible to be a
   negative if it is absent from *all four* driver references — NCG
   (Network of Cancer Genes), CGC (COSMIC Cancer Gene Census), IntOGen, and
   Bailey et al. 2018 — not just CGC. A gene with driver evidence in any one
   of these databases should not be usable as a "confirmed non-driver."

2. **Covariate-matched sampling.** Rather than treating every gene that
   survives the exclusion/mutation/pathway filters as an equally valid
   negative, each positive (driver) gene is matched to nearby negative
   candidate(s) in gene-length / expression (/ mutation-rate) space. Driver
   genes tend to be longer and more highly expressed than a random
   background (more exons = more room for mutations to land and be
   detected; higher expression -> better sequencing depth/coverage and
   more reliable variant calling), so an unmatched random negative set lets
   a classifier learn "is this gene long/expressed" as a shortcut for "is
   this a driver," rather than real driver biology. Matching removes that
   confound.
"""

from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1. Expanded driver-gene exclusion set (NCG + CGC + IntOGen + Bailey)
# ---------------------------------------------------------------------------

def load_ncg_genes(ncg_file: Path, gene_col: str = "symbol") -> Set[str]:
    """Load driver gene symbols from the Network of Cancer Genes (NCG)."""
    df = pd.read_csv(ncg_file, sep="\t")
    return set(df[gene_col].dropna())


def load_cgc_genes(cgc_file: Path, gene_col: str = "Symbol") -> Set[str]:
    """Load driver gene symbols from the COSMIC Cancer Gene Census."""
    df = pd.read_csv(cgc_file, sep="\t")
    return set(df[gene_col].dropna())


def load_intogen_genes(intogen_file: Path, gene_col: str = "SYMBOL") -> Set[str]:
    """Load driver gene symbols from IntOGen's compendium of driver genes."""
    df = pd.read_csv(intogen_file, sep="\t")
    return set(df[gene_col].dropna())


def load_bailey_genes(bailey_file: Path, gene_col: str = "Gene") -> Set[str]:
    """Load driver gene symbols from Bailey et al. 2018 ("Comprehensive
    Characterization of Cancer Driver Genes and Mutations", Cell).
    """
    df = pd.read_csv(bailey_file, sep="\t")
    return set(df[gene_col].dropna())


def build_driver_exclusion_set(
    ncg_file: Optional[Path] = None,
    cgc_file: Optional[Path] = None,
    intogen_file: Optional[Path] = None,
    bailey_file: Optional[Path] = None,
) -> Set[str]:
    """Union of driver genes across all provided reference databases.

    Any gene in the returned set must be excluded from the negative
    candidate pool. Pass only the files you have; sources you omit are
    simply skipped (with a printed note), so this degrades gracefully if
    e.g. an IntOGen or Bailey export isn't available yet.
    """
    loaders = {
        "NCG": (ncg_file, load_ncg_genes),
        "CGC": (cgc_file, load_cgc_genes),
        "IntOGen": (intogen_file, load_intogen_genes),
        "Bailey et al.": (bailey_file, load_bailey_genes),
    }

    excluded: Set[str] = set()

    for name, (file_path, loader) in loaders.items():
        if file_path is None:
            print(f"[{name}] skipped (no file provided)")
            continue
        genes = loader(file_path)
        print(f"[{name}] {len(genes)} genes")
        excluded |= genes

    print(f"[DRIVER EXCLUSION SET] {len(excluded)} unique genes across all sources")
    return excluded


# ---------------------------------------------------------------------------
# 2. Gene covariates: length and expression
# ---------------------------------------------------------------------------

def compute_gene_lengths(gtf_file: Path) -> pd.Series:
    """Compute gene length (bp, end - start + 1) per gene symbol from a
    GTF annotation. Returns a Series indexed by gene name.
    """
    records = []

    with open(gtf_file) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            start, end = int(fields[3]), int(fields[4])
            info = fields[8]

            gene_name = None
            for item in info.split(";"):
                item = item.strip()
                if item.startswith("gene_name"):
                    gene_name = item.split('"')[1]
                    break

            if gene_name:
                records.append((gene_name, end - start + 1))

    length_df = pd.DataFrame(records, columns=["gene", "length"])
    # a gene can appear on more than one GTF line (e.g. PAR genes on X/Y);
    # keep the longest recorded span
    return length_df.groupby("gene")["length"].max()


def load_gene_expression(
    expression_file: Path,
    gene_col: str = "gene",
    value_col: str = "median_expression",
) -> pd.Series:
    """Load a precomputed median expression value per gene (e.g. median
    TPM across GTEx or TCGA normal samples). Expects a CSV with at least
    `gene_col` and `value_col`.
    """
    df = pd.read_csv(expression_file)
    return df.set_index(gene_col)[value_col]


def load_gene_mutation_rate(
    mutation_frequency_file: Path,
    gene_col: str = "gene",
    value_col: str = "mutation_frequency",
) -> pd.Series:
    """Per-gene mutation rate for matching, computed as the maximum
    mutation frequency observed across cancer types (consistent with how
    the mutation-frequency exclusion filter treats "frequently mutated").
    Genes absent from the table are left out of the returned Series (and
    will simply be dropped from matching, same as any other missing
    covariate) rather than being silently imputed as zero — for matching
    purposes we only want genes with directly observed mutation rates.
    """
    df = pd.read_csv(mutation_frequency_file)
    return df.groupby(gene_col)[value_col].max()


# ---------------------------------------------------------------------------
# 3. Length / expression / mutation-rate matched negative sampling
# ---------------------------------------------------------------------------

def build_covariate_table(
    genes: Iterable[str],
    gene_lengths: pd.Series,
    gene_expression: pd.Series,
    mutation_rate: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Assemble a per-gene covariate table (length, expression, optional
    mutation rate) for a set of genes. Genes missing any requested
    covariate are dropped, since they can't be placed in the matching
    space.
    """
    genes = list(genes)
    df = pd.DataFrame({"gene": genes})
    df["length"] = df["gene"].map(gene_lengths)
    df["expression"] = df["gene"].map(gene_expression)

    if mutation_rate is not None:
        df["mutation_rate"] = df["gene"].map(mutation_rate)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[COVARIATES] dropped {dropped}/{before} genes missing a covariate value")

    return df


def match_negatives(
    positive_genes: Iterable[str],
    candidate_negative_genes: Iterable[str],
    gene_lengths: pd.Series,
    gene_expression: pd.Series,
    mutation_rate: Optional[pd.Series] = None,
    n_per_positive: int = 1,
    random_seed: int = 42,
) -> pd.DataFrame:
    """For each positive (driver) gene, find its `n_per_positive` nearest
    unused negative-candidate gene(s) in covariate space.

    Matching is performed with a KD-tree over log-transformed, z-scored
    covariates (log so that length/expression, which span several orders
    of magnitude, don't dominate the distance purely from scale; z-scored
    so length, expression, and mutation rate contribute comparably).
    Positives are matched in a randomized order so that no gene
    systematically wins ties for a popular negative just because of input
    ordering, and each negative is used at most once (sampling without
    replacement).

    Args:
        positive_genes: driver gene symbols to match against.
        candidate_negative_genes: genes eligible to be sampled as
            negatives (i.e. already filtered to exclude NCG/CGC/IntOGen/
            Bailey driver genes, and any other exclusion criteria).
        gene_lengths: Series of gene length in bp, indexed by gene symbol.
        gene_expression: Series of expression value, indexed by gene
            symbol.
        mutation_rate: optional Series of mutation rate, indexed by gene
            symbol. If provided, matching is done on length + expression +
            mutation rate; if omitted, matching is length + expression
            only.
        n_per_positive: how many matched negatives to find per positive
            gene (>1 gives a larger, still-matched negative set).
        random_seed: seed for match-order randomization.

    Returns:
        DataFrame with columns: positive_gene, matched_negative_gene,
        match_distance.
    """
    rng = np.random.default_rng(random_seed)

    positive_genes = list(positive_genes)
    candidate_negative_genes = list(candidate_negative_genes)

    pos_df = build_covariate_table(
        positive_genes, gene_lengths, gene_expression, mutation_rate
    )
    neg_df = build_covariate_table(
        candidate_negative_genes, gene_lengths, gene_expression, mutation_rate
    )

    if len(pos_df) == 0:
        raise ValueError("No positive genes have complete covariate data")
    if len(neg_df) == 0:
        raise ValueError("No candidate negative genes have complete covariate data")

    covariate_cols = ["length", "expression"] + (
        ["mutation_rate"] if mutation_rate is not None else []
    )

    def _log_transform(df: pd.DataFrame) -> np.ndarray:
        X = df[covariate_cols].to_numpy(dtype=float).copy()
        return np.log1p(X)

    pos_X = _log_transform(pos_df)
    neg_X = _log_transform(neg_df)

    # z-score using the negative pool's mean/std so the matching space is
    # anchored to the population being sampled from
    mean, std = neg_X.mean(axis=0), neg_X.std(axis=0)
    std[std == 0] = 1.0  # guard against a zero-variance covariate
    pos_Xz = (pos_X - mean) / std
    neg_Xz = (neg_X - mean) / std

    tree = cKDTree(neg_Xz)

    used_neg_idx: Set[int] = set()
    matches = []

    order = rng.permutation(len(pos_df))

    for i in order:
        query_point = pos_Xz[i]
        needed = n_per_positive
        k = min(len(neg_df), max(needed * 20, 50))
        dists, idxs = tree.query(query_point, k=k)
        dists = np.atleast_1d(dists)
        idxs = np.atleast_1d(idxs)

        chosen = 0
        for dist, idx in zip(dists, idxs):
            if idx in used_neg_idx:
                continue
            used_neg_idx.add(idx)
            matches.append(
                {
                    "positive_gene": pos_df.loc[i, "gene"],
                    "matched_negative_gene": neg_df.loc[idx, "gene"],
                    "match_distance": float(dist),
                }
            )
            chosen += 1
            if chosen >= needed:
                break

        if chosen < needed:
            print(
                f"[MATCHING] warning: only found {chosen}/{needed} unused "
                f"matches for '{pos_df.loc[i, 'gene']}' — negative candidate "
                "pool may be too small or too dissimilar in this region of "
                "covariate space"
            )

    match_df = pd.DataFrame(matches)
    print(
        f"[MATCHING] matched {match_df['positive_gene'].nunique()}/{len(pos_df)} "
        f"positive genes to {match_df['matched_negative_gene'].nunique()} "
        "unique negative genes"
    )
    return match_df


# ---------------------------------------------------------------------------
# 4. Match-quality diagnostics
# ---------------------------------------------------------------------------

def summarize_matching_quality(
    pos_df: pd.DataFrame,
    matched_negative_genes: Iterable[str],
    unmatched_candidate_genes: Iterable[str],
    gene_lengths: pd.Series,
    gene_expression: pd.Series,
    mutation_rate: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compare covariate distributions (median length/expression/mutation
    rate) across three groups: positives, the matched negatives, and the
    full unmatched candidate pool. If matching worked, the "matched
    negatives" row should sit much closer to "positives" than the
    "unmatched candidates" row does — that's the check to report in the
    methods section as evidence the matching is doing its job.
    """
    covariate_cols = ["length", "expression"] + (
        ["mutation_rate"] if mutation_rate is not None else []
    )

    groups = {
        "positives": pos_df["gene"] if "gene" in pos_df.columns else pos_df,
        "matched_negatives": list(matched_negative_genes),
        "unmatched_candidates": list(unmatched_candidate_genes),
    }

    rows = []
    for group_name, genes in groups.items():
        cov_df = build_covariate_table(genes, gene_lengths, gene_expression, mutation_rate)
        row = {"group": group_name, "n_genes": len(cov_df)}
        for col in covariate_cols:
            row[f"median_{col}"] = cov_df[col].median()
        rows.append(row)

    return pd.DataFrame(rows)
