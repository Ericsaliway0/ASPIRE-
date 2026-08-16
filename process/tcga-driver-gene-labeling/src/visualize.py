"""Plotting helpers for exploring mutation frequency results."""

import matplotlib.pyplot as plt
import pandas as pd


def plot_mutation_frequency_distribution(df: pd.DataFrame, ax=None):
    """Log-scale histogram of mutation frequency across all genes/cancers."""
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(df["mutation_frequency"] + 1e-6, bins=50)
    ax.set_yscale("log")
    ax.set_xlabel("Mutation Frequency")
    ax.set_ylabel("Log Count")
    ax.set_title("Log-scale Mutation Frequency Distribution")
    return ax


def plot_mutation_frequency_by_cancer(df: pd.DataFrame, ax=None):
    """Overlaid per-cancer-type mutation frequency histograms."""
    if "cancer_type" not in df.columns:
        raise ValueError("DataFrame has no 'cancer_type' column")

    if ax is None:
        _, ax = plt.subplots()

    for cancer in df["cancer_type"].unique():
        subset = df[df["cancer_type"] == cancer]["mutation_frequency"]
        ax.hist(subset + 1e-6, bins=50, alpha=0.5, label=cancer)

    ax.set_yscale("log")
    ax.set_xlabel("Mutation Frequency")
    ax.set_ylabel("Log Count")
    ax.set_title("Multi-Cancer Mutation Distribution")
    ax.legend()
    return ax


def plot_top_genes_by_cancer(df: pd.DataFrame, cancer: str, top_n: int = 10, ax=None):
    """Horizontal bar chart of the top-N most frequently mutated genes for a
    single cancer type.
    """
    if "cancer_type" not in df.columns:
        raise ValueError("DataFrame has no 'cancer_type' column")

    if ax is None:
        _, ax = plt.subplots()

    sub = df[df["cancer_type"] == cancer]
    top = sub.sort_values("mutation_frequency", ascending=False).head(top_n)

    ax.barh(top["gene"], top["mutation_frequency"])
    ax.invert_yaxis()
    ax.set_title(f"Top Genes - {cancer}")
    ax.set_xlabel("Mutation Frequency")
    return ax
