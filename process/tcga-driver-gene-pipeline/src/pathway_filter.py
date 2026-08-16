"""Build the set of genes belonging to disease-related Reactome pathways.

Restricted to pathways that are descendants of the top-level "Disease" node
in the Reactome pathway hierarchy, rather than every pathway in the full
catalog (which would include unrelated biology such as metabolism,
translation, or vesicle transport).
"""

from pathlib import Path
from typing import Set

import networkx as nx
import pandas as pd

from config import REACTOME_DISEASE_ROOT


def build_pathway_hierarchy(relations_file: Path) -> nx.DiGraph:
    """Build a directed graph of Reactome pathway parent -> child relations."""
    relations_df = pd.read_csv(relations_file)

    graph = nx.DiGraph()
    graph.add_edges_from(
        zip(relations_df["reactome_id_1"], relations_df["reactome_id_2"])
    )
    return graph


def get_disease_pathway_ids(
    graph: nx.DiGraph, root: str = REACTOME_DISEASE_ROOT
) -> Set[str]:
    """Return the root pathway ID plus every pathway ID descended from it."""
    if root not in graph:
        raise ValueError(f"{root} not found in the pathway relations graph")

    descendant_ids = nx.descendants(graph, root)
    descendant_ids.add(root)
    return descendant_ids


def build_disease_pathway_genes(
    gmt_file: Path, relations_file: Path, root: str = REACTOME_DISEASE_ROOT
) -> Set[str]:
    """Return the set of genes belonging to any Reactome pathway that is a
    descendant of `root` (default: the top-level "Disease" pathway).
    """
    graph = build_pathway_hierarchy(relations_file)
    disease_pathway_ids = get_disease_pathway_ids(graph, root)

    genes: Set[str] = set()
    kept_pathways = 0
    skipped_pathways = 0

    with open(gmt_file) as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 3:
                continue

            pathway_id = parts[1]
            pathway_genes = parts[2:]

            if pathway_id in disease_pathway_ids:
                genes.update(pathway_genes)
                kept_pathways += 1
            else:
                skipped_pathways += 1

    print(
        f"[PATHWAY FILTER] kept {kept_pathways} disease-related pathways, "
        f"skipped {skipped_pathways} unrelated pathways"
    )

    return genes
