"""Build the protein-coding gene universe from a GENCODE GTF annotation.

Only entries with `gene_type == "protein_coding"` are kept. Earlier versions
of this pipeline kept every GTF entry of feature type "gene" regardless of
biotype, which pulled in lncRNAs, pseudogenes, miRNAs, and other non-coding
genes and inflated the gene universe from ~20,000 to ~77,000 genes.
"""

from pathlib import Path
from typing import Optional, Set


def build_protein_coding_gene_universe(gtf_file: Path) -> Set[str]:
    """Parse a GENCODE GTF file and return the set of protein-coding gene
    symbols (`gene_name`).
    """
    genes: Set[str] = set()

    with open(gtf_file, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")

            if len(fields) < 9 or fields[2] != "gene":
                continue

            info = fields[8]

            gene_name: Optional[str] = None
            gene_type: Optional[str] = None

            for item in info.split(";"):
                item = item.strip()
                if item.startswith("gene_name"):
                    gene_name = item.split('"')[1]
                elif item.startswith("gene_type"):
                    gene_type = item.split('"')[1]

            if gene_name and gene_type == "protein_coding":
                genes.add(gene_name)

    return genes
