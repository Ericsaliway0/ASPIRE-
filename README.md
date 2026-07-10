## ASPIRE: an interpretable graph neural network framework for biological representation learning

This repository contains the code for our research, "ASPIRE: an interpretable graph neural network framework for biological representation learning".

![Alt text](images/framework_aspire.png)


## Data Source

The dataset is obtained from the following sources:

- **[Reactome pathway database](https://reactome.org/)**

REACTOME is an open-source, open access, manually curated and peer-reviewed pathway database.

The different dataset and KG used in this project are located in the `data` directory. These files include:

-) The data about pathways from https://reactome.org/download/current/ReactomePathways.txt, relationships between pathways from https://reactome.org/download/current/ReactomePathwaysRelation.txt and pathway-protein relations from https://reactome.org/download/current/NCBI2Reactome.txt on 25 October, 2025.

-) The built knowledge graph including pathway-pathway and gene-pathway relationships.

- **[Genomic Data Commons (GDC)](https://portal.gdc.cancer.gov/)**

TCGA multi-omics data (Mutation/MAF, Gene Expression, Copy Number Alteration, and miRNA Expression) were queried and downloaded programmatically from the GDC Files/Data API for the following cancer cohorts: `TCGA-BLCA`, `TCGA-BRCA`, `TCGA-COAD`, `TCGA-ESCA`, `TCGA-LUAD`, `TCGA-LUSC`, `TCGA-STAD`.

- **[GENCODE gene annotation](https://www.gencodegenes.org)**

Gene annotations were obtained from the latest available GENCODE human release (v49 at time of writing) and used to map Ensembl gene identifiers to gene symbols and genomic coordinates. Only protein-coding genes were retained for downstream analysis.

The produced gene-level, per-cancer, min-max normalized multi-omics feature matrices were used as node features for the graph neural network. 

Processing steps include:

- Binary/frequency-based mutation scoring from MAF files (silent mutations excluded)
- Log2-transformed, gene-aggregated RNA-Seq expression scoring
- Overlap-weighted, signed gene-level CNA scoring from genomic segments
- miRNA expression aggregation mapped to gene-level regulatory scores via miRTarBase interactions
- Per-modality min-max normalization and removal of non-informative (all-zero) genes
- A GDC metadata snapshot/manifest for reproducibility of the selected input files

Outputs of this pipeline (gene x cancer feature matrices per omics modality) are written to `data/tcga_omics/` and are combined with the Reactome-derived knowledge graph for model training.

## Setup and Get Started

1. Install the required dependencies:
   - `pip install -r requirements.txt`

2. Activate your Conda environment:
   - `conda activate gnn`

3. Install PyTorch:
   - `conda install pytorch torchvision torchaudio -c pytorch`

4. Install the necessary Python packages:
   - `pip install pandas numpy`
   - `pip install py2neo pandas matplotlib scikit-learn`
   - `pip install tqdm`
   - `pip install seaborn`
   - `pip install requests`
   - `pip install umap-learn`
   - `pip install gseapy`
   - `pip install gprofiler-official`
   - `pip install plotly`
   - `pip install scipy`

5. Install DGL:
   - `conda install -c dglteam dgl`

6. Download the data from the built knowledge graph using the link below and place it in the `data` directory before training:
   - [Download KG](https://drive.google.com/file/d/1RvGw3T_gWvTIRekYuFofy4bjgc0bS53N/view?usp=drive_link)

7. (Optional) Rebuild the TCGA multi-omics gene feature matrices from raw GDC/GENCODE sources by running the `process/tcga_multi_omics_feature_matrix.ipynb`. Pre-built feature matrices are already provided in `data/tcga_omics/` if you'd rather skip this step.

8. To train the model, run the following command:
   - `python prediction/main.py --num-layers 6 --lr 0.001 --input-size 2 --hidden-size 16 --epochs 100`
