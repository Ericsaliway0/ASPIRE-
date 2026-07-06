import os  
import subprocess

from tqdm.notebook import tqdm
import re
import shutil
import numpy as np
import pandas as pd
import igraph as ig
from scipy.sparse import lil_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel, pipeline

# Make required directories if they don't already exist
required_directories = ["data", "data/reactome", "data/gene"]
for directory in required_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)


#########################################################
## Download datasources and preprepare processing

# GENE NAMES
# Download gene names from https://www.genenames.org/download/custom/
gene_url = "https://www.genenames.org/cgi-bin/download/custom?col=gd_app_sym&col=gd_app_name&col=gd_pub_acc_ids&col=gd_pub_refseq_ids&col=gd_pub_eg_id&col=md_eg_id&col=md_prot_id&col=md_mim_id&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort&format=text&submit=submit"
gene_output_path = "data/gene/gene_names.csv"
subprocess.run(["curl", gene_url, "-o", gene_output_path])

# Database: Reactome
# Download Reactome files
reactome_files = [
    ("ReactomePathways.txt", "data/reactome/ReactomePathways.txt"),
    ("ReactomePathwaysRelation.txt", "data/reactome/ReactomePathwaysRelation.txt"),
    ("NCBI2Reactome.txt", "data/reactome/NCBI2Reactome.txt")
]

for file_name, file_path in reactome_files:
    subprocess.run(["curl", f"https://reactome.org/download/current/{file_name}", "-o", file_path])

# Run reactome_csv.py script
subprocess.run(["python3", "scripts/reactome/reactome_csv.py"])

df_ncbi2reactome = pd.read_csv('data/reactome/NCBI2Reactome.txt', sep='\t', names=['ncbi_id', 'reactome_id', 'url', 'reactome_name', 'evidence_code', 'species'], dtype={"ncbi_id": "string"})
# Filter rows where species is 'Homo sapiens'
df_ncbi2reactome = df_ncbi2reactome.query('species=="Homo sapiens"')
# Drop columns 'url', 'evidence_code', 'species'
df_ncbi2reactome = df_ncbi2reactome.drop(['url', 'evidence_code', 'species'], axis=1)
# Reset index and drop the original index column, then drop duplicate rows
df_ncbi2reactome = df_ncbi2reactome.reset_index().drop('index', axis=1).drop_duplicates()
df_ncbi2reactome.to_csv('data/reactome/reactome_ncbi.csv', index=False)

df_terms = pd.read_csv('data/reactome/ReactomePathways.txt', sep='\t', names=['reactome_id', 'reactome_name', 'species'])
# Filter rows where species is 'Homo sapiens'
df_terms = df_terms.query('species=="Homo sapiens"')
# Reset index and drop the original index column
df_terms = df_terms.reset_index().drop('index', axis=1)
df_terms.to_csv('data/reactome/reactome_terms.csv', index=False)

# Get list of valid reactome_id values
valid_terms = df_terms.get('reactome_id').values
df_rels = pd.read_csv('data/reactome/ReactomePathwaysRelation.txt', sep='\t', names=['reactome_id_1', 'reactome_id_2'])
# Filter rows where reactome_id_1 and reactome_id_2 are in valid_terms
df_rels = df_rels.query('reactome_id_1 in @valid_terms and reactome_id_2 in @valid_terms')
df_rels = df_rels.reset_index().drop('index', axis=1)
df_rels.to_csv('data/reactome/reactome_relations.csv', index=False)



#########################################################
## build knowledge graph from reactome pathway

data_path = 'data/' # updated path
save_path = data_path+'kg/'

# note, run pip install urllib3==1.26.6 to avoid an OpenSSL dependency in the current version of urllib3

def assert_dtypes(df): 
    all_string = True
    for i, x in enumerate(df.dtypes.values): 
        if x != np.dtype('O'): 
            all_string = False
            print(df.columns[i], x)
    if not all_string: assert False
    
df_reactome_terms = pd.read_csv(data_path+'reactome/reactome_terms.csv', low_memory=False)
assert_dtypes(df_reactome_terms)

df_reactome_rels = pd.read_csv(data_path+'reactome/reactome_relations.csv', low_memory=False)
assert_dtypes(df_reactome_rels)

df_reactome_ncbi = pd.read_csv(data_path+'reactome/reactome_ncbi.csv', low_memory=False)
df_reactome_ncbi = df_reactome_ncbi[df_reactome_ncbi.ncbi_id.str.isnumeric()]
assert_dtypes(df_reactome_ncbi)

df_prot_names = pd.read_csv(data_path+'gene/gene_names.csv', low_memory=False, sep='\t')
df_prot_names = df_prot_names.rename(columns={'NCBI Gene ID(supplied by NCBI)':'ncbi_id', 'NCBI Gene ID':'ncbi_id2', 'Approved symbol':'symbol', 'Approved name':'name'})
df_prot_names = df_prot_names.get(['ncbi_id', 'symbol']).dropna()
df_prot_names = df_prot_names.astype({'ncbi_id':int}).astype({'ncbi_id':str})
assert_dtypes(df_prot_names)


def clean_edges(df): 
    df = df.get(['relation', 'display_relation', 'x_id','x_type', 'x_name', 'x_source','y_id','y_type', 'y_name', 'y_source'])
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.query('not ((x_id == y_id) and (x_type == y_type) and (x_source == y_source) and (x_name == y_name))')
    return df

df_path_path = pd.merge(df_reactome_rels, df_reactome_terms, 'inner', left_on='reactome_id_1', right_on='reactome_id')
df_path_path = df_path_path.rename(columns={'reactome_id': 'x_id', 'reactome_name':'x_name'})
df_path_path = pd.merge(df_path_path, df_reactome_terms, 'inner', left_on='reactome_id_2', right_on='reactome_id')
df_path_path = df_path_path.rename(columns={'reactome_id': 'y_id', 'reactome_name':'y_name'})

df_path_path['x_source'] = 'REACTOME'
df_path_path['x_type'] = 'pathway'
df_path_path['y_source'] = 'REACTOME'
df_path_path['y_type'] = 'pathway'
df_path_path['relation'] = 'pathway_pathway'
df_path_path['display_relation'] = 'parent-child'
df_path_path = clean_edges(df_path_path)
df_path_path = df_path_path[['x_id', 'x_name', 'y_id', 'y_name']]  # Keep only desired columns
df_path_path.head(1)

# Save the dataframe to a CSV file
df_path_path.to_csv(save_path+'reactome_pathway.csv', index=False)
