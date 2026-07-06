################################
## Prepare the data processing for pathway information to construct the knowledge graph.

import os  
import numpy as np  
import pandas as pd  


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
