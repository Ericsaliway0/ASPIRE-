import os
import pickle
import pandas as pd
import torch
import marker
import network
import dataset
import model
import train
from sklearn.model_selection import train_test_split

def get_stid_mapping(graph):
    stid_mapping = {}  # Mapping of node_id to stId
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        stid_mapping[node_id] = stId  # Store the mapping
    return stid_mapping  # Return the stId mapping

def save_graph_to_neo4j(graph):
    from py2neo import Graph, Node, Relationship

    neo4j_url = "neo4j+s://7ffb183d.databases.neo4j.io"
    user = "neo4j"
    password = "BGc2jKUI44h_awhU5gEp8NScyuyx-iSSkTbFHEHJRpY"
    
    neo4j_graph = Graph(neo4j_url, auth=(user, password))

    # Clear the existing graph
    neo4j_graph.delete_all()

    # Create nodes
    nodes = {}
    for node_id, data in graph.graph_nx.nodes(data=True):
        stId = data['stId']
        node = Node("Pathway", stId=stId, name=data['name'], weight=data['weight'], significance=data['significance'])
        nodes[node_id] = node
        neo4j_graph.create(node)

    # Create relationships
    for source, target in graph.graph_nx.edges():
        relationship = Relationship(nodes[source], "parent-child", nodes[target])
        neo4j_graph.create(relationship)


def create_network_from_markers(marker_list, p_value, kge):
    enrichment_analysis = marker.Marker(marker_list, p_value)
    graph = network.Network(enrichment_analysis.result, kge)
    return graph

import pandas as pd

def save_network_to_csv(graph, output_file="network_edges.csv"):
    """
    Save the Network object's graph to a CSV file.
    Columns: GeneA, GeneB, stIdA, stIdB, nameA, nameB, typeA, typeB
    """
    edges_data = []
    for u, v in graph.graph_nx.edges():
        node_u = graph.graph_nx.nodes[u]
        node_v = graph.graph_nx.nodes[v]
        edges_data.append({
            "GeneA": u,
            "GeneB": v,
            "stIdA": node_u.get("stId"),
            "stIdB": node_v.get("stId"),
            "nameA": node_u.get("name"),
            "nameB": node_v.get("name"),
            "typeA": node_u.get("type"),
            "typeB": node_v.get("type"),
            "weightA": node_u.get("weight", None),
            "weightB": node_v.get("weight", None),
            "significanceA": node_u.get("significance", None),
            "significanceB": node_v.get("significance", None),
        })
    
    df_edges = pd.DataFrame(edges_data)
    df_edges.to_csv(output_file, sep=",", index=False)
    print(f"✅ Saved network to {output_file}")

def save_to_disk(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    save_path = os.path.join(save_dir, graph.kge + '.pkl')
    pickle.dump(graph.graph_nx, open(save_path, 'wb'))

def save_stid_to_csv(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    assert os.path.isdir(save_dir), 'Directory does not exist!'
    stid_data = {'stId': [node['stId'] for node in graph.graph_nx.nodes.values()]}
    df = pd.DataFrame(stid_data)
    csv_path = os.path.join(save_dir, 'stId_nodes.csv')
    df.to_csv(csv_path, index=False)


def create_embedding_with_markers_(p_value=0.05, save=True, data_dir='reactome_embedding/data/emb'):
    # Read symbols from the CSV file
    csv_path = 'reactome_embedding/data/genes_pathways.csv'
    data = pd.read_csv(csv_path)
    symbols = data['symbol'].tolist()
    
    # Split the symbols into train and test sets
    emb_train, emb_test = train_test_split(symbols, test_size=0.3, random_state=42)
    ##print('emb_train=========================\n', emb_train)

    # Create networks for train and test sets
    graph_train = create_network_from_markers(emb_train, p_value, 'emb_train')
    graph_test = create_network_from_markers(emb_test, p_value, 'emb_test')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_embedding_with_markers_ori(p_value=0.05, save=True, data_dir='embedding/data/emb'):
    emb_train = ['MS4A1', 'CD8A', 'CD4', 'KRT19', 'PCNA', 'CD68', 'PDCD1', 'PTRPC', 'KRT8', 'HER2', 'FOXP3', 'KRT5', 'H3F3A', 'H3F3B', 'RPS6', 'ESR1', 'CD44', 'KRT17', 'PDPN', 'PECAM1', 'GZMB', 'VIM', 'pAb', 'RB1', 'CD3', 'ACTA2', 'PARP1', 'H2AFX', 'CDH1', 'KRT7', 'KRT14', 'COL4A1', 'LMNA', 'H3K27', 'CD274', 'MKI67', 'PGR', 'LMNB1', 'H3K4', 'LMNB2', 'COL1A1', 'CD34', 'AR', 'HIF1A', 'FOXP3']
    emb_test = ['AKT1', 'BMP2', 'BMP4', 'MAPK1', 'MAPK3', 'BRD4', 'CASP3', 'NCAM1', 'MTOR']
    
    '''
    markers = 'MS4A1, CD8A, CD4, KRT19, PCNA, CD68, PDCD1, PTRPC, KRT8, HER2, FOXP3, KRT5, H3F3A, H3F3B, RPS6, ESR1, CD44, KRT17, PDPN, PECAM1, GZMB, VIM, pAb, RB1, CD3, ACTA2, PARP1, H2AFX, CDH1, KRT7, KRT14, COL4A1, LMNA, H3K27, CD274, MKI67, PGR, LMNB1, H3K4, LMNB2, COL1A1, CD34, AR, HIF1A, FOXP3, AKT1, BMP2, BMP4, MAPK1, MAPK3, BRD4, CASP3, NCAM1, MTOR'
    result = analysis.identifiers(ids=markers)
    token = result['summary']['token']
    token
    '''
    
    graph_train = create_network_from_markers(emb_train, p_value, 'emb_train')
    graph_test = create_network_from_markers(emb_test, p_value, 'emb_test')

    if save:
        save_dir = os.path.join(data_dir, 'raw')
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_embedding_with_markers_orii(p_value=0.05, save=True, data_dir='reactome_embedding/data/emb'):
    # Read symbols from the CSV file
    csv_path = 'reactome_embedding/data/genes_pathways.csv'
    data = pd.read_csv(csv_path)
    symbols = data['symbol'].tolist()

    # Read pathway-to-first-gene mapping
    # mapping_path = "data/pathways_mapped_all_gene.tsv"
    mapping_path = "data/processed/pathways_mapped_all_genes.tsv"
    mapping_df = pd.read_csv(mapping_path, sep="\t", low_memory=False)
    pathway_to_gene = dict(zip(mapping_df["PathwayID"], mapping_df["Gene1"]))

    # Map pathway IDs to their representative genes if possible
    symbols_mapped = [pathway_to_gene.get(sym, sym) for sym in symbols]

    # Split the mapped symbols into train and test sets
    emb_train, emb_test = train_test_split(symbols_mapped, test_size=0.3, random_state=42)

    # Create networks for train and test sets
    graph_test = create_network_from_markers(emb_test, p_value, "emb_test")
    graph_train = create_network_from_markers(emb_train, p_value, "emb_train")

    if save:
        os.makedirs(data_dir, exist_ok=True)
        save_dir = os.path.join(data_dir, "raw")
        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

    return graph_train, graph_test

def create_embedding_with_markers(
    p_value=0.05,
    save=True,
    data_dir='reactome_embedding/data/emb'
):
    # ==========================================================
    # 1. Load embedding gene list (GROUND TRUTH)
    # ==========================================================
    emb_path = "data/processed/combined_omics_embeddings_64x7x3.csv"
    emb_df = pd.read_csv(emb_path)

    embedding_genes = set(emb_df["stId"].str.upper())

    print(f"✅ Embedding genes loaded: {len(embedding_genes)}")

    # ==========================================================
    # 2. Load marker symbols
    # ==========================================================
    csv_path = 'reactome_embedding/data/genes_pathways.csv'
    data = pd.read_csv(csv_path)

    symbols = data['symbol'].astype(str).str.upper().tolist()

    print(f"✅ Raw symbols: {len(symbols)}")

    # ==========================================================
    # 3. Filter symbols to embedding gene universe
    # ==========================================================
    symbols_filtered = [
        s for s in symbols if s in embedding_genes
    ]

    print(f"✅ Symbols after filtering (in embeddings): {len(symbols_filtered)}")

    # ==========================================================
    # 4. Train/test split
    # ==========================================================
    emb_train, emb_test = train_test_split(
        symbols_filtered,
        test_size=0.3,
        random_state=42
    )

    print(f"✅ Train genes: {len(emb_train)}")
    print(f"✅ Test genes: {len(emb_test)}")

    # ==========================================================
    # 5. Build networks
    # ==========================================================
    graph_test = create_network_from_markers(
        emb_test,
        p_value,
        "emb_test"
    )

    graph_train = create_network_from_markers(
        emb_train,
        p_value,
        "emb_train"
    )

    # ==========================================================
    # 6. Save
    # ==========================================================
    if save:
        os.makedirs(data_dir, exist_ok=True)
        save_dir = os.path.join(data_dir, "raw")

        save_to_disk(graph_train, save_dir)
        save_to_disk(graph_test, save_dir)

        print(f"💾 Saved graphs → {save_dir}")

    return graph_train, graph_test

def create_embeddings(load_model=True, save=True, data_dir='reactome_embedding/data/emb', hyperparams=None, plot=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset.PathwayDataset(data_dir)
    emb_dir = os.path.abspath(os.path.join(data_dir, 'embeddings'))
    if not os.path.isdir(emb_dir):
        os.mkdir(emb_dir)

    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']

    net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads).to(device)

    if load_model:
        model_path = os.path.abspath(os.path.join(data_dir, 'models/model.pth'))
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot)
        net.load_state_dict(torch.load(model_path))

    embedding_dict = {}
    
    for idx in range(len(data)):
        graph, name = data[idx]
        graph = graph.to(device)  # Move graph to the same device as net
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding
        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)

    return embedding_dict

