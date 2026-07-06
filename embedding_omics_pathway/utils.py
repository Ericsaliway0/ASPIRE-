import os
import pickle
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import dataset
import model
import train
from network import Network  # Importing the updated Network class
import glob

# Read stId mapping from the graph
def get_stid_mapping(graph):
    stid_mapping = {node_id: data['stId'] for node_id, data in graph.graph_nx.nodes(data=True)}
    return stid_mapping

# Save graph to disk
def save_to_disk(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{graph.kge}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(graph.graph_nx, f)
    print(f"Graph saved to {save_path}")

# Save stId to CSV
def save_stid_to_csv(graph, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    stid_data = {'stId': [data['stId'] for _, data in graph.graph_nx.nodes(data=True)]}
    df = pd.DataFrame(stid_data)
    csv_path = os.path.join(save_dir, 'stId_nodes.csv')
    df.to_csv(csv_path, index=False)
    print(f"stId nodes saved to {csv_path}")

# Create network using protein interaction data
def create_network_from_genes(data, kge, weight_path):
    ##graph = Network('data/split_files/protein_interaction_p_value_results_with_fdr_ptmod.csv') 
    ##graph = Network('data/protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    graph = Network(weight_path)
    ########graph = Network('data/__protein_interaction_p_value_CPDB_ppi_0.99.csv')
    ##graph = Network('data/inhibition_protein_interaction_p_value_results_with_fdr_SHS27k.csv')
    # Initialize the protein network
    graph.interaction_data = data  # Assign the filtered data directly
    graph.build_graph()  # Build the graph with the interaction data
    graph.kge = kge  # Set the knowledge graph embedding identifier
    return graph


def create_embedding_with_genes(
    p_value=0.05,
    save=True,
    data_dir='data',
    omics_types=None,
    cancer_types=None
):
    if omics_types is None:
        omics_types = ['ge', 'mirna', 'mf']
    if cancer_types is None:
        cancer_types = ['BLCA', 'BRCA', 'COAD', 'ESCA', 'LUAD', 'LUSC', 'STAD']

    all_graphs = {}

    # 🔥 normalize base path
    base_dir = os.path.abspath(data_dir)

    print(f"📁 Using data_dir = {base_dir}")

    for omics in omics_types:
        for cancer in cancer_types:

            # ✅ FIXED PATH (no hardcoding project name)
            pattern = os.path.join(
                base_dir,
                # "omics",
                omics,
                f"{cancer}.csv"
            )

            # optional fallback if multiple files exist
            files = glob.glob(pattern)

            if len(files) == 0:
                print(f"⚠️ Skipping {omics}-{cancer}: File not found ({pattern})")
                continue

            p_value_path = files[0]

            df = pd.read_csv(p_value_path)

            if 'p_value' not in df.columns:
                print(f"⚠️ Skipping {omics}-{cancer}: Missing p_value column")
                continue

            filtered_df = df[df['p_value'] <= p_value]

            if len(filtered_df) < 5:
                genes_train = filtered_df
                genes_test = filtered_df
            else:
                genes_train, genes_test = train_test_split(
                    filtered_df, test_size=0.2, random_state=42
                )

            graph_train = create_network_from_genes(
                genes_train, 'emb_train', p_value_path
            )
            graph_test = create_network_from_genes(
                genes_test, 'emb_test', p_value_path
            )

            if save:
                save_dir = os.path.join(base_dir, omics, cancer, 'emb/raw')
                os.makedirs(save_dir, exist_ok=True)
                save_to_disk(graph_train, save_dir)
                save_to_disk(graph_test, save_dir)

            all_graphs[(omics, cancer)] = (graph_train, graph_test)

            print(f"✅ Processed {omics}-{cancer}")

    return all_graphs

# Function to create embeddings using GAT model
def create_embeddings(load_model=True, save=True, data_dir='data', hyperparams=None, plot=True, omics='cna', cancer='KIRC'):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data_dir_ = os.path.join(data_dir, omics, cancer)##, 'emb/raw')
    # Load dataset and set up directories
    data = dataset.Dataset(data_dir_)  # Adjust dataset to handle protein interactions
    emb_dir = os.path.abspath(os.path.join(data_dir_))##, 'embeddings'))
    os.makedirs(emb_dir, exist_ok=True)

    # Model parameters
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams.get('num_heads', 2)  # Default to 2 heads if not specified


    dim_latent = hyperparams['out_feats']
    '''num_layers = hyperparams['num_layers']
    
    net = model.SAGEModel(dim_latent=dim_latent, num_layers=num_layers).to(device)'''
    ## net = model.GATModel(in_feats=in_feats, out_feats=out_feats, num_layers=num_layers, num_heads=num_heads).to(device)
    ##net = model.GCNModel(dim_latent=dim_latent, num_layers=num_layers).to(device)
    net = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers).to('cpu')

    # Load or train the model
    if load_model:
        model_path = os.path.join(data_dir, 'models', 'model.pth')
        net.load_state_dict(torch.load(model_path))
    else:
        model_path = train.train(hyperparams=hyperparams, data_path=data_dir, plot=plot, omics=omics, cancer=cancer)
        net.load_state_dict(torch.load(model_path))

    # Generate and save embeddings
    embedding_dict = {}
    for idx in tqdm(range(len(data))):
        graph, name = data[idx]
        graph = graph.to('cpu')
        
        with torch.no_grad():
            embedding = net(graph)
        embedding_dict[name] = embedding.cpu()

        if save:
            emb_path = os.path.join(emb_dir, f'{name[:-4]}.pth')
            torch.save(embedding.cpu(), emb_path)
            print(f"Embedding for {name} saved to {emb_path}")

    return embedding_dict
