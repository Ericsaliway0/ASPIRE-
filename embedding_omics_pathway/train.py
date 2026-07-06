import copy
import json
import os
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import dataset
import model, utils, network
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from matplotlib.patches import Patch
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from py2neo import Graph, Node, Relationship
from neo4j import GraphDatabase
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm

COLORS = [
    '#0077B6','#0000FF','#00B4D8','#48EAC4','#F1C0E8','#B9FBC0',
    '#32CD32','#BEE1E6','#8A2BE2','#E377C2','#8EECF5','#A3C4F3',
    '#FFB347','#FFD700','#FF69B4','#CD5C5C','#7FFFD4','#FF7F50',
    '#C71585','#20B2AA','#6A5ACD','#40E0D0','#FF8C00','#DC143C',
    '#9ACD32','#1F77B4','#FF1493','#2E8B57','#D2691E','#9932CC',
    '#00CED1','#FF4500','#708090'
]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Ensure the input and target have the same shape
        if inputs.dim() > targets.dim():
            inputs = inputs.squeeze(dim=-1)
        elif targets.dim() > inputs.dim():
            targets = targets.squeeze(dim=-1)

        # Check if the shapes match after squeezing
        if inputs.size() != targets.size():
            raise ValueError(f"Target size ({targets.size()}) must be the same as input size ({inputs.size()})")

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def train_ori(hyperparams=None, data_path='data', plot=True, omics='mf', cancer='KIRC'):
    num_epochs = hyperparams['num_epochs']
    ##feat_drop = hyperparams['feat_drop']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']
    
    model_path = os.path.join(data_path, omics, cancer, 'emb/models')
    os.makedirs(model_path, exist_ok=True)
    model_path = os.path.join(model_path, f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth')
    
    '''omics_types = ['cna', 'ge', 'meth', 'mf']
    cancer_types = ['KIRC', 'BRCA', 'READ', 'PRAD', 'STAD', 'HNSC', 'LUAD', 
                    'THCA', 'BLCA', 'ESCA', 'LIHC', 'UCEC', 'COAD', 'LUSC', 'CESC', 'KIRP']

    for omics in omics_types:
        for cancer in cancer_types:    
            ##data_path = os.path.join(data_path, 'processed', omics, cancer)'''
            
    data_path_ = os.path.join(data_path, omics, cancer)##, 'emb/processed')
    ds = dataset.Dataset(data_path_)
    ds_train = [ds[0]]
    ds_valid = [ds[1]]
    dl_train = GraphDataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_valid = GraphDataLoader(ds_valid, batch_size=batch_size, shuffle=False)
    
    # Create the TAGCN model instance
    net = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers, do_train=True).to(device)

    # Set up the optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Save the best model
    best_model = model.TAGCNModel(dim_latent=out_feats, num_layers=num_layers, do_train=True)
    best_model.load_state_dict(copy.deepcopy(net.state_dict()))

    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    ##criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    best_train_loss, best_valid_loss = float('inf'), float('inf')
    best_f1_score = 0.0

    max_f1_scores_train = []
    max_f1_scores_valid = []
    
    results_path = 'embedding_omics_pw/results/embeddings/'
    results_path = os.path.join(results_path, omics, cancer)
    os.makedirs(results_path, exist_ok=True)

    
    all_embeddings_initial, cluster_labels_initial, colors = calculate_cluster_labels(
        net,
        dl_train,    
        eps=0.5,
        min_samples=15
    )

    cluster_plot_path_initial = os.path.join(
        results_path,
        f"clusters_dbscan_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png"
    )


    plot_clusters(
        all_embeddings_initial,
        cluster_labels_initial,
        colors,
        save_path=cluster_plot_path_initial,
        omics=omics,
        cancer=cancer,
        method="t-SNE + DBSCAN"
    )

    X_initial, labels_initial, colors = calculate_cluster_labels(net, dl_train)
    save_path_umap_initial = os.path.join(results_path, f'embeddings_umap_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')

    # visualize_embeddings_umap(X_initial, labels_initial, colors, save_path_umap_initial)
    visualize_embeddings_umap(
        X_initial,
        labels_initial,
        colors,
        save_path=save_path_umap_initial,
        omics=omics,
        cancer=cancer,
        method="UMAP + DBSCAN"
    )


    
    ##print('all_embeddings_initial---------------------------------\n', all_embeddings_initial)
    all_embeddings_initial = all_embeddings_initial.reshape(all_embeddings_initial.shape[0], -1)  # Flatten 
    save_path_heatmap_initial= os.path.join(results_path, f'embeddings_heatmap_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_matrix_initial= os.path.join(results_path, f'embeddings_matrix_stId_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_pca_initial = os.path.join(results_path, f'embeddings_pca_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
    save_path_t_SNE_initial = os.path.join(results_path, f'embeddings_t-SNE_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.png')
        
    for data in dl_train:
        graph, _ = data
        node_embeddings_initial= best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, omics, cancer, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels_initial) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index_initial = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster_initial= {}
        first_node_embedding_in_cluster_initial= {}

        stid_dic_initial= {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stId = nx_graph.nodes[node]['stId']
                stid_dic_initial[nx_graph.nodes[node]['stId']] = node_embeddings_initial[node_to_index_initial[node]]

        # Convert stid_dic_initial to a DataFrame
        stid_df_initial = pd.DataFrame.from_dict(stid_dic_initial, orient='index')

        # Save to CSV
        ##csv_save_path = 'gat/data/gene_embeddings_initial_sage.csv'
        csv_save_path_initial = os.path.join('embedding_omics_pw/data/', omics, cancer, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv')
        ##csv_save_path_initial = os.path.join('gat/data/', f'inhibition_gene_embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.csv')
        stid_df_initial.to_csv(csv_save_path_initial, index_label='gene')
                
        ##print('stid_dic_initial=======================\n',stid_dic_initial) 
        
        for node, cluster in zip(nx_graph.nodes, cluster_labels_initial):
            if 'stId' in nx_graph.nodes[node]:
                if cluster not in first_node_stId_in_cluster_initial:
                    first_node_stId_in_cluster_initial[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster_initial[cluster] = node_embeddings_initial[node_to_index_initial[node]]

        print('first_node_stId_in_cluster_initial-------------------------------\n', first_node_stId_in_cluster_initial)
        stid_list = list(first_node_stId_in_cluster_initial.values())
        embedding_list_initial = list(first_node_embedding_in_cluster_initial.values())
        create_heatmap_with_stid(embedding_list_initial, stid_list, save_path_heatmap_initial)
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list_initial, stid_list, save_path_matrix_initial)

        break

    # visualize_embeddings_tsne(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_t_SNE_initial)
    # visualize_embeddings_pca(all_embeddings_initial, cluster_labels_initial, stid_list, save_path_pca_initial)
    silhouette_avg_ = silhouette_score(all_embeddings_initial, cluster_labels_initial)
    davies_bouldin_ = davies_bouldin_score(all_embeddings_initial, cluster_labels_initial)
    summary_  = f"Silhouette Score: {silhouette_avg_}\n"
    summary_ += f"Davies-Bouldin Index: {davies_bouldin_}\n"

    save_file_= os.path.join(results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_initial.txt')
    with open(save_file_, 'w') as f:
        f.write(summary_)

    # Start training  
    with tqdm(total=num_epochs, desc="Training", unit="epoch", leave=False) as pbar:
        for epoch in range(num_epochs):
            loss_per_graph = []
            f1_per_graph = [] 
            net.train()
            for data in dl_train:
                graph, name = data
                name = name[0]
                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)
                weight_ = weight[labels.data.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss_weighted = loss * weight_
                loss_weighted = loss_weighted.mean()

                # Update parameters
                optimizer.zero_grad()
                loss_weighted.backward()
                optimizer.step()
                
                # Append output metrics
                loss_per_graph.append(loss_weighted.item())
                ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                preds = (logits.sigmoid() > 0.5).int()
                labels = labels.squeeze(1).int()
                f1 = metrics.f1_score(labels, preds)
                f1_per_graph.append(f1)

            running_loss = np.array(loss_per_graph).mean()
            running_f1_train = np.array(f1_per_graph).mean()
            loss_per_epoch_train.append(running_loss)
            f1_per_epoch_train.append(running_f1_train)

            # Validation iteration
            with torch.no_grad():
                loss_per_graph = []
                f1_per_graph = []
                net.eval()
                for data in dl_valid:
                    graph, name = data
                    name = name[0]
                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)
                    weight_ = weight[labels.data.view(-1).long()].view_as(labels)
                    loss = criterion(logits, labels)
                    loss_weighted = loss * weight_
                    loss_weighted = loss_weighted.mean()
                    loss_per_graph.append(loss_weighted.item())
                    ##preds = (logits.sigmoid() > 0.5).squeeze(1).int()
                    preds = (logits.sigmoid() > 0.5).int()
                    labels = labels.squeeze(1).int()
                    f1 = metrics.f1_score(labels, preds)
                    f1_per_graph.append(f1)

                running_loss = np.array(loss_per_graph).mean()
                running_f1_val = np.array(f1_per_graph).mean()
                loss_per_epoch_valid.append(running_loss)
                f1_per_epoch_valid.append(running_f1_val)
                
                max_f1_train = max(f1_per_epoch_train)
                max_f1_valid = max(f1_per_epoch_valid)
                max_f1_scores_train.append(max_f1_train)
                max_f1_scores_valid.append(max_f1_valid)

                if running_loss < best_valid_loss:
                    best_train_loss = running_loss
                    best_valid_loss = running_loss
                    best_f1_score = running_f1_val
                    best_model.load_state_dict(copy.deepcopy(net.state_dict()))
                    print(f"Best F1 Validation Score: {best_f1_score}")

            pbar.update(1)
            print(f"Epoch {epoch + 1} - F1 Train: {running_f1_train}, F1 Valid: {running_f1_val}")
            ## print(f"Epoch {epoch + 1} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}")

    
    
    all_embeddings, cluster_labels, colors = calculate_cluster_labels(
        best_model, 
        dl_train,    
        eps=0.5,
        min_samples=15
    )

    cluster_plot_path = os.path.join(
        results_path,
        f"clusters_dbscan_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png"
    )

    plot_clusters(
        all_embeddings,
        cluster_labels,
        colors,
        save_path=cluster_plot_path,
        omics=omics,
        cancer=cancer,
        method="t-SNE + DBSCAN"
    )



    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)  # Flatten 
    ##print('cluster_labels=========================\n', cluster_labels)

    cos_sim = np.dot(all_embeddings, all_embeddings.T)
    norms = np.linalg.norm(all_embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    if plot:
        loss_path = os.path.join(results_path, f'embeddings_loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        f1_path = os.path.join(results_path, f'embeddings_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        max_f1_path = os.path.join(results_path, f'embeddings_max_f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')
        matrix_path = os.path.join(results_path, f'embeddings_matrix_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png')

        draw_loss_plot(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        draw_max_f1_plot(max_f1_scores_train, max_f1_scores_valid, max_f1_path)
        draw_f1_plot(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    torch.save(best_model.state_dict(), model_path)

    save_path_pca = os.path.join(results_path, f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    save_path_t_SNE = os.path.join(results_path, f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    save_path_heatmap_= os.path.join(results_path, f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    save_path_matrix = os.path.join(results_path, f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')
    
    cluster_stId_dict = {}  # Dictionary to store clusters and corresponding stIds
    significant_stIds = []  # List to store significant stIds
    clusters_with_significant_stId = {}  # Dictionary to store clusters and corresponding significant stIds
    clusters_node_info = {}  # Dictionary to store node info for each cluster
    
    for data in dl_train:
        graph, _ = data
        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()
        graph_path = os.path.join(data_path, omics, cancer, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels) == len(nx_graph.nodes), "Cluster labels and number of nodes must match"
        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}
        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}

        stid_dic = {}

        # Populate stid_dic with node stIds mapped to embeddings
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stid = nx_graph.nodes[node]['stId']
                stid_dic[nx_graph.nodes[node]['stId']] = node_embeddings[node_to_index[node]]
                # Check if the node's significance is 'significant' and add its stId to the list
                if graph.ndata['significance'][node_to_index[node]].item() == 'significant':
                    significant_stIds.append(nx_graph.nodes[node]['stId'])

        # Convert stid_dic_initial to a DataFrame
        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')

        # Save to CSV
        ##csv_save_path = 'gat/data/gene_embeddings_final_sage.csv'
        csv_save_path_final = os.path.join('embedding_omics_pw/data/', omics, cancer, f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv')
        stid_df_final.to_csv(csv_save_path_final, index_label='stId')
                
        for node, cluster in zip(nx_graph.nodes, cluster_labels):
            if 'stId' in nx_graph.nodes[node]:
                stid = nx_graph.nodes[node]['stId']
                if cluster not in first_node_stId_in_cluster:
                    first_node_stId_in_cluster[cluster] = nx_graph.nodes[node]['stId']
                    first_node_embedding_in_cluster[cluster] = node_embeddings[node_to_index[node]]
                    
                # Populate cluster_stId_dict
                if cluster not in cluster_stId_dict:
                    cluster_stId_dict[cluster] = []
                cluster_stId_dict[cluster].append(nx_graph.nodes[node]['stId'])

                # Populate clusters_with_significant_stId
                if cluster not in clusters_with_significant_stId:
                    clusters_with_significant_stId[cluster] = []
                if nx_graph.nodes[node]['stId'] in significant_stIds:
                    clusters_with_significant_stId[cluster].append(nx_graph.nodes[node]['stId'])
                
                # Populate clusters_node_info with node information for each cluster
                if cluster not in clusters_node_info:
                    clusters_node_info[cluster] = []
                node_info = {
                    'stId': nx_graph.nodes[node]['stId'],
                    'significance': graph.ndata['significance'][node_to_index[node]].item(),
                    'other_info': nx_graph.nodes[node]  # Add other relevant info if necessary
                }
                clusters_node_info[cluster].append(node_info)
            
        print(first_node_stId_in_cluster)
        stid_list = list(first_node_stId_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())
        heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
        create_heatmap_with_stid(embedding_list, stid_list, save_path_heatmap_)
        # Call the function to plot cosine similarity matrix for cluster representatives with similarity values
        plot_cosine_similarity_matrix_for_clusters_with_values(embedding_list, stid_list, save_path_matrix)

        break

    # visualize_embeddings_umap(
    #     X_final,
    #     labels_final,
    #     colors_final,
    #     save_path_umap
    # )
    X, labels, colors = calculate_cluster_labels(best_model, dl_train)
    save_path_umap = os.path.join(results_path, f'embeddings_umap_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png')

    # visualize_embeddings_umap(X, labels, colors, save_path_umap)
    visualize_embeddings_umap(
        X,
        labels,
        colors,
        save_path=save_path_umap,
        omics=omics,
        cancer=cancer,
        method="UMAP + DBSCAN"
    )

    # visualize_embeddings_tsne(all_embeddings, cluster_labels, stid_list, save_path_t_SNE)
    # visualize_embeddings_pca(all_embeddings, cluster_labels, stid_list, save_path_pca)
    silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
    davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)

    print(f"Silhouette Score%%%%%%%%%%%%###########################: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    summary = f"Epoch {num_epochs} - Max F1 Train: {max_f1_train}, Max F1 Valid: {max_f1_valid}\n"
    summary += f"Best Train Loss: {best_train_loss}\n"
    summary += f"Best Validation Loss: {best_valid_loss}\n"
    summary += f"Best F1 Score: {max_f1_train}\n"
    summary += f"Silhouette Score: {silhouette_avg}\n"
    summary += f"Davies-Bouldin Index: {davies_bouldin}\n"

    save_file = os.path.join(results_path, f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt')
    with open(save_file, 'w') as f:
        f.write(summary)
    return model_path

def train_leak(hyperparams=None, data_path='data', plot=True, omics='mf', cancer='KIRC'):

    import os, copy, pickle
    import numpy as np
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn import metrics
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from dgl.dataloading import GraphDataLoader

    # ===============================
    # 1. Hyperparameters
    # ===============================
    num_epochs = hyperparams['num_epochs']
    in_feats = hyperparams['in_feats']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    batch_size = hyperparams['batch_size']
    device = hyperparams['device']

    # ===============================
    # 2. Paths
    # ===============================
    model_dir = os.path.join(data_path, omics, cancer, 'emb/models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    data_path_ = os.path.join(data_path, omics, cancer)

    # ===============================
    # 3. Dataset (single graph)
    # ===============================
    ds = dataset.Dataset(data_path_)
    dl_train = GraphDataLoader([ds[0]], batch_size=1, shuffle=False)
    dl_valid = GraphDataLoader([ds[1]], batch_size=1, shuffle=False)

    # ===============================
    # 4. Model
    # ===============================
    net = model.TAGCNModel(out_feats, num_layers, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_model = copy.deepcopy(net)

    # ===============================
    # 5. Loss
    # ===============================
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    # ===============================
    # 6. Tracking
    # ===============================
    loss_per_epoch_train, loss_per_epoch_valid = [], []
    f1_per_epoch_train, f1_per_epoch_valid = [], []

    max_f1_scores_train, max_f1_scores_valid = [], []

    best_valid_loss = float('inf')

    # ===============================
    # 7. Results
    # ===============================
    results_path = os.path.join('embedding_omics_pw/results/embeddings/', omics, cancer)
    os.makedirs(results_path, exist_ok=True)

    # ===============================
    # 🔥 INITIAL EMBEDDINGS
    # ===============================
    all_embeddings_initial, cluster_labels_initial, colors = calculate_cluster_labels(net, dl_train)

    plot_clusters(
        all_embeddings_initial, cluster_labels_initial, colors,
        save_path=os.path.join(results_path, "tsne_clusters_initial.png"),
        omics=omics, cancer=cancer
    )

    visualize_embeddings_umap(
        all_embeddings_initial, cluster_labels_initial, colors,
        save_path=os.path.join(results_path, "umap_initial.png"),
        omics=omics, cancer=cancer
    )

    # ===============================
    # 🚀 TRAINING LOOP
    # ===============================
    with tqdm(total=num_epochs) as pbar:

        for epoch in range(num_epochs):

            # -------- TRAIN --------
            net.train()
            train_losses, train_f1s = [], []

            for graph, _ in dl_train:
                graph = graph.to(device)

                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)

                weight_ = weight[labels.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss = (loss * weight_).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                preds = (logits.sigmoid() > 0.3).int()  # 🔥 more stable
                f1 = metrics.f1_score(labels.cpu(), preds.cpu())
                train_f1s.append(f1)

            train_loss = np.mean(train_losses)
            train_f1 = np.mean(train_f1s)

            loss_per_epoch_train.append(train_loss)
            f1_per_epoch_train.append(train_f1)

            # -------- VALID --------
            net.eval()
            val_losses, val_f1s = [], []

            with torch.no_grad():
                for graph, _ in dl_valid:
                    graph = graph.to(device)

                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)

                    weight_ = weight[labels.view(-1).long()].view_as(labels)

                    loss = criterion(logits, labels)
                    loss = (loss * weight_).mean()

                    val_losses.append(loss.item())

                    preds = (logits.sigmoid() > 0.3).int()
                    f1 = metrics.f1_score(labels.cpu(), preds.cpu())
                    val_f1s.append(f1)

            val_loss = np.mean(val_losses)
            val_f1 = np.mean(val_f1s)

            loss_per_epoch_valid.append(val_loss)
            f1_per_epoch_valid.append(val_f1)

            max_f1_scores_train.append(max(f1_per_epoch_train))
            max_f1_scores_valid.append(max(f1_per_epoch_valid))

            # -------- SAVE BEST --------
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

            # -------- LOG --------
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1} | Train F1: {train_f1:.3f} | Val F1: {val_f1:.3f}")

            pbar.update(1)

    # ===============================
    # FINAL CLUSTERING
    # ===============================
    all_embeddings, cluster_labels, colors = calculate_cluster_labels(best_model, dl_train)

    plot_clusters(
        all_embeddings, cluster_labels, colors,
        save_path=os.path.join(results_path, "tsne_clusters_final.png"),
        omics=omics, cancer=cancer
    )

    # ===============================
    # SAVE MODEL
    # ===============================
    torch.save(best_model.state_dict(), model_path)

        # return model_path
    # ===============================
    # PATHS
    # ===============================
    save_path_pca = os.path.join(
        results_path,
        f'embeddings_pca_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    save_path_t_SNE = os.path.join(
        results_path,
        f'embeddings_t-SNE_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    save_path_heatmap_ = os.path.join(
        results_path,
        f'embeddings_heatmap_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    save_path_matrix = os.path.join(
        results_path,
        f'embeddings_matrix_stId_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    # ===============================
    # CONTAINERS
    # ===============================
    cluster_stId_dict = {}
    significant_stIds = []
    clusters_with_significant_stId = {}
    clusters_node_info = {}

    # ===============================
    # EXTRACT EMBEDDINGS
    # ===============================
    for data in dl_train:
        graph, _ = data
        graph = graph.to(device)

        node_embeddings = best_model.get_node_embeddings(graph).detach().cpu().numpy()

        graph_path = os.path.join(data_path, omics, cancer, 'emb/raw', 'emb_train.pkl')
        nx_graph = pickle.load(open(graph_path, 'rb'))

        assert len(cluster_labels) == len(nx_graph.nodes), \
            "Cluster labels and number of nodes must match"

        node_to_index = {node: idx for idx, node in enumerate(nx_graph.nodes)}

        first_node_stId_in_cluster = {}
        first_node_embedding_in_cluster = {}

        stid_dic = {}

        # -----------------------------
        # BUILD stId → embedding
        # -----------------------------
        for node in nx_graph.nodes:
            if 'stId' in nx_graph.nodes[node]:
                stid = nx_graph.nodes[node]['stId']
                idx = node_to_index[node]

                stid_dic[stid] = node_embeddings[idx]

                # ✅ FIX: significance is numeric (0/1)
                if graph.ndata['significance'][idx].item() == 1:
                    significant_stIds.append(stid)

        # -----------------------------
        # SAVE CSV
        # -----------------------------
        stid_df_final = pd.DataFrame.from_dict(stid_dic, orient='index')

        csv_dir = os.path.join('embedding_omics_pw/data/', omics, cancer)
        os.makedirs(csv_dir, exist_ok=True)

        csv_save_path_final = os.path.join(
            csv_dir,
            f'embeddings_lr{learning_rate}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.csv'
        )

        stid_df_final.to_csv(csv_save_path_final, index_label='stId')

        # -----------------------------
        # CLUSTER MAPPING
        # -----------------------------
        for node, cluster in zip(nx_graph.nodes, cluster_labels):

            if 'stId' not in nx_graph.nodes[node]:
                continue

            stid = nx_graph.nodes[node]['stId']
            idx = node_to_index[node]

            # representative node
            if cluster not in first_node_stId_in_cluster:
                first_node_stId_in_cluster[cluster] = stid
                first_node_embedding_in_cluster[cluster] = node_embeddings[idx]

            # cluster → stIds
            cluster_stId_dict.setdefault(cluster, []).append(stid)

            # cluster → significant stIds
            clusters_with_significant_stId.setdefault(cluster, [])
            if stid in significant_stIds:
                clusters_with_significant_stId[cluster].append(stid)

            # cluster → node info
            clusters_node_info.setdefault(cluster, []).append({
                'stId': stid,
                'significance': graph.ndata['significance'][idx].item(),
                'other_info': nx_graph.nodes[node]
            })

        # -----------------------------
        # HEATMAP + COSINE MATRIX
        # -----------------------------
        print(first_node_stId_in_cluster)

        stid_list = list(first_node_stId_in_cluster.values())
        embedding_list = list(first_node_embedding_in_cluster.values())

        create_heatmap_with_stid(
            embedding_list,
            stid_list,
            save_path_heatmap_
        )

        plot_cosine_similarity_matrix_for_clusters_with_values(
            embedding_list,
            stid_list,
            save_path_matrix
        )

        break

    # ===============================
    # FINAL UMAP
    # ===============================
    X, labels, colors = calculate_cluster_labels(best_model, dl_train)

    save_path_umap = os.path.join(
        results_path,
        f'umap_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}_final.png'
    )

    visualize_embeddings_umap(
        X,
        labels,
        colors,
        save_path=save_path_umap,
        omics=omics,
        cancer=cancer,
        method="UMAP + DBSCAN"
    )

    # ===============================
    # CLUSTER METRICS (SAFE)
    # ===============================
    all_embeddings = all_embeddings.reshape(all_embeddings.shape[0], -1)

    if len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(all_embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(all_embeddings, cluster_labels)
    else:
        silhouette_avg = -1
        davies_bouldin = -1

    print(f"Silhouette Score: {silhouette_avg}")
    print(f"Davies-Bouldin Index: {davies_bouldin}")

    # ===============================
    # FIX: MAX F1
    # ===============================
    max_f1_train = max(f1_per_epoch_train) if f1_per_epoch_train else 0.0
    max_f1_valid = max(f1_per_epoch_valid) if f1_per_epoch_valid else 0.0

    # ===============================
    # SUMMARY
    # ===============================
    summary = (
        f"Epoch {num_epochs}\n"
        f"Max F1 Train: {max_f1_train}\n"
        f"Max F1 Valid: {max_f1_valid}\n"
        f"Best Validation Loss: {best_valid_loss}\n"
        f"Silhouette Score: {silhouette_avg}\n"
        f"Davies-Bouldin Index: {davies_bouldin}\n"
    )

    save_file = os.path.join(
        results_path,
        f'embeddings_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.txt'
    )

    with open(save_file, 'w') as f:
        f.write(summary)

    # ===============================
    # SAVE MODEL
    # ===============================
    torch.save(best_model.state_dict(), model_path)

    if plot:
        os.makedirs(results_path, exist_ok=True)

        loss_path = os.path.join(
            results_path,
            f"loss_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png"
        )

        f1_path = os.path.join(
            results_path,
            f"f1_head{num_heads}_dim{out_feats}_lay{num_layers}_epo{num_epochs}.png"
        )

        plot_loss_curve(loss_per_epoch_train, loss_per_epoch_valid, loss_path)
        plot_f1_curve(f1_per_epoch_train, f1_per_epoch_valid, f1_path)

    return model_path

def train_(hyperparams=None, data_path='data', plot=True, omics='mf', cancer='KIRC'):

    import os, copy, pickle
    import numpy as np
    import pandas as pd
    import torch
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn import metrics
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_score,
        recall_score,
        brier_score_loss
    )
    from sklearn.calibration import calibration_curve

    num_epochs = hyperparams['num_epochs']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    device = hyperparams['device']

    # -----------------------------
    # Paths
    # -----------------------------
    model_dir = os.path.join(data_path, omics, cancer, 'emb/models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    results_path = os.path.join(
        'embedding_omics_pw/results/embeddings/',
        omics,
        cancer
    )
    os.makedirs(results_path, exist_ok=True)

    # -----------------------------
    # Dataset
    # -----------------------------
    ds = dataset.Dataset(os.path.join(data_path, omics, cancer))
    dl_train = GraphDataLoader([ds[0]], batch_size=1, shuffle=False)
    dl_valid = GraphDataLoader([ds[1]], batch_size=1, shuffle=False)

    # -----------------------------
    # Model
    # -----------------------------
    net = model.TAGCNModel(out_feats, num_layers, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_model = copy.deepcopy(net)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    # -----------------------------
    # Tracking
    # -----------------------------
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "val_auc": [],
        "val_pr_auc": [],
        "val_brier": []
    }

    best_pr_auc = 0.0  # 🔥 better for imbalance

    # =============================
    # TRAIN LOOP
    # =============================
    with tqdm(total=num_epochs, desc="Training") as pbar:

        for epoch in range(num_epochs):

            # ---------------- TRAIN ----------------
            net.train()
            train_losses, train_preds, train_labels = [], [], []

            for graph, _ in dl_train:
                graph = graph.to(device)

                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)

                weight_ = weight[labels.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss = (loss * weight_).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                probs = logits.sigmoid().detach().cpu().numpy()
                preds = (probs > 0.3).astype(int)

                train_preds.extend(preds.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

            # Metrics
            train_loss = np.mean(train_losses)
            train_f1 = metrics.f1_score(train_labels, train_preds)

            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)

            # ---------------- VALID ----------------
            net.eval()
            val_losses, val_probs, val_labels = [], [], []

            with torch.no_grad():
                for graph, _ in dl_valid:
                    graph = graph.to(device)

                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)

                    weight_ = weight[labels.view(-1).long()].view_as(labels)

                    loss = criterion(logits, labels)
                    loss = (loss * weight_).mean()

                    val_losses.append(loss.item())

                    probs = logits.sigmoid().cpu().numpy()
                    val_probs.extend(probs.flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())

            val_loss = np.mean(val_losses)

            val_preds = (np.array(val_probs) > 0.3).astype(int)

            val_f1 = metrics.f1_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs)
            val_pr_auc = average_precision_score(val_labels, val_probs)
            val_precision = precision_score(val_labels, val_preds)
            val_recall = recall_score(val_labels, val_preds)

            # Calibration
            val_brier = brier_score_loss(val_labels, val_probs)

            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_auc"].append(val_auc)
            history["val_pr_auc"].append(val_pr_auc)
            history["val_brier"].append(val_brier)

            # ---------------- SAVE BEST ----------------
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

            # ---------------- LOG ----------------
            if (epoch + 1) % 5 == 0:
                print(
                    f"[Epoch {epoch+1}] "
                    f"F1={val_f1:.3f} | "
                    f"AUC={val_auc:.3f} | "
                    f"PR-AUC={val_pr_auc:.3f} | "
                    f"Brier={val_brier:.4f}"
                )

            pbar.update(1)

    # =============================
    # SAVE MODEL
    # =============================
    torch.save(best_model.state_dict(), model_path)

    # =============================
    # SAVE METRICS
    # =============================
    metrics_df = pd.DataFrame(history)
    metrics_csv = os.path.join(results_path, "training_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    summary_txt = os.path.join(results_path, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"Best PR-AUC: {best_pr_auc}\n")
        f.write(f"Final F1: {history['val_f1'][-1]}\n")
        f.write(f"Final ROC-AUC: {history['val_auc'][-1]}\n")

    # =============================
    # CALIBRATION CURVE
    # =============================
    if plot:
        import matplotlib.pyplot as plt

        prob_true, prob_pred = calibration_curve(
            val_labels,
            val_probs,
            n_bins=10
        )

        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("Calibration Curve")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.savefig(os.path.join(results_path, "calibration_curve.png"))
        plt.close()

    return model_path

def train(hyperparams=None, data_path='data', plot=True, omics='mf', cancer='KIRC'):

    # =============================
    # 🔥 Precision@K FUNCTIONS
    # =============================
    def precision_at_k(y_true, y_scores, k):
        idx = np.argsort(y_scores)[::-1][:k]
        return np.sum(y_true[idx]) / k

    def ranking_metrics_at_k(y_true, y_scores, ks=[10, 20, 50, 100]):
        results = {}
        for k in ks:
            results[f"P@{k}"] = precision_at_k(y_true, y_scores, k)
        return results

    def plot_precision_at_k_curve(y_true, y_scores, max_k=200, save_path=None):
        ks = list(range(1, max_k + 1))
        precisions = [precision_at_k(y_true, y_scores, k) for k in ks]

        plt.figure(figsize=(6, 4))
        plt.plot(ks, precisions, linewidth=2)
        plt.xlabel("K")
        plt.ylabel("Precision@K")
        plt.title("Precision@K Curve")
        plt.grid(False)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    # =============================
    # HYPERPARAMETERS
    # =============================
    num_epochs = hyperparams['num_epochs']
    out_feats = hyperparams['out_feats']
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    learning_rate = hyperparams['lr']
    device = hyperparams['device']

    # =============================
    # PATHS
    # =============================
    model_dir = os.path.join(data_path, omics, cancer, 'emb/models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir,
        f'model_dim{out_feats}_lay{num_layers}_epo{num_epochs}.pth'
    )

    results_path = os.path.join(
        'embedding_omics_pw/results/embeddings/',
        omics,
        cancer
    )
    os.makedirs(results_path, exist_ok=True)

    # =============================
    # DATASET (SINGLE GRAPH)
    # =============================
    ds = dataset.Dataset(os.path.join(data_path, omics, cancer))
    dl_train = GraphDataLoader([ds[0]], batch_size=1, shuffle=False)
    dl_valid = GraphDataLoader([ds[1]], batch_size=1, shuffle=False)

    # =============================
    # MODEL
    # =============================
    net = model.TAGCNModel(out_feats, num_layers, do_train=True).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    best_model = copy.deepcopy(net)

    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    weight = torch.tensor([0.00001, 0.99999]).to(device)

    # =============================
    # TRACKING
    # =============================
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "val_auc": [],
        "val_pr_auc": [],
        "val_brier": [],
        "P@10": [],
        "P@20": [],
        "P@50": [],
        "P@100": []
    }

    best_pr_auc = 0.0

    # =============================
    # TRAIN LOOP
    # =============================
    with tqdm(total=num_epochs, desc="Training") as pbar:

        for epoch in range(num_epochs):

            # -------- TRAIN --------
            net.train()
            train_losses, train_preds, train_labels = [], [], []

            for graph, _ in dl_train:
                graph = graph.to(device)

                logits = net(graph)
                labels = graph.ndata['significance'].unsqueeze(-1)

                weight_ = weight[labels.view(-1).long()].view_as(labels)

                loss = criterion(logits, labels)
                loss = (loss * weight_).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                probs = logits.sigmoid().detach().cpu().numpy()
                preds = (probs > 0.3).astype(int)

                train_preds.extend(preds.flatten())
                train_labels.extend(labels.cpu().numpy().flatten())

            train_loss = np.mean(train_losses)
            train_f1 = metrics.f1_score(train_labels, train_preds)

            history["train_loss"].append(train_loss)
            history["train_f1"].append(train_f1)

            # -------- VALID --------
            net.eval()
            val_losses, val_probs, val_labels = [], [], []

            with torch.no_grad():
                for graph, _ in dl_valid:
                    graph = graph.to(device)

                    logits = net(graph)
                    labels = graph.ndata['significance'].unsqueeze(-1)

                    weight_ = weight[labels.view(-1).long()].view_as(labels)

                    loss = criterion(logits, labels)
                    loss = (loss * weight_).mean()

                    val_losses.append(loss.item())

                    probs = logits.sigmoid().cpu().numpy()
                    val_probs.extend(probs.flatten())
                    val_labels.extend(labels.cpu().numpy().flatten())

            val_loss = np.mean(val_losses)
            val_preds = (np.array(val_probs) > 0.3).astype(int)

            val_f1 = metrics.f1_score(val_labels, val_preds)
            val_auc = roc_auc_score(val_labels, val_probs)
            val_pr_auc = average_precision_score(val_labels, val_probs)
            val_brier = brier_score_loss(val_labels, val_probs)

            # 🔥 Precision@K
            p_at_k = ranking_metrics_at_k(np.array(val_labels), np.array(val_probs))

            history["val_loss"].append(val_loss)
            history["val_f1"].append(val_f1)
            history["val_auc"].append(val_auc)
            history["val_pr_auc"].append(val_pr_auc)
            history["val_brier"].append(val_brier)

            for k, v in p_at_k.items():
                history[k].append(v)

            # -------- SAVE BEST --------
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_model.load_state_dict(copy.deepcopy(net.state_dict()))

            # -------- LOG --------
            if (epoch + 1) % 5 == 0:
                print(
                    f"[Epoch {epoch+1}] "
                    f"F1={val_f1:.3f} | "
                    f"AUC={val_auc:.3f} | "
                    f"PR-AUC={val_pr_auc:.3f} | "
                    f"P@50={p_at_k['P@50']:.3f}"
                )

            pbar.update(1)

    # =============================
    # SAVE MODEL
    # =============================
    torch.save(best_model.state_dict(), model_path)

    # =============================
    # SAVE METRICS
    # =============================
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(os.path.join(results_path, "training_metrics.csv"), index=False)

    # =============================
    # FINAL PRECISION@K CURVE
    # =============================
    plot_precision_at_k_curve(
        np.array(val_labels),
        np.array(val_probs),
        save_path=os.path.join(results_path, "precision_at_k_curve.png")
    )

    # =============================
    # CALIBRATION CURVE
    # =============================
    if plot:
        prob_true, prob_pred = calibration_curve(val_labels, val_probs, n_bins=10)

        plt.figure()
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title("Calibration Curve")
        plt.xlabel("Predicted Probability")
        plt.ylabel("True Probability")
        plt.grid(False)

        plt.savefig(os.path.join(results_path, "calibration_curve.png"))
        plt.close()

    # =============================
    # SUMMARY
    # =============================
    with open(os.path.join(results_path, "summary.txt"), "w") as f:
        f.write(f"Best PR-AUC: {best_pr_auc}\n")
        f.write(f"Final F1: {history['val_f1'][-1]}\n")
        f.write(f"Final ROC-AUC: {history['val_auc'][-1]}\n")
        f.write(f"Final P@10: {history['P@10'][-1]}\n")
        f.write(f"Final P@50: {history['P@50'][-1]}\n")

    if plot:
        plot_training_curves(history, results_path)
        
    return model_path

def plot_training_curves(history, results_path):

    import os
    import matplotlib.pyplot as plt

    os.makedirs(results_path, exist_ok=True)

    def _plot(metric_keys, title, ylabel, filename):
        plt.figure(figsize=(6, 4))

        for key in metric_keys:
            if key in history and len(history[key]) > 0:
                plt.plot(history[key], label=key, linewidth=2)

        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)

        plt.legend()
        plt.grid(False)  # 🔥 NO GRID

        plt.tight_layout()
        plt.savefig(os.path.join(results_path, filename), dpi=300)
        plt.close()

    # ===============================
    # LOSS
    # ===============================
    _plot(
        ["train_loss", "val_loss"],
        "Training vs Validation Loss",
        "Loss",
        "loss_curve.png"
    )

    # ===============================
    # F1
    # ===============================
    _plot(
        ["train_f1", "val_f1"],
        "Training vs Validation F1",
        "F1 Score",
        "f1_curve.png"
    )

    # ===============================
    # AUC
    # ===============================
    _plot(
        ["val_auc"],
        "Validation ROC-AUC",
        "AUC",
        "roc_auc_curve.png"
    )

    # ===============================
    # PR-AUC
    # ===============================
    _plot(
        ["val_pr_auc"],
        "Validation PR-AUC",
        "PR-AUC",
        "pr_auc_curve.png"
    )

    # ===============================
    # BRIER (Calibration)
    # ===============================
    _plot(
        ["val_brier"],
        "Validation Brier Score",
        "Brier Score",
        "brier_curve.png"
    )

    # ===============================
    # PRECISION@K
    # ===============================
    _plot(
        ["P@10", "P@20", "P@50", "P@100"],
        "Precision@K",
        "Precision",
        "precision_at_k_curves.png"
    )

# ===============================
# 📉 LOSS PLOT
# ===============================
def plot_loss_curve(train_loss, val_loss, save_path):
    plt.figure(figsize=(6, 4))

    plt.plot(train_loss, label="Train", linewidth=2)
    plt.plot(val_loss, label="Validation", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")

    plt.legend()
    plt.grid(False)  # ❌ remove grid (clean look)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved loss curve → {save_path}")


# ===============================
# 📊 F1 SCORE PLOT
# ===============================
def plot_f1_curve(train_f1, val_f1, save_path):
    plt.figure(figsize=(6, 4))

    plt.plot(train_f1, label="Train", linewidth=2)
    plt.plot(val_f1, label="Validation", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training vs Validation F1 Score")

    plt.legend()
    plt.grid(False)  # ❌ remove grid

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved F1 curve → {save_path}")

def visualize_embeddings_umap_(X, labels, colors, save_path=None):
    """
    UMAP visualization of embeddings
    
    X: (N, d) embeddings
    labels: cluster labels
    colors: dict {cluster_id: color}
    """

    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.3,
        init="random",
        random_state=42
    )

    X = X + np.random.normal(0, 1e-5, X.shape)

    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for c in set(labels):
        idx = labels == c
        color = colors.get(c, "#000000")

        label_name = f"Cluster {c}" if c != -1 else "Noise"

        plt.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            s=10,
            c=color,
            label=label_name,
            alpha=0.8
        )

    plt.legend(markerscale=2, fontsize=8)
    plt.title("UMAP Projection of GNN Embeddings")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()


def visualize_embeddings_umap(X, labels, colors, save_path=None,
                              omics=None, cancer=None,
                              method="DBSCAN + UMAP"):

    import umap
    import matplotlib.pyplot as plt

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )

    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))

    # ===============================
    # PLOT CLUSTERS
    # ===============================
    for c in sorted(set(labels)):
        idx = labels == c

        color = colors.get(c, "#000000")
        label_name = f"Cluster {c}" if c != -1 else "Noise"

        # filled points
        plt.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            s=10,
            c=color,
            alpha=0.5,
            linewidths=0
        )

        # outline points (for legend clarity)
        plt.scatter(
            X_umap[idx, 0],
            X_umap[idx, 1],
            s=14,
            facecolors='none',
            edgecolors=color,
            linewidths=0.8,
            label=label_name
        )

    # ===============================
    # LEGEND (clean + readable)
    # ===============================
    plt.legend(
        fontsize=8,
        markerscale=1.5,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    plt.grid(False)

    # ===============================
    # TITLE
    # ===============================
    title = ""
    if method:
        title += f"{method}"
    if omics:
        title += f" | Omics: {omics}"
    if cancer:
        title += f" | Cancer: {cancer}"

    plt.title(title)
    plt.tight_layout()

    # ===============================
    # SAVE
    # ===============================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved UMAP plot → {save_path}")

    plt.close()

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_clusters(X, labels, cluster_colors, save_path=None,
                  omics=None, cancer=None, method="DBSCAN"):

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))

    for c in sorted(set(labels)):
        idx = (labels == c)

        color = cluster_colors.get(c, "#000000")
        label_name = "Noise" if c == -1 else f"Cluster {c}"

        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=10,
            c=color,
            label=label_name,
            alpha=0.5
        )

    # remove grid
    plt.grid(False)

    # legend
    plt.legend(
        markerscale=2,
        fontsize=8,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # -----------------------------
    # TITLE with full metadata
    # -----------------------------
    title = ""

    if method:
        title += f"{method}"

    if omics:
        title += f" | Omics: {omics}"

    if cancer:
        title += f" | Cancer: {cancer}"

    plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster plot → {save_path}")

    plt.close()


def _plot_clusters(X, labels, cluster_colors, save_path=None, omics=None, cancer=None):

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    X_2d = TSNE(n_components=2, random_state=42).fit_transform(X)

    plt.figure(figsize=(8, 6))

    for c in sorted(set(labels)):
        idx = (labels == c)

        color = cluster_colors.get(c, "#000000")
        label_name = "Noise" if c == -1 else f"Cluster {c}"

        plt.scatter(
            X_2d[idx, 0],
            X_2d[idx, 1],
            s=10,
            c=color,
            label=label_name,
            alpha=0.8
        )

    # remove grid
    plt.grid(False)

    # legend
    plt.legend(
        markerscale=2,
        fontsize=8,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    # -----------------------------
    # TITLE with metadata
    # -----------------------------
    title = "Cluster Visualization"
    if omics is not None:
        title += f" | Omics: {omics}"
    if cancer is not None:
        title += f" | Cancer: {cancer}"

    plt.title(title)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster plot → {save_path}")

    plt.close()

def plot_cosine_similarity_matrix_for_clusters_with_values(embeddings, stids, save_path):
    cos_sim = np.dot(embeddings, np.array(embeddings).T)
    norms = np.linalg.norm(embeddings, axis=1)
    cos_sim /= np.outer(norms, norms)

    plt.figure(figsize=(10, 8))
    
    vmin = cos_sim.min()
    vmax = cos_sim.max()
    # Create the heatmap with a custom color bar
    ##sns.heatmap(data, cmap='cividis')
    ##sns.heatmap(data, cmap='Blues') 'Greens' sns.heatmap(data, cmap='Spectral') 'coolwarm') 'YlGnBu') viridis cubehelix inferno

    ax = sns.heatmap(cos_sim, cmap="Spectral", annot=True, fmt=".3f", annot_kws={"size": 6},
                     xticklabels=stids, yticklabels=stids,
                     cbar_kws={"shrink": 0.2, "aspect": 15, "ticks": [vmin, vmax]})

    # Highlight the diagonal squares with value 1 by setting their background color to black
    for i in range(len(stids)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', alpha=0.5, zorder=3))
        
    ax.xaxis.tick_top()  # Move x-axis labels to the top
    ax.xaxis.set_label_position('top')  # Set x-axis label position to top
    plt.xticks(rotation=-30, fontsize=8, ha='right')  # Rotate x-axis labels, set font size, and align to the right
    plt.yticks(fontsize=8)  # Set font size for y-axis labels

    # Set the title below the plot
    ax.text(x=0.5, y=-0.03, s="Gene-gene similarities", fontsize=12, ha='center', va='top', transform=ax.transAxes)

    plt.savefig(save_path)
    ##plt.show()
    plt.close()
    
def create_gene_map(reactome_file, output_file):
    """
    Extracts gene IDs with the same gene STID and saves them to a new CSV file.

    Parameters:
    reactome_file (str): Path to the NCBI2Reactome.csv file.
    output_file (str): Path to save the output CSV file.
    """
    gene_map = {}  # Dictionary to store gene IDs for each gene STID

    # Read the NCBI2Reactome.csv file and populate the gene_map
    with open(reactome_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            gene_id = row[0]
            gene_stid = row[1]
            gene_map.setdefault(gene_stid, []).append(gene_id)

    # Write the gene_map to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Protein STID", "Gene IDs"])  # Write header
        for gene_stid, gene_ids in gene_map.items():
            writer.writerow([gene_stid, ",".join(gene_ids)])
    
    return gene_map
        
def save_to_neo4j(graph, stid_dic, stid_mapping, gene_map, gene_id_to_name_mapping, gene_id_to_symbol_mapping, uri, user, password):
    from neo4j import GraphDatabase

    # Connect to Neo4j
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()

    # Clean the database
    session.run("MATCH (n) DETACH DELETE n")

    try:
        # Create nodes with embeddings and additional attributes
        for node_id in stid_dic:
            embedding = stid_dic[node_id].tolist()  
            stId = stid_mapping[node_id]  # Access stId based on node_id
            name = graph.graph_nx.nodes[node_id]['name']
            weight = graph.graph_nx.nodes[node_id]['weight']
            significance = graph.graph_nx.nodes[node_id]['significance']
            session.run(
                "CREATE (n:Protein {embedding: $embedding, stId: $stId, name: $name, weight: $weight, significance: $significance})",
                embedding=embedding, stId=stId, name=name, weight=weight, significance=significance
            )

            # Create gene nodes and relationships
            ##genes = get_genes_by_gene_stid(node_id, reactome_file, gene_names_file)
            genes = gene_map.get(node_id, [])


            ##print('stid_to_gene_info=========================-----------------------------\n', genes)
    
            # Create gene nodes and relationships
            for gene_id in genes:
                gene_name = gene_id_to_name_mapping.get(gene_id, None)
                gene_symbol = gene_id_to_symbol_mapping.get(gene_id, None)
                if gene_name:  # Only create node if gene name exists
                    session.run(
                        "MERGE (g:Gene {id: $gene_id, name: $gene_name, symbol: $gene_symbol})",
                        gene_id=gene_id, gene_name=gene_name, gene_symbol = gene_symbol
                    )
                    session.run(
                        "MATCH (p:Protein {stId: $stId}), (g:Gene {id: $gene_id}) "
                        "MERGE (p)-[:INVOLVES]->(g)",
                        stId=stId, gene_id=gene_id
                    )
                
                session.run(
                    "MATCH (p:Protein {stId: $stId}), (g:Gene {id: $gene_id}) "
                    "MERGE (p)-[:INVOLVES]->(g)",
                    stId=stId, gene_id=gene_id
                )
                
        # Create relationships using the stId mapping
        for source, target in graph.graph_nx.edges():
            source_stId = stid_mapping[source]
            target_stId = stid_mapping[target]
            session.run(
                "MATCH (a {stId: $source_stId}), (b {stId: $target_stId}) "
                "CREATE (a)-[:CONNECTED]->(b)",
                source_stId=source_stId, target_stId=target_stId
            )

    finally:
        session.close()
        driver.close()

def read_gene_names(file_path):
    """
    Reads the gene names from a CSV file and returns a dictionary mapping gene IDs to gene names.

    Parameters:
    file_path (str): Path to the gene names CSV file.

    Returns:
    dict: A dictionary mapping gene IDs to gene names.
    """
    gene_id_to_name_mapping = {}
    gene_id_to_symbol_mapping = {}

    # Read the gene names CSV file and populate the dictionary
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            gene_id = row['NCBI_Gene_ID']
            gene_name = row['Name']
            gene_symbol = row['Approved symbol']
            gene_id_to_name_mapping[gene_id] = gene_name
            gene_id_to_symbol_mapping[gene_id] = gene_symbol

    return gene_id_to_name_mapping, gene_id_to_symbol_mapping

def create_heatmap_with_stid_ori(embedding_list, stid_list, save_path):
    # Convert the embedding list to a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='tab20', standard_scale=1, figsize=(10, 10))
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    plt.close()

def create_heatmap_with_stid_dark_green(embedding_list, stid_list, save_path):
    # Convert the embedding list to a DataFrame and transpose it to switch axes
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()
    
    # Create a clustermap
    ax = sns.clustermap(heatmap_data, cmap='viridis', standard_scale=1, figsize=(10, 10))
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_stid_grey(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using a grey-dark white colormap.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert the embedding list to a DataFrame and transpose it
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()
    
    # Create a clustermap with a grey-dark white colormap
    ax = sns.clustermap(
        heatmap_data, 
        cmap=cm.get_cmap('Greys'),  # Use Greys colormap
        standard_scale=1, 
        figsize=(10, 10)
    )
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels
    
    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_10_discrete_colors(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using 10 discrete colors.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert the embedding list to a DataFrame and transpose it
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()

    # Define 10 discrete colors
    discrete_colors = [
        '#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4',
        '#4eb3d3', '#2b8cbe', '#0868ac', '#084081', '#081d58'
    ]  # Gradient from light green to dark blue

    # Define value ranges (bins) for the colors
    color_bounds = np.linspace(heatmap_data.min().min(), heatmap_data.max().max(), len(discrete_colors) + 1)

    # Create a discrete colormap and norm
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(color_bounds, cmap.N)

    # Create a clustermap
    ax = sns.clustermap(
        heatmap_data,
        cmap=cmap, 
        norm=norm, 
        figsize=(10, 10),
        linewidths=0.5  # Add gridlines for clarity
    )
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels

    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_stid_light_blue(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using 10 discrete colors.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert the embedding list to a DataFrame and transpose it
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list).transpose()
    ##heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

    # Define 10 discrete colors
    discrete_colors = [
        '#f7fcf0', '#e0f3db', '#ccebc5', '#a8ddb5', '#7bccc4',
        '#4eb3d3', '#2b8cbe', '#0868ac', '#084081', '#081d58'
    ]  # Gradient from light green to dark blue

    # Define value ranges (bins) for the colors
    color_bounds = np.linspace(heatmap_data.min().min(), heatmap_data.max().max(), len(discrete_colors) + 1)

    # Create a discrete colormap and norm
    cmap = ListedColormap(discrete_colors)
    norm = BoundaryNorm(color_bounds, cmap.N)

    # Create a clustermap
    ax = sns.clustermap(
        heatmap_data,
        cmap=cmap, 
        norm=norm, 
        figsize=(10, 10),
        linewidths=0.5  # Add gridlines for clarity
    )
    
    # Set smaller font sizes for various elements
    ax.ax_heatmap.tick_params(axis='both', which='both', labelsize=8)  # Tick labels
    ax.ax_heatmap.set_xlabel(ax.ax_heatmap.get_xlabel(), fontsize=8)  # X-axis label
    ax.ax_heatmap.set_ylabel(ax.ax_heatmap.get_ylabel(), fontsize=8)  # Y-axis label
    ax.ax_heatmap.collections[0].colorbar.ax.tick_params(labelsize=8)  # Color bar labels

    # Save the clustermap to a file
    plt.savefig(save_path)

    # Close the plot to free memory
    plt.close()

def create_heatmap_with_stid(embedding_list, stid_list, save_path):
    """
    Creates a heatmap with hierarchical clustering using a better colormap.
    Ensures no spaces between cells and limits x-axis labels to 30.

    Parameters:
    - embedding_list: List of embeddings.
    - stid_list: List of sample or feature IDs corresponding to embeddings.
    - save_path: Path to save the heatmap.
    """
    # Convert embeddings into a DataFrame
    heatmap_data = pd.DataFrame(embedding_list, index=stid_list)

    # Create the clustermap with improved settings
    ax = sns.clustermap(
        heatmap_data, 
        cmap="coolwarm",  # Use 'coolwarm' or 'viridis' for clarity
        standard_scale=1, 
        figsize=(8, 7.5),  # Larger size for better readability
        linewidths=0,  # Remove spaces between cells
        dendrogram_ratio=(0.1, 0.1),  # Reduce dendrogram size
        ##xticklabels=30,  # Show only 30 x-axis ticks
        cbar_pos=(1.00, 0.4, 0.01, 0.2),  # Adjusted for better spacing
    )

    # Adjust color bar font size
    cbar = ax.ax_cbar  # Get the color bar axis
    ##cbar.set_ylabel('Intensity', fontsize=6)  # Set color bar label font size
    cbar.tick_params(labelsize=6)  # Set tick label font size
    # Adjust x-axis labels
    ax.ax_heatmap.set_xticklabels(ax.ax_heatmap.get_xticklabels(), rotation=90, fontsize=8)
    ax.ax_heatmap.set_yticklabels(ax.ax_heatmap.get_yticklabels(), fontsize=8)

    # Save the figure
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def calculate_cluster_labels_ori(net, dataloader, num_clusters=7):
    all_embeddings = []
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            embeddings = net.get_node_embeddings(graph.to('cpu'))
            all_embeddings.append(embeddings)
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Use KMeans clustering to assign cluster labels
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(all_embeddings)
    return all_embeddings, cluster_labels

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import torch

def calculate_cluster_labels_(net, dataloader, k_min=2, k_max=15):
    all_embeddings = []
    net.eval()

    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            emb = net.get_node_embeddings(graph.to('cpu'))
            all_embeddings.append(emb.cpu().numpy())

    X = np.concatenate(all_embeddings, axis=0)

    best_k = None
    best_score = -1
    best_labels = None

    # 🔍 search optimal k
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # avoid degenerate case
        if len(set(labels)) < 2:
            continue

        score = silhouette_score(X, labels)

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    print(f"[INFO] Best k = {best_k}, silhouette = {best_score:.4f}")

    return X, best_labels


# _dbscan
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np

COLORS = [
    '#0077B6','#0000FF','#00B4D8','#48EAC4','#F1C0E8','#B9FBC0',
    '#32CD32','#BEE1E6','#8A2BE2','#E377C2','#8EECF5','#A3C4F3',
    '#FFB347','#FFD700','#FF69B4','#CD5C5C','#7FFFD4','#FF7F50',
    '#C71585','#20B2AA','#6A5ACD','#40E0D0','#FF8C00','#DC143C',
    '#9ACD32'
]

MAX_CLUSTERS = 25


def calculate_cluster_labels(net, dataloader, eps=0.5, min_samples=3):

    # =============================
    # 1. Extract embeddings
    # =============================
    all_embeddings = []
    net.eval()

    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            emb = net.get_node_embeddings(graph.to('cpu'))
            all_embeddings.append(emb.cpu().numpy())

    X = np.concatenate(all_embeddings, axis=0)

    # =============================
    # 2. Normalize
    # =============================
    X = StandardScaler().fit_transform(X)

    # =============================
    # 3. DBSCAN
    # =============================
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    unique_clusters = sorted(set(labels) - {-1})
    n_clusters = len(unique_clusters)

    # =============================
    # 4. Auto-adjust clusters
    # =============================
    TARGET_CLUSTERS = 15  # 👈 control here

    if n_clusters < 2 or n_clusters > MAX_CLUSTERS:
        print(f"[INFO] Switching to KMeans (target={TARGET_CLUSTERS})")

        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=TARGET_CLUSTERS, random_state=42).fit_predict(X)

    # =============================
    # 4. Handle ALL NOISE case
    # =============================
    if len(set(labels)) == 1 and -1 in labels:
        print("[WARN] All points are classified as noise.")
        cluster_colors = {-1: "#CCCCCC"}
        return X, labels, cluster_colors

    # =============================
    # 5. Count clusters
    # =============================
    unique_clusters = sorted(set(labels) - {-1})
    n_clusters = len(unique_clusters)

    print(f"[INFO] clusters found: {n_clusters}")
    print(f"[INFO] noise points: {(labels == -1).sum()}")

    # =============================
    # 6. Limit clusters
    # =============================
    if n_clusters > MAX_CLUSTERS:
        print(f"[WARN] Reducing clusters from {n_clusters} → {MAX_CLUSTERS}")

        cluster_sizes = {
            c: np.sum(labels == c)
            for c in unique_clusters
        }

        top_clusters = sorted(
            cluster_sizes,
            key=cluster_sizes.get,
            reverse=True
        )[:MAX_CLUSTERS]

        # remap to contiguous labels
        cluster_map = {c: i for i, c in enumerate(top_clusters)}

        new_labels = np.full_like(labels, -1)

        for old_c, new_c in cluster_map.items():
            new_labels[labels == old_c] = new_c

        labels = new_labels

    # =============================
    # 7. Re-index clusters (clean)
    # =============================
    unique_clusters = sorted(set(labels) - {-1})
    cluster_map = {c: i for i, c in enumerate(unique_clusters)}

    final_labels = np.full_like(labels, -1)

    for old_c, new_c in cluster_map.items():
        final_labels[labels == old_c] = new_c

    labels = final_labels

    n_clusters = len(unique_clusters)

    print(f"[INFO] final clusters: {n_clusters}")

    # =============================
    # 8. Assign colors
    # =============================
    cluster_colors = {}

    for c in set(labels):
        if c == -1:
            cluster_colors[c] = "#CCCCCC"
        else:
            cluster_colors[c] = COLORS[c % len(COLORS)]

    return X, labels, cluster_colors

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# # -----------------------------
# # Predefined colors
# # -----------------------------
# COLORS = [
#     '#0077B6','#0000FF','#00B4D8','#48EAC4','#F1C0E8','#B9FBC0',
#     '#32CD32','#BEE1E6','#8A2BE2','#E377C2','#8EECF5','#A3C4F3',
#     '#FFB347','#FFD700','#FF69B4','#CD5C5C','#7FFFD4','#FF7F50',
#     '#C71585','#20B2AA','#6A5ACD','#40E0D0','#FF8C00','#DC143C',
#     '#9ACD32','#1F77B4','#FF1493','#2E8B57','#D2691E','#9932CC',
#     '#00CED1','#FF4500','#708090'
# ]

# -----------------------------
# Auto-estimate eps (k-distance)
# -----------------------------
def estimate_eps(X, k=5):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])
    
    # heuristic: take 90th percentile
    eps = np.percentile(kth_dist, 90)
    return eps


# -----------------------------
# DBSCAN clustering
# -----------------------------
def calculate_cluster_labels__(net, dataloader, eps=None, min_samples=5, max_clusters=25):
    all_embeddings = []
    net.eval()

    with torch.no_grad():
        for data in dataloader:
            graph, _ = data
            emb = net.get_node_embeddings(graph.to('cpu'))
            all_embeddings.append(emb.cpu().numpy())

    X = np.concatenate(all_embeddings, axis=0)

    # -----------------------------
    # Normalize (critical)
    # -----------------------------
    X = StandardScaler().fit_transform(X)

    # -----------------------------
    # Auto eps if not provided
    # -----------------------------
    if eps is None:
        eps = estimate_eps(X, k=min_samples)
        print(f"[INFO] Auto eps = {eps:.4f}")

    # -----------------------------
    # Run DBSCAN
    # -----------------------------
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    # -----------------------------
    # Stats
    # -----------------------------
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    print(f"[INFO] clusters found: {n_clusters}")
    print(f"[INFO] noise points: {n_noise}")

    # -----------------------------
    # Limit clusters if too many
    # -----------------------------
    if n_clusters > max_clusters:
        print(f"[WARN] Too many clusters ({n_clusters}), merging small clusters")

        # keep largest clusters only
        counts = {c: (labels == c).sum() for c in set(labels) if c != -1}
        sorted_clusters = sorted(counts.items(), key=lambda x: -x[1])

        keep_clusters = [c for c, _ in sorted_clusters[:max_clusters]]

        # reassign others to noise
        new_labels = []
        for c in labels:
            if c in keep_clusters:
                new_labels.append(c)
            else:
                new_labels.append(-1)

        labels = np.array(new_labels)
        n_clusters = len(keep_clusters)

    # -----------------------------
    # Assign colors
    # -----------------------------
    unique_clusters = sorted([c for c in set(labels) if c != -1])

    cluster_to_color = {
        c: COLORS[i % len(COLORS)]
        for i, c in enumerate(unique_clusters)
    }

    # noise = gray
    node_colors = [
        cluster_to_color[c] if c != -1 else "#B0B0B0"
        for c in labels
    ]

    return X, labels, node_colors

def visualize_embeddings_pca(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_pca_ori(embeddings, cluster_labels, stid_list, save_path):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))

    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i])

    # Add labels and title
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
def visualize_embeddings_tsne_ori(embeddings, cluster_labels, stid_list, save_path):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style
    sns.set(style="whitegrid")

    # Define unique clusters and sort them
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)  # Sort the clusters

    # Define a color palette
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot with a continuous colormap
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'{stid_list[cluster]}', s=20, color=palette[i], edgecolor='k')

    # Add labels and title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    # Ensure the plot is square
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend with dot shapes and stid labels
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], markersize=8, label=stid_list[cluster]) for i, cluster in enumerate(sorted_clusters)]
    plt.legend(handles=handles, title='Label', bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0., fontsize='small', handlelength=0.5, handletextpad=0.5)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_embeddings_tsne(embeddings, cluster_labels, stid_list, save_path):
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Initialize the figure
    plt.figure(figsize=(10, 10))  # Square figure

    # Set the style for the plot
    sns.set(style="whitegrid")  # White background with grid lines

    # Identify unique clusters and sort them for consistent ordering
    unique_clusters = np.unique(cluster_labels)
    sorted_clusters = sorted(unique_clusters)

    # Define a color palette for the clusters
    palette = sns.color_palette("viridis", len(sorted_clusters))
    
    # Create a scatter plot for each cluster
    for i, cluster in enumerate(sorted_clusters):
        cluster_points = embeddings_2d[cluster_labels == cluster]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f'{stid_list[cluster]}',  # Label based on stid_list
            s=20,  # Size of the scatter points
            color=palette[i],  # Color based on the cluster
            edgecolor='k'  # Black edge around each point
        )

    # Add axis labels and a title
    plt.xlabel('dim_1')
    plt.ylabel('dim_2')
    plt.title('T-SNE of Embeddings')

    # Customize the plot grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')  # Light gray background for the plot area
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)

    # Maintain square aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')

    # Create a custom legend to label clusters
    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=palette[i], markersize=8,
            label=stid_list[cluster]
        ) for i, cluster in enumerate(sorted_clusters)
    ]
    plt.legend(
        handles=handles,
        title='Label',
        bbox_to_anchor=(1.02, 0.5),
        loc='center left',
        borderaxespad=0.,
        fontsize='small',
        handlelength=0.5,
        handletextpad=0.5
    )

    # Save the plot to the specified path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def export_to_cytoscape(node_embeddings, cluster_labels, stid_list, output_path):
    # Create a DataFrame for Cytoscape export
    data = {
        'Node': stid_list,
        'Cluster': cluster_labels,
        'Embedding': list(node_embeddings)
    }
    df = pd.DataFrame(data)
    
    # Expand the embedding column into separate columns
    embeddings_df = pd.DataFrame(node_embeddings, columns=[f'Embed_{i}' for i in range(node_embeddings.shape[1])])
    df = df.drop('Embedding', axis=1).join(embeddings_df)

    # Save to CSV for Cytoscape import
    df.to_csv(output_path, index=False)
    print(f"Data exported to {output_path} for Cytoscape visualization.")

def draw_loss_plot(train_loss, valid_loss, save_path):
    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(train_loss, label='train')
    plt.plot(valid_loss, label='validation')

    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # -----------------------------
    # Clean style (NO GRID + WHITE BACKGROUND)
    # -----------------------------
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def draw_max_f1_plot(max_train_f1, max_valid_f1, save_path):
    plt.figure()
    plt.plot(max_train_f1, label='train')
    plt.plot(max_valid_f1, label='validation')
    plt.title('Max F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot_background_grid(train_f1, valid_f1, save_path):
    plt.figure()
    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')
    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # Customize the grid and background
    ax = plt.gca()
    ax.set_facecolor('#eae6f0')
    ax.grid(True, which='both', color='white', linestyle='-', linewidth=1.0, alpha=0.9)  # Light grid lines with low alpha for near invisibility

    plt.savefig(f'{save_path}')
    plt.close()

def draw_f1_plot(train_f1, valid_f1, save_path):
    import matplotlib.pyplot as plt

    plt.figure()

    plt.plot(train_f1, label='train')
    plt.plot(valid_f1, label='validation')

    plt.title('F1-score over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend()

    # -----------------------------
    # Clean styling (NO GRID + NO COLOR BACKGROUND)
    # -----------------------------
    ax = plt.gca()
    ax.set_facecolor('white')   # plain background
    ax.grid(False)              # remove grid completely

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    hyperparams = {
        'num_epochs': 100,
        'out_feats': 128,
        'num_layers': 2,
        'lr': 0.001,
        'batch_size': 1,
        'device': torch.device('cpu')
        # 'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    train(hyperparams=hyperparams)

