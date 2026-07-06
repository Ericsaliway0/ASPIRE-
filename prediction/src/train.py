
import networkx as nx
import leidenalg as la
import igraph as ig
import plotly.graph_objects as go
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.gridspec import GridSpec
from collections import defaultdict
import statistics
import dgl
import torch
import torch.nn as nn
import numpy as np
import os
import time
import psutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import scipy.stats
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib.lines import Line2D
from scipy.stats import ttest_ind
from torch_geometric.nn import GCNConv
import igraph as ig
import leidenalg
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import SpectralBiclustering
from .models import ACGNN, HGDC, EMOGI, MTGCN, GCN, GAT, GraphSAGE, GIN, ChebNet, FocalLoss
from src.utils import (choose_model, plot_roc_curve, plot_pr_curve, load_graph_data, 
                       load_oncokb_genes, plot_and_analyze, save_and_plot_results)

import os
import numpy as np
import umap
from matplotlib.colors import LogNorm, to_rgb
from matplotlib.ticker import LogFormatter
from matplotlib.gridspec import GridSpec

CLUSTER_COLORS = {
    0:  '#0077B6',  1:  '#0000FF',  2:  '#00B4D8',  3:  '#48EAC4',
    4:  '#F1C0E8',  5:  '#B9FBC0',  6:  '#32CD32',  7:  '#bee1e6',
    8:  '#8A2BE2',  9:  '#E377C2', 10: '#8EECF5', 11: '#A3C4F3',
    12: '#FFB347', 13: '#FFD700', 14: '#FF69B4', 15: '#CD5C5C',
    16: '#7FFFD4', 17: '#FF7F50', 18: '#C71585', 19: '#20B2AA',
    20: '#48CAE4', 21: '#90DBF4', 22: '#0077B6', 23: '#00B4D8',
    24: '#6A5ACD', 25: '#66CDAA', 26: '#FF8C00', 27: '#9370DB'
}

import os
import time
import psutil
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F
import dgl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.cluster import SpectralBiclustering


def train(args):
    epoch_times, cpu_usages, gpu_usages = [], [], []

    data_path = os.path.join('data/processed/', f'{args.net_type}_gene_pathway_graph_random_5000_1433.json')
    data_path = os.path.join('process/data/processed/', f'{args.net_type}_gene_pathway_graph_random_5000_2866.json')
    # data_path = os.path.join('process/data/processed/', f'{args.net_type}_gene_pathway_graph_tp53_balanced_2g.json')

    # data_path = os.path.join('../ACGNN_data/data/multiomics_meth/', f'{args.net_type}_omics_ppi_embeddings_graph_2048.json')
    nodes, edges, embeddings, labels = load_graph_data(data_path)

    # Ensure nodes is a 1D array
    if isinstance(nodes, dict):
        nodes = list(nodes.keys())
    elif not isinstance(nodes, (list, np.ndarray)):
        nodes = list(nodes)
    nodes = np.array(nodes)

    graph = dgl.graph(edges)
    graph.ndata['feat'] = embeddings
    graph.ndata['label'] = labels
    graph.ndata['train_mask'] = labels != -1
    graph.ndata['test_mask'] = torch.ones_like(labels, dtype=torch.bool)
    graph = dgl.add_self_loop(graph)

    in_feats = embeddings.shape[1]
    hidden_feats = args.hidden_feats
    out_feats = 1  # Binary classification

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph = graph.to(device)
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].to(device).float()
    train_mask = graph.ndata['train_mask'].to(device)
    test_mask = graph.ndata['test_mask'].to(device)

    model = choose_model(args.model_type, in_feats, hidden_feats, out_feats).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = FocalLoss(alpha=0.25, gamma=2)

    output_dir = 'results/gene_prediction/'
    os.makedirs(output_dir, exist_ok=True)
    
    model_csv_path = os.path.join(output_dir, f'{args.model_type}_model_structure.csv')

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in tqdm(range(args.num_epochs), desc="Training Progress"):
        start_time = time.time()
        model.train()
        logits = model(graph, features).squeeze()
        loss = loss_fn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time
        tqdm.write(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s")

    model.eval()
    with torch.no_grad():
        logits = model(graph, features).squeeze()
        scores = torch.sigmoid(logits).cpu().numpy()

    print("Computing gene saliency...")
    gene_saliency = compute_gene_saliency(model, graph, features, mask=test_mask)
    # ensure numpy array
    if torch.is_tensor(gene_saliency):
        gene_saliency = gene_saliency.cpu().numpy()

    # Apply test_mask to saliency
    mask_np = test_mask.cpu().numpy()
    if len(nodes) != len(gene_saliency):
        nodes = np.array(nodes[:len(gene_saliency)])  # trim if mismatch
    saliency_nodes = nodes[mask_np]
    gene_saliency_masked = gene_saliency[mask_np]

    # -----------------------------
    # 5. Top-K Gene Selection
    # -----------------------------
    TOP_K = getattr(args, "top_k", 2000)
    saliency_df = pd.DataFrame({
        "Gene": saliency_nodes,
        "Saliency": gene_saliency_masked
    }).sort_values("Saliency", ascending=False).head(TOP_K)

    # -----------------------------
    # 6. Leiden Clustering on Top-K Genes
    # -----------------------------
    saliency_df = leiden_cluster_gene_saliency(
        saliency_df,
        nodes,
        similarity_threshold=0.25,
        resolution=0.5
    )
    saliency_df.to_csv(
        os.path.join(output_dir, f"leiden_gene_clusters_topK_{TOP_K}.csv"),
        index=False
    )

    # Rank nodes by scores for only non-labeled nodes

    if isinstance(nodes, dict):
        node_names = list(nodes.keys())
    elif isinstance(nodes, (list, np.ndarray)):
        node_names = list(nodes)
    else:
        raise TypeError(f"Unsupported type for nodes: {type(nodes)}")

    # Map clusters back to node order
    cluster_map = dict(zip(saliency_df["Gene"], saliency_df["LeidenCluster"]))

    cluster_array = np.array([
        cluster_map.get(g, -1) for g in node_names
    ])
    
    
    plot_gene_saliency_umap(
    embeddings=embeddings,
        gene_clusters=cluster_array,
        saliency=gene_saliency_masked,
        output_dir=output_dir,
        filename=f"{args.model_type}_gene_saliency_umap_epo{args.num_epochs}.png",
        title="Gene Saliency UMAP (Leiden)",
    )

    plot_gene_umap(
        embeddings=embeddings,
        gene_labels=labels,
        # gene_clusters=cluster_array,
        output_path=output_dir
    )
    
    # -----------------------------
    # Build cluster → color map
    # -----------------------------
    unique_clusters = np.sort(np.unique(cluster_array))

    gene_cmap = {c: CLUSTER_COLORS[c] for c in unique_clusters if c in CLUSTER_COLORS}
    gene_cmap[-1] = "#B0B0B0"   # noise / unassigned

    # -----------------------------
    # Run UMAP on embeddings
    # -----------------------------
    gene_umap_file = os.path.join(
        output_dir,
        f"gene_umap_leiden"
        f"_epo{args.num_epochs}.png"
    )

    # plot_umap_with_cluster_colors(
    #     X=embeddings,                 # <-- numeric matrix
    #     cluster_ids=cluster_array,    # <-- per-gene cluster id
    #     cluster_colors=gene_cmap,     # <-- mapping dict
    #     output_path=output_dir,
    #     filename=os.path.basename(gene_umap_file),
    #     title="Gene Saliency UMAP (Leiden)",
    # )

    X_saliency = embeddings * gene_saliency[:, None]

    plot_umap_with_cluster_colors(
        X=X_saliency,
        cluster_ids=cluster_array,
        cluster_colors=gene_cmap,
        output_path=output_dir,
        filename=f"{args.model_type}_gene_saliency_umap_epo{args.num_epochs}.png",
        title="Gene Saliency UMAP (Leiden clusters)",
    )
    
    # -----------------------------
    # 7. Gene–Cancer Multi-Omics Matrix
    # -----------------------------
    omics_df = pd.read_csv("process/data/processed/multiomics_7cancers_normalized_2866.csv", index_col=0)
    # omics_df = pd.read_csv("data/multiomics_features.csv", index_col=0)
    cancers = sorted({c.split(": ")[1] for c in omics_df.columns})
    gene_cancer_matrix = pd.DataFrame(index=omics_df.index, columns=cancers, dtype=float)

    for cancer in cancers:
        cols = [c for c in omics_df.columns if c.endswith(cancer)]
        gene_cancer_matrix[cancer] = omics_df[cols].mean(axis=1)

    # Restrict to top-K genes
    top_genes = saliency_df["Gene"].values
    gene_cancer_matrix = gene_cancer_matrix.loc[gene_cancer_matrix.index.intersection(top_genes)]

    # Weight by saliency
    saliency_map = saliency_df.set_index("Gene")["Saliency"]
    gene_cancer_weighted = gene_cancer_matrix.mul(saliency_map, axis=0)


    # =====================================================
    #  OMICS-SPECIFIC BIPARTITE LEIDEN HEATMAPS
    # =====================================================

    omics_matrices = split_gene_cancer_by_omics(gene_cancer_weighted)

    for omics, mat in omics_matrices.items():
        print(f"Processing {omics} bipartite graph: {mat.shape}")

        reordered, gene_clust, cancer_clust = leiden_bipartite_clustering(
            mat,
            resolution=0.3
        )

        # Save reordered matrix
        reordered.to_csv(
            os.path.join(output_dir, f"{omics}_gene_cancer_matrix_leiden.csv")
        )

        # Plot heatmap
        plot_omics_bipartite_heatmap(
            reordered,
            gene_clusters=gene_clust,
            cancer_clusters=cancer_clust,
            omics=omics,
            output_dir=output_dir,
            vmax_percentile=99,
            figsize=(14, 10),
        )

    # -----------------------------
    # 8. Spectral Biclustering
    # -----------------------------
    bicluster = SpectralBiclustering(n_clusters=(6, 4), method="log", random_state=0)

    gene_cancer_weighted = gene_cancer_weighted.replace([np.inf, -np.inf], np.nan)
    gene_cancer_weighted = gene_cancer_weighted.fillna(0)

    print("NaN count:", gene_cancer_weighted.isna().sum().sum())

    common_genes = gene_cancer_matrix.index.intersection(saliency_map.index)

    gene_cancer_matrix = gene_cancer_matrix.loc[common_genes]
    saliency_map = saliency_map.loc[common_genes]

    gene_cancer_weighted = gene_cancer_matrix.mul(saliency_map, axis=0)

    bicluster.fit(gene_cancer_weighted.values)

    gene_order = np.argsort(bicluster.row_labels_)
    cancer_order = np.argsort(bicluster.column_labels_)

    gene_cancer_biclustered = gene_cancer_weighted.iloc[gene_order, cancer_order]


    # -----------------------------
    # Create bipartite graph
    # -----------------------------
    B = nx.Graph()
    B.add_nodes_from(gene_cancer_weighted.index, bipartite=0, type='gene')
    B.add_nodes_from(gene_cancer_weighted.columns, bipartite=1, type='cancer')

    for gene in gene_cancer_weighted.index:
        for cancer in gene_cancer_weighted.columns:
            value = gene_cancer_weighted.loc[gene, cancer]
            # Ensure numeric, skip NaN or zero
            if np.isnan(value):
                continue
            w = float(value)
            if w > 0:
                B.add_edge(gene, cancer, weight=w)

    # -----------------------------
    # Convert to igraph
    # -----------------------------
    # Extract edge list with weight
    edges = [(u, v, float(d['weight'])) for u, v, d in B.edges(data=True)]
    ig_B = ig.Graph.TupleList(edges, edge_attrs=['weight'], directed=False)

    # -----------------------------
    # Run Leiden algorithm
    # -----------------------------
    partition = la.find_partition(
        ig_B,
        la.RBConfigurationVertexPartition,
        weights='weight',            # must be float
        resolution_parameter=1.0
    )

    # -----------------------------
    # Extract clusters
    # -----------------------------
    cluster_dict = {node: cid for node, cid in zip(ig_B.vs['name'], partition.membership)}
    gene_clusters = {n: c for n, c in cluster_dict.items() if n in gene_cancer_weighted.index}
    cancer_clusters = {n: c for n, c in cluster_dict.items() if n in gene_cancer_weighted.columns}

    # -----------------------------
    # Reorder for heatmap
    # -----------------------------
    gene_order = sorted(gene_clusters.keys(), key=lambda x: gene_clusters[x])
    cancer_order = sorted(cancer_clusters.keys(), key=lambda x: cancer_clusters[x])

    gene_cancer_biclustered = gene_cancer_weighted.loc[gene_order, cancer_order]

    plot_gene_cancer_biclustering_heatmap(
        gene_cancer_biclustered,
        output_dir=output_dir
    )
    
    # -----------------------------
    # Align gene Leiden labels to heatmap rows
    # -----------------------------
    gene_leiden_labels = [
        gene_clusters[g] for g in gene_cancer_biclustered.index
    ]

    plot_gene_cancer_biclustering_heatmap_with_gene_clusters(
        gene_cancer_biclustered,
        gene_clusters=gene_leiden_labels,  # same order as rows
        output_dir=output_dir
    )
    
    # -----------------------------
    # Leiden Gene → Cancer Sankey
    # -----------------------------
    plot_leiden_cluster_cancer_sankey(
        gene_cancer_biclustered,
        gene_clusters=gene_leiden_labels,
        cancer_clusters=[cancer_clusters[c] for c in gene_cancer_biclustered.columns],
        output_dir=output_dir,
        filename="gene_cancer_leiden_sankey.html"
    )
    

    # # G = nx.karate_club_graph()
    # G = build_nx_graph_from_loaded_data(nodes, edges)

    

    # cluster_labels = {
    #     node: 0 if node < 10 else 1
    #     for node in G.nodes()
    # }

    # # Enrichment label per cluster
    # enrichment_labels = {
    #     0: "Synaptic signaling",
    #     1: "Immune response"
    # }

    # # Optional: fix layout for reproducibility
    # pos = nx.spring_layout(G, seed=42)


    # plot_graph_with_cluster_enrichment_labels(
    #     G=G,
    #     cluster_labels=gene_clusters_dict,
    #     enrichment_labels=cluster_enrichment_dict,
    #     output_dir="results/figures",
    #     filename="gene_graph_with_enrichment.png",
    # )
    # plot_leiden_cluster_cancer_sankey_with_enrichment(
    #     gene_cancer_biclustered=gene_cancer_biclustered,
    #     gene_clusters=gene_clusters,
    #     cancer_clusters=cancer_clusters,
    #     output_dir=output_dir,
    #     filename="fig5_leiden_gene_cancer_sankey.html",
    #     min_weight=0.01,        # remove weak flows
    #     top_k_labels=4          # show more enriched genes/cancers
    # )
    # print(type(gene_clusters), gene_clusters)
    
    # gene_clusters_vec = np.array(
    #     [gene_clusters[g] for g in gene_cancer_matrix.index],
    #     dtype=int
    # )

    # plot_leiden_cluster_cancer_sankey(
    #     gene_cancer_matrix=gene_cancer_biclustered.values,
    #     gene_clusters=gene_clusters_vec,
    #     cancer_names=gene_cancer_biclustered.columns.tolist(),
    #     output_dir=output_dir,
    #     agg="sum",          # or "mean" / "count"
    #     min_flow=1e-3,
    #     title="Leiden Gene Cluster × Cancer Associations"
    # )


    plot_leiden_cancer_gene_heatmap(
        gene_cancer_weighted=gene_cancer_weighted,
        gene_names=list(gene_cancer_weighted.index),
        cancer_names=list(gene_cancer_weighted.columns),
        gene_clusters=gene_clusters,
        cancer_clusters=cancer_clusters,
        output_path=output_dir,
        args=args,
        vmax_percentile=99,  # optional, can adjust
        figsize=(16, 12),    # optional
    )


    plot_leiden_gene_cancer_heatmap(
        gene_cancer_weighted=gene_cancer_weighted,
        gene_names=list(gene_cancer_weighted.index),
        cancer_names=list(gene_cancer_weighted.columns),
        gene_clusters=gene_clusters,
        cancer_clusters=cancer_clusters,
        output_path=output_dir,
        args=args,
        vmax_percentile=99,  # optional, can adjust
        figsize=(16, 12),    # optional
    )



    # # Convert epoch_times to floats during summation
    # total_time = sum(epoch_times)  # Total time in seconds
    # average_time_per_epoch = total_time / args.num_epochs  # Average time per epoch
    # # Calculate average memory usage for CPU and GPU
    # average_cpu_usage = sum(cpu_usages) / args.num_epochs  # CPU usage in MB
    # average_gpu_usage = sum(gpu_usages) / args.num_epochs  # GPU usage in MB
    
    save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats)
    csv_file_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )
    label_scores = save_predicted_scores(scores, labels, nodes, args, csv_file_path)
    save_average_scores(label_scores, args)
    plot_average_scores(label_scores, args)
    plot_score_distributions(label_scores, args)
    save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args)
    # After calculating total_time, average_time_per_epoch, etc.
    # save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir)


    ##################################################################
    # # Rank nodes by scores for only non-labeled nodes

    # if isinstance(nodes, dict):
    #     node_names = list(nodes.keys())
    # elif isinstance(nodes, (list, np.ndarray)):
    #     node_names = list(nodes)
    # else:
    #     raise TypeError(f"Unsupported type for nodes: {type(nodes)}")



    non_labeled_nodes = [i for i, label in enumerate(labels) if label == -1]  # Indices of non-labeled nodes
    non_labeled_scores = [(node_names[i], scores[i]) for i in non_labeled_nodes]
    
    ranking = sorted(non_labeled_scores, key=lambda x: x[1], reverse=True)


    process_predictions(ranking, args, "data/drivers/796_drivers.txt", "data/drivers/oncokb_1172.txt", "data/drivers/depmap_35331.txt", "data/drivers/ncg_8886.txt", "data/drivers/intogen_23444.txt", node_names, non_labeled_nodes)

    # Load driver and reference gene sets


    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]
    

    # Save both above and below threshold scores, sorted by scores in descending order
    predicted_driver_nodes_above_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] >= args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )
    predicted_driver_nodes_below_threshold = sorted(
        [(node_names[i], scores[i]) for i in non_labeled_nodes if scores[i] < args.score_threshold],
        key=lambda x: x[1],
        reverse=True
    )



    # Get the ground truth driver gene indices and names
    ground_truth_driver_indices = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_driver_names = {node_names[i] for i in ground_truth_driver_indices}

    # Save predictions (above and below threshold) to CSV
    output_file_above = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_above_epo{args.num_epochs}.csv'
    )
    output_file_below = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_predicted_driver_genes_below_epo{args.num_epochs}.csv'
    )

    with open(output_file_above, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_above_threshold)

    with open(output_file_below, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Gene Name', 'Score'])  # Header row
        csvwriter.writerows(predicted_driver_nodes_below_threshold)

    print(f"Predicted driver genes (above threshold) saved to {output_file_above}")
    print(f"Predicted driver genes (below threshold) saved to {output_file_below}")

    # Calculate degrees for nodes above and below the threshold (connecting only to label 1 nodes)
    degree_counts_above = defaultdict(int)
    degree_counts_below = defaultdict(int)

    # for src, dst in edges:
    #     src_name = node_names[src]
    #     dst_name = node_names[dst]

    #     # Count only connections to ground truth driver genes (label 1 nodes)
    #     if dst_name in ground_truth_driver_names:
    #         if src_name in [gene for gene, _ in predicted_driver_nodes_above_threshold]:
    #             degree_counts_above[src_name] += 1
    #         elif src_name in [gene for gene, _ in predicted_driver_nodes_below_threshold]:
    #             degree_counts_below[src_name] += 1

    # # Sort degrees by degree count in descending order
    # sorted_degree_counts_above = sorted(degree_counts_above.items(), key=lambda x: x[1], reverse=True)
    # sorted_degree_counts_below = sorted(degree_counts_below.items(), key=lambda x: x[1], reverse=True)

    # # Save degrees of predicted driver genes connecting to ground truth driver genes (above threshold)
    # degree_output_file_above = os.path.join(
    #     'results/gene_prediction/',
    #     f'{args.model_type}_{args.net_type}_predicted_driver_gene_degrees_above_epo{args.num_epochs}.csv'
    # )
    # with open(degree_output_file_above, 'w', newline='') as csvfile:
    #     csvwriter = csv.writer(csvfile)
    #     csvwriter.writerow(['Predicted Driver Gene', 'Degree'])  # Header row
    #     csvwriter.writerows(sorted_degree_counts_above)
    #     ##csvwriter.writerow(['Average Degree', average_degree_above])  # Save average degree

    # print(f"Degrees of predicted driver genes (above threshold) saved to {degree_output_file_above}")



    # # Prepare DataFrame for nodes with degrees
    # nodes_with_degrees = []

    # # Above threshold
    # for gene, degree in degree_counts_above.items():
    #     nodes_with_degrees.append({'Gene_Set': 'Above Threshold', 'Degree': degree})

    # # Below threshold
    # for gene, degree in degree_counts_below.items():
    #     nodes_with_degrees.append({'Gene_Set': 'Below Threshold', 'Degree': degree})

    # # Convert to DataFrame
    # nodes_with_degrees = pd.DataFrame(nodes_with_degrees)

    # ##print(f"sorted_degree_counts_below: {sorted_degree_counts_below}")
    # sorted_degree_counts_above_value = [value for _, value in sorted_degree_counts_above if value <= 20]
    # sorted_degree_counts_below_value = [value for _, value in sorted_degree_counts_below if value <= 20]
    
    # output_above_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_above_file_epo{args.num_epochs}.csv')
    # output_below_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_output_below_file_epo{args.num_epochs}.csv')

    # calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args)

    # plot_degree_distributions(sorted_degree_counts_above_value, sorted_degree_counts_below_value, args, output_dir)

    # generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args)

    # plot_model_performance(args)


def plot_gene_saliency_umap(
    embeddings,
    gene_clusters,
    saliency=None,
    output_dir="results/",
    filename="gene_saliency_umap.png",
    title="Gene Saliency UMAP",
    random_state=42,
):
    """
    UMAP of gene embeddings colored by Leiden clusters.
    Optionally scales point size by saliency.
    """

    import umap
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # UMAP projection
    # -------------------------
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.3,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )

    X_umap = reducer.fit_transform(embeddings)

    # -------------------------
    # Prepare colors
    # -------------------------
    cluster_ids = np.asarray(gene_clusters)

    unique_clusters = np.unique(cluster_ids)
    palette = list(CLUSTER_COLORS.values())

    cluster_to_color = {
        c: palette[i % len(palette)]
        for i, c in enumerate(unique_clusters)
    }

    colors = [cluster_to_color[c] for c in cluster_ids]


    # -------------------------
    # Point sizes (saliency-aware)
    # -------------------------
    if saliency is not None:
        saliency = np.asarray(saliency)
        s = 20 + 80 * (saliency - saliency.min()) / (saliency.ptp() + 1e-9)
    else:
        s = 40

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(14, 10))
    plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=colors,
        s=s,
        alpha=0.85,
        linewidths=0,
    )

    plt.title(title, fontsize=22)
    plt.xlabel("UMAP-1", fontsize=16)
    plt.ylabel("UMAP-2", fontsize=16)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"✔ Gene saliency UMAP saved to {save_path}")

def plot_umap_with_cluster_colors_no_clusters(
    X,
    cluster_ids,
    cluster_colors,
    output_path,
    filename,
    title,
    n_neighbors=15,
    min_dist=0.2,
    metric="cosine",
    point_size=30,
    alpha=0.85,
    random_state=42,
):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import umap

    # -----------------------
    # Safety conversions
    # -----------------------
    X = np.asarray(X, dtype=float)
    cluster_ids = np.asarray(cluster_ids)

    # Replace NaNs/Infs
    X = np.nan_to_num(X)

    # -----------------------
    # Row-normalize embeddings
    # -----------------------
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # -----------------------
    # UMAP
    # -----------------------
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    embedding = reducer.fit_transform(X)

    # -----------------------
    # Colors (safe lookup)
    # -----------------------
    colors = [
        cluster_colors.get(int(c), "#B0B0B0")   # gray fallback
        for c in cluster_ids
    ]

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(7, 6))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    plt.xlabel("UMAP-1", fontsize=14)
    plt.ylabel("UMAP-2", fontsize=14)
    plt.title(title, fontsize=15)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, filename),
        dpi=300,
    )
    plt.show()

    return embedding

def plot_umap_with_cluster_colors(
    X,
    cluster_ids,
    cluster_colors,
    output_path,
    filename,
    title,
    n_neighbors=15,
    min_dist=0.2,
    metric="cosine",
    point_size=30,
    alpha=0.85,
    random_state=42,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import umap
    from matplotlib.lines import Line2D

    # -----------------------
    # Safety
    # -----------------------
    X = np.asarray(X, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    X = np.nan_to_num(X)

    # Row normalize
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    # -----------------------
    # UMAP
    # -----------------------
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(X)

    # -----------------------
    # Color mapping
    # -----------------------
    colors = [
        cluster_colors.get(int(c), "#B0B0B0")
        for c in cluster_ids
    ]

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(7.5, 6.5))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )

    plt.xlabel("UMAP-1", fontsize=16)
    plt.ylabel("UMAP-2", fontsize=16)
    plt.title(title, fontsize=17)

    # -----------------------
    # Legend (cluster-wise)
    # -----------------------
    legend_elements = []
    for cid in sorted(np.unique(cluster_ids)):
        n = np.sum(cluster_ids == cid)
        label = f"Cluster {cid} (n={n})" if cid != -1 else f"Unassigned (n={n})"

        legend_elements.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                label=label,
                markerfacecolor=cluster_colors.get(cid, "#B0B0B0"),
                markeredgecolor="none",
                markersize=9,
            )
        )

    plt.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, filename), dpi=300)
    plt.show()

    return embedding


def plot_gene_umap_pas(
    embeddings,
    node_names,
    labels=None,
    clusters=None,
    output_dir="results/gene_prediction/",
    filename="gene_umap.png",
    n_neighbors=15,
    min_dist=0.1,
    random_state=42,
    figsize=(8, 7),
):
    """
    Plot UMAP projection of gene node embeddings.

    Parameters
    ----------
    embeddings : np.ndarray (N x D)
    node_names : array-like
    labels : optional class labels
    clusters : optional Leiden cluster ids
    """


    os.makedirs(output_dir, exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="cosine",
        random_state=random_state,
    )

    embedding_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=figsize)

    # --------------------------------
    # Choose coloring priority
    # --------------------------------
    if clusters is not None:
        color_values = clusters
        title = "UMAP of Gene Embeddings (Leiden Clusters)"
    elif labels is not None:
        color_values = labels
        title = "UMAP of Gene Embeddings (Labels)"
    else:
        color_values = None
        title = "UMAP of Gene Embeddings"

    if color_values is not None:
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=color_values,
            s=8,
            alpha=0.8,
        )
        plt.colorbar(scatter)
    else:
        plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            s=8,
            alpha=0.8,
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"UMAP saved → {save_path}")


def plot_gene_umap_ori(embeddings, gene_labels, output_path):
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )

    Z = reducer.fit_transform(embeddings)

    # ----------------------------------
    # Convert tensor labels → int
    # ----------------------------------
    if torch.is_tensor(gene_labels):
        gene_labels = gene_labels.cpu().numpy()

    gene_labels = np.array(gene_labels).astype(int)

    # ----------------------------------
    # Map cluster → color
    # ----------------------------------
    colors = [
        CLUSTER_COLORS.get(int(l), "#BBBBBB")
        for l in gene_labels
    ]

    # ----------------------------------
    # Large-font paper-ready figure
    # ----------------------------------
    plt.rcParams.update({
        "font.size": 18,
        "legend.fontsize": 16
    })

    plt.figure(figsize=(10, 8))

    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=colors,
        s=20,
        alpha=0.85
    )

    # Legend
    unique_clusters = sorted(set(gene_labels))
    for cid in unique_clusters:
        plt.scatter(
            [],
            [],
            c=CLUSTER_COLORS.get(int(cid), "#BBBBBB"),
            label=f"C{cid}",
            s=60
        )

    plt.legend(
        frameon=False,
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/umap_genes_leiden_clusters.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()


def plot_gene_umap(embeddings, gene_labels, output_path):
    import umap
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.lines import Line2D

    # ----------------------------------
    # UMAP
    # ----------------------------------
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42
    )
    Z = reducer.fit_transform(embeddings)

    # ----------------------------------
    # Convert tensor labels → int
    # ----------------------------------
    if torch.is_tensor(gene_labels):
        gene_labels = gene_labels.cpu().numpy()
    gene_labels = np.array(gene_labels).astype(int)

    # ----------------------------------
    # Semantic label mapping
    # ----------------------------------
    LABEL_MAP = {
        1: "Driver genes",
        0: "Non-Driver genes",
       -1: "Others"
    }

    # ----------------------------------
    # Map labels → colors
    # ----------------------------------
    colors = [
        CLUSTER_COLORS.get(int(l), "#BBBBBB")
        for l in gene_labels
    ]

    # ----------------------------------
    # Large-font paper-ready figure
    # ----------------------------------
    plt.rcParams.update({
        "font.size": 20,
        "legend.fontsize": 22
    })

    plt.figure(figsize=(10, 8))

    plt.scatter(
        Z[:, 0],
        Z[:, 1],
        c=colors,
        s=20,
        alpha=0.85,
        linewidths=0
    )

    # ----------------------------------
    # Legend INSIDE plot (semantic)
    # ----------------------------------
    legend_elements = []
    for k, label in LABEL_MAP.items():
        if k in gene_labels:
            legend_elements.append(
                Line2D(
                    [0], [0],
                    marker='o',
                    linestyle='None',
                    label=label,
                    markerfacecolor=CLUSTER_COLORS.get(k, "#BBBBBB"),
                    markeredgecolor='none',
                    markersize=12
                )
            )

    plt.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        edgecolor="none",
        borderpad=0.6,
        handletextpad=0.6,
        labelspacing=0.4
    )

    # ----------------------------------
    # Clean axes
    # ----------------------------------
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(
        f"{output_path}/umap_genes_driver_status.png",
        dpi=600,
        bbox_inches="tight"
    )
    plt.show()

# -------------------------------------------------
# PAPER-READY GENE UMAP WITH FIXED CLUSTER COLORS
# -------------------------------------------------
def plot_gene_umap_one_cluster(
    embeddings,
    node_names,
    clusters,
    output_dir,
    filename="gene_umap.png",
):

    print("Running UMAP projection...")

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    umap_coords = reducer.fit_transform(embeddings)

    # -------------------------------------------------
    # LARGE FONT SETTINGS (Pattern/Bioinformatics ready)
    # -------------------------------------------------
    plt.rcParams.update({
        "font.size": 20,
        "axes.labelsize": 24,
        "axes.titlesize": 26,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18
    })

    plt.figure(figsize=(12, 10))

    unique_clusters = sorted(set(clusters))

    for cid in unique_clusters:

        mask = clusters == cid

        # default gray if cluster not defined
        color = CLUSTER_COLORS.get(cid, "#BBBBBB")

        plt.scatter(
            umap_coords[mask, 0],
            umap_coords[mask, 1],
            s=40,
            color=color,
            alpha=0.85,
            label=f"Cluster {cid}" if cid != -1 else "Unclustered"
        )

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")

    plt.legend(
        markerscale=1.5,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved gene UMAP → {save_path}")

def build_nx_graph_from_loaded_data(nodes, edges):
    """
    nodes: array-like of gene names, length N
    edges: list of (src, dst) node indices
    """
    G = nx.Graph()

    # Add nodes with gene names
    for i, gene in enumerate(nodes):
        G.add_node(gene)

    # Add edges using node indices
    for u, v in edges:
        if u < len(nodes) and v < len(nodes):
            G.add_edge(nodes[u], nodes[v])

    return G

def hex_to_rgba(hex_color, alpha=0.85):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_leiden_cluster_cancer_sankey_(
    gene_cancer_matrix,
    gene_clusters,
    cancer_names,
    output_dir,
    filename="leiden_cluster_cancer_sankey.html",
    agg="sum",
    min_flow=0.0,
    title="Leiden Gene Cluster × Cancer Sankey",
):


    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Sanitize inputs
    # -------------------------
    gene_clusters = np.asarray(gene_clusters)

    # Force cluster IDs to plain ints
    gene_clusters = np.array([int(c) for c in gene_clusters])

    # Ensure cancer names are strings
    cancer_names = [str(c) for c in cancer_names]

    unique_clusters = np.unique(gene_clusters)

    # -------------------------
    # Aggregate cluster → cancer
    # -------------------------
    rows = []
    for c in unique_clusters:
        mask = gene_clusters == c

        if isinstance(gene_cancer_matrix, pd.DataFrame):
            submat = gene_cancer_matrix.loc[mask, :].values
        else:
            submat = gene_cancer_matrix[mask, :]

        if agg == "sum":
            values = submat.sum(axis=0)
        elif agg == "mean":
            values = submat.mean(axis=0)
        elif agg == "count":
            values = (submat > 0).sum(axis=0)
        else:
            raise ValueError("agg must be one of {'sum','mean','count'}")

        values = np.asarray(values).ravel()

        for cancer, v in zip(cancer_names, values):
            v = float(v)
            if v > min_flow:
                rows.append(
                    {
                        "cluster_id": int(c),
                        "cluster": f"Cluster {int(c)}",
                        "cancer": cancer,
                        "value": v,
                    }
                )

    sankey_df = pd.DataFrame(rows)

    if sankey_df.empty:
        print("No flows above min_flow; Sankey not generated.")
        return

    # -------------------------
    # Node indexing
    # -------------------------
    cluster_nodes = (
        sankey_df[["cluster_id", "cluster"]]
        .drop_duplicates()
        .sort_values("cluster_id")
    )

    cancer_nodes = (
        sankey_df[["cancer"]]
        .drop_duplicates()
        .sort_values("cancer")
    )

    nodes = (
        cluster_nodes["cluster"].tolist()
        + cancer_nodes["cancer"].tolist()
    )

    node_index = {n: i for i, n in enumerate(nodes)}

    sources = sankey_df["cluster"].map(node_index)
    targets = sankey_df["cancer"].map(node_index)
    values = sankey_df["value"]

    # -------------------------
    # Colors
    # -------------------------
    cluster_colors = [
        hex_to_rgba(
            CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
            alpha=0.85,
        )
        for cid in cluster_nodes["cluster_id"]
    ]

    cancer_colors = ["rgba(180,180,180,0.7)"] * len(cancer_nodes)

    # -------------------------
    # Sankey plot
    # -------------------------
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15,
                thickness=18,
                label=nodes,
                color=cluster_colors + cancer_colors,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            ),
        )
    )

    fig.update_layout(
        title_text=title,
        font_size=12,
        width=1000,
        height=600,
    )

    save_path = os.path.join(output_dir, filename)
    fig.write_html(save_path)
    fig.show()

    print(f"Sankey saved to {save_path}")



def plot_leiden_cluster_cancer_sankey_pas(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_sankey.html",
    min_weight=0.0,
):
    """
    Cluster-first Gene → Cancer Sankey using Leiden bipartite clusters.
    """

    os.makedirs(output_dir, exist_ok=True)

    genes = gene_cancer_biclustered.index.tolist()
    cancers = gene_cancer_biclustered.columns.tolist()

    # -----------------------------
    # Aggregate flow by cluster
    # -----------------------------
    flow = defaultdict(float)

    for i, g in enumerate(genes):
        for j, c in enumerate(cancers):
            w = gene_cancer_biclustered.iloc[i, j]
            if np.isnan(w) or w <= min_weight:
                continue
            flow[(gene_clusters[i], cancer_clusters[j])] += float(w)

    gene_cluster_ids = sorted(set(gene_clusters))
    cancer_cluster_ids = sorted(set(cancer_clusters))

    gene_nodes = [f"Gene C{c}" for c in gene_cluster_ids]
    cancer_nodes = [f"Cancer C{c}" for c in cancer_cluster_ids]

    nodes = gene_nodes + cancer_nodes
    node_index = {n: i for i, n in enumerate(nodes)}

    sources, targets, values = [], [], []

    for (gc, cc), v in flow.items():
        sources.append(node_index[f"Gene C{gc}"])
        targets.append(node_index[f"Cancer C{cc}"])
        values.append(v)

    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=15,
                thickness=18,
                label=nodes,
                line=dict(color="black", width=0.4),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            ),
        )
    )

    fig.update_layout(
        title="Leiden Gene → Cancer Cluster Flow",
        font=dict(size=12),
        width=900,
        height=600,
    )

    # save_path = os.path.join(output_dir, filename)
    # fig.write_html(save_path)

    # print(f"Sankey saved to {save_path}")
    save_path = os.path.join(output_dir, filename)

    # Interactive HTML
    fig.write_html(save_path)

    # Static PNG
    png_path = save_path.replace(".html", ".png")
    fig.write_image(png_path, scale=2)

    print(f"Sankey saved to {save_path}")
    print(f"Sankey PNG saved to {png_path}")

import os
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict, Counter


def plot_leiden_cluster_cancer_sankey(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_sankey.html",
    min_weight=0.0,
    normalize=True,
):
    """
    Interpretable Cluster-first Gene → Cancer Sankey using Leiden bipartite clusters.

    Improvements:
    - cluster size annotations
    - normalized flow %
    - biologically intuitive coloring
    - flow-based cluster ordering
    - informative hover tooltips
    """

    os.makedirs(output_dir, exist_ok=True)

    genes = gene_cancer_biclustered.index.tolist()
    cancers = gene_cancer_biclustered.columns.tolist()

    # -----------------------------
    # Aggregate flow by cluster
    # -----------------------------
    flow = defaultdict(float)

    for i, g in enumerate(genes):
        for j, c in enumerate(cancers):
            w = gene_cancer_biclustered.iloc[i, j]
            if np.isnan(w) or w <= min_weight:
                continue
            flow[(gene_clusters[i], cancer_clusters[j])] += float(w)

    # -----------------------------
    # Cluster stats
    # -----------------------------
    gene_cluster_sizes = Counter(gene_clusters)
    cancer_cluster_sizes = Counter(cancer_clusters)

    gene_flow_total = defaultdict(float)
    cancer_flow_total = defaultdict(float)

    for (gc, cc), v in flow.items():
        gene_flow_total[gc] += v
        cancer_flow_total[cc] += v

    # -----------------------------
    # Sort clusters by importance
    # -----------------------------
    gene_cluster_ids = sorted(gene_flow_total, key=gene_flow_total.get, reverse=True)
    cancer_cluster_ids = sorted(cancer_flow_total, key=cancer_flow_total.get, reverse=True)

    # -----------------------------
    # Node labels with meaning
    # -----------------------------
    gene_nodes = [
        f"Gene C{c} (n={gene_cluster_sizes[c]})"
        for c in gene_cluster_ids
    ]

    cancer_nodes = [
        f"Cancer C{c} (n={cancer_cluster_sizes[c]})"
        for c in cancer_cluster_ids
    ]

    nodes = gene_nodes + cancer_nodes
    node_index = {n: i for i, n in enumerate(nodes)}

    # Colors
    gene_colors = ["rgba(70,130,180,0.85)"] * len(gene_nodes)   # steelblue
    cancer_colors = ["rgba(220,20,60,0.85)"] * len(cancer_nodes) # crimson
    node_colors = gene_colors + cancer_colors

    # -----------------------------
    # Build links
    # -----------------------------
    total_flow = sum(flow.values())

    sources, targets, values, hovertext, link_colors = [], [], [], [], []

    for (gc, cc), v in flow.items():

        src = f"Gene C{gc} (n={gene_cluster_sizes[gc]})"
        tgt = f"Cancer C{cc} (n={cancer_cluster_sizes[cc]})"

        val = v / total_flow if normalize else v

        sources.append(node_index[src])
        targets.append(node_index[tgt])
        values.append(val)

        pct = (v / total_flow) * 100 if total_flow > 0 else 0

        hovertext.append(
            f"Gene Cluster {gc} → Cancer Cluster {cc}"
            f"<br>Flow Weight: {v:.3f}"
            f"<br>Contribution: {pct:.2f}%"
        )

        alpha = min(0.9, 0.2 + (v / max(flow.values())))
        link_colors.append(f"rgba(120,120,120,{alpha})")

    # -----------------------------
    # Plot
    # -----------------------------
    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=18,
                thickness=22,
                label=nodes,
                color=node_colors,
                line=dict(color="black", width=0.5),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                customdata=hovertext,
                hovertemplate="%{customdata}<extra></extra>",
            ),
        )
    )

    fig.update_layout(
        title="Interpretable Leiden Gene → Cancer Cluster Flow",
        font=dict(size=18),
        width=1100,
        height=700,
    )

    save_path = os.path.join(output_dir, filename)

    fig.write_html(save_path)

    png_path = save_path.replace(".html", ".png")
    fig.write_image(png_path, scale=2)

    print(f"Sankey saved to {save_path}")
    print(f"Sankey PNG saved to {png_path}")


def plot_leiden_cluster_cancer_sankey_with_enrichment(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_sankey_enriched.html",
    min_weight=0.0,
    top_k_labels=3,
):
    """
    Leiden Gene → Cancer Sankey with cluster enrichment labels attached to nodes.
    """

    os.makedirs(output_dir, exist_ok=True)

    genes = gene_cancer_biclustered.index.tolist()
    cancers = gene_cancer_biclustered.columns.tolist()

    # --------------------------------------------------
    # 1. Aggregate flow by (gene cluster → cancer cluster)
    # --------------------------------------------------
    flow = defaultdict(float)

    for i, g in enumerate(genes):
        for j, c in enumerate(cancers):
            w = gene_cancer_biclustered.iloc[i, j]
            if np.isnan(w) or w <= min_weight:
                continue
            flow[(gene_clusters[i], cancer_clusters[j])] += float(w)

    gene_cluster_ids = sorted(set(gene_clusters))
    cancer_cluster_ids = sorted(set(cancer_clusters))

    # --------------------------------------------------
    # 2. Enrichment labels for clusters
    # --------------------------------------------------
    gene_cluster_members = defaultdict(list)
    cancer_cluster_members = defaultdict(list)

    for i, g in enumerate(genes):
        gene_cluster_members[gene_clusters[i]].append(g)

    for j, c in enumerate(cancers):
        cancer_cluster_members[cancer_clusters[j]].append(c)

    def make_label(prefix, cid, members):
        short = ", ".join(members[:top_k_labels])
        return f"{prefix} {cid}\n({len(members)} nodes)\n{short}"

    gene_nodes = [
        make_label("Gene Cluster", c, gene_cluster_members[c])
        for c in gene_cluster_ids
    ]

    cancer_nodes = [
        make_label("Cancer Cluster", c, cancer_cluster_members[c])
        for c in cancer_cluster_ids
    ]

    nodes = gene_nodes + cancer_nodes
    node_index = {n: i for i, n in enumerate(nodes)}

    # --------------------------------------------------
    # 3. Build Sankey links
    # --------------------------------------------------
    sources, targets, values = [], [], []

    for (gc, cc), v in flow.items():
        sources.append(node_index[gene_nodes[gene_cluster_ids.index(gc)]])
        targets.append(node_index[cancer_nodes[cancer_cluster_ids.index(cc)]])
        values.append(v)

    # --------------------------------------------------
    # 4. Colors
    # --------------------------------------------------
    gene_colors = px.colors.qualitative.Set2
    cancer_colors = px.colors.qualitative.Dark2

    node_colors = (
        [gene_colors[i % len(gene_colors)] for i in range(len(gene_nodes))] +
        [cancer_colors[i % len(cancer_colors)] for i in range(len(cancer_nodes))]
    )

    link_colors = [
        f"rgba{(*px.colors.hex_to_rgb(gene_colors[gene_cluster_ids.index(gc) % len(gene_colors)]), 0.6)}"
        for (gc, cc) in flow.keys()
    ]

    # --------------------------------------------------
    # 5. Plot Sankey
    # --------------------------------------------------
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=30,
                thickness=24,
                label=nodes,
                color=node_colors,
                x=[0.1] * len(gene_nodes) + [0.9] * len(cancer_nodes),
                line=dict(color="black", width=0.6),
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=link_colors,
                hovertemplate=(
                    "%{source.label} → %{target.label}<br>"
                    "Total weight: %{value:.3f}<extra></extra>"
                ),
            ),
        )
    )

    fig.update_layout(
        title=dict(
            text="Leiden Gene → Cancer Cluster Flow with Enrichment Labels",
            x=0.5,
            font=dict(size=22),
        ),
        font=dict(size=15),
        width=1100,
        height=700,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    # --------------------------------------------------
    # 6. Save
    # --------------------------------------------------
    html_path = os.path.join(output_dir, filename)
    png_path = html_path.replace(".html", ".png")

    fig.write_html(html_path)
    fig.write_image(png_path, scale=2)

    print(f"Sankey saved to {html_path}")
    print(f"Sankey PNG saved to {png_path}")

def plot_graph_with_cluster_enrichment_labels_(
    G,
    cluster_labels,
    enrichment_labels,
    output_png,
    pos=None,
    node_size=600,
    font_size=8,
    label_offset=(0.02, 0.02),
    figsize=(10, 8)
):
    """
    Plot a graph with enrichment labels added directly onto nodes.

    Parameters
    ----------
    G : networkx.Graph
        Graph with nodes corresponding to genes.
    cluster_labels : dict
        {node_id: cluster_id}
    enrichment_labels : dict
        {cluster_id: enrichment_label (str)}
        e.g., {0: 'Cell cycle', 1: 'DNA repair'}
    output_png : str
        Path to save the PNG file.
    pos : dict, optional
        Precomputed node positions (e.g., spring_layout).
    node_size : int
        Size of nodes.
    font_size : int
        Font size for node text.
    label_offset : tuple
        Offset for enrichment label placement.
    figsize : tuple
        Figure size.
    """

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=figsize)

    # Color nodes by cluster
    clusters = [cluster_labels[n] for n in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=clusters,
        cmap=plt.cm.tab10,
        node_size=node_size,
        alpha=0.9
    )

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Draw node names
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        font_color="black"
    )

    # Add enrichment labels on nodes
    for node, (x, y) in pos.items():
        cluster_id = cluster_labels[node]
        enrichment = enrichment_labels.get(cluster_id, "")
        if enrichment:
            plt.text(
                x + label_offset[0],
                y + label_offset[1],
                enrichment,
                fontsize=font_size - 1,
                color="darkred",
                alpha=0.85
            )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()



def plot_graph_with_cluster_enrichment_labels(
    G,
    cluster_labels,
    enrichment_labels,
    output_dir,
    filename="graph_cluster_enrichment.png",
    pos=None,
    node_size=600,
    font_size=8,
    label_offset=(0.02, 0.02),
    figsize=(10, 8),
):
    """
    Plot a graph with enrichment labels added directly onto nodes
    and save the figure to the given output directory.
    """

    os.makedirs(output_dir, exist_ok=True)
    output_png = os.path.join(output_dir, filename)

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=figsize)

    # Color nodes by cluster
    clusters = [cluster_labels[n] for n in G.nodes()]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=clusters,
        cmap=plt.cm.tab10,
        node_size=node_size,
        alpha=0.9,
    )

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    # Draw node names
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=font_size,
        font_color="black",
    )

    # Add enrichment labels on nodes
    for node, (x, y) in pos.items():
        cluster_id = cluster_labels[node]
        enrichment = enrichment_labels.get(cluster_id, "")
        if enrichment:
            plt.text(
                x + label_offset[0],
                y + label_offset[1],
                enrichment,
                fontsize=font_size - 1,
                color="darkred",
                alpha=0.85,
            )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Graph saved to {output_png}")
    return output_png

def plot_omics_bipartite_heatmap_ori(
    matrix,
    gene_clusters,
    cancer_clusters,
    omics,
    output_dir,
    vmax_percentile=99,
    figsize=(14, 10),
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    vmax = np.percentile(matrix.values, vmax_percentile)

    plt.figure(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmax=vmax,
        xticklabels=True,
        yticklabels=True,
    )

    plt.title(f"{omics} Gene–Cancer Bipartite Heatmap (Leiden clustered)", fontsize=14)
    plt.xlabel("Cancer Type")
    plt.ylabel("Gene")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{omics}_gene_cancer_bipartite_heatmap.png"),
        dpi=300,
    )
    plt.show()

def plot_omics_bipartite_heatmap_no_separate_line(
    matrix,
    gene_clusters,
    cancer_clusters,
    omics,
    output_dir,
    vmax_percentile=99,
    figsize=(14, 10),
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Plotting {omics}: {matrix.shape}")

    vmax = np.percentile(
        matrix.values,
        vmax_percentile
    )

    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        cmap="viridis",
        vmax=vmax,
        xticklabels=True,
        yticklabels=True,
    )

    plt.title(
        f"{omics} Gene–Cancer Bipartite Heatmap (Leiden clustered)",
        fontsize=14
    )

    plt.xlabel("Cancer Type")
    plt.ylabel("Gene")

    plt.tight_layout()

    save_path = os.path.join(
        output_dir,
        f"{omics}_gene_cancer_bipartite_heatmap.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[SAVED] {save_path}")

    plt.show()
    plt.close()

def plot_omics_bipartite_heatmap_grid(
    matrix,
    gene_clusters,
    cancer_clusters,
    omics,
    output_dir,
    vmax_percentile=99,
    figsize=(14, 10),
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # =====================================================
    # Create output directory
    # =====================================================

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Plotting {omics}: {matrix.shape}")

    # =====================================================
    # Compute vmax for robust visualization
    # =====================================================

    vmax = np.percentile(
        matrix.values,
        vmax_percentile
    )

    # =====================================================
    # Create figure
    # =====================================================

    plt.figure(figsize=figsize)

    # =====================================================
    # Heatmap
    # =====================================================

    ax = sns.heatmap(
        matrix,
        cmap="viridis",
        vmax=vmax,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "Interaction Weight"},
    )

    # =====================================================
    # Titles
    # =====================================================

    ax.set_title(
        # f"{omics} Gene–Cancer Bipartite Heatmap (Leiden clustered)",
        f"{omics}",
        fontsize=20,
        pad=12
    )

    ax.set_xlabel("", fontsize=18)
    ax.set_ylabel("Gene", fontsize=18)

    # =====================================================
    # Draw gene cluster boundaries
    # =====================================================

    prev_cluster = None

    for i, gene in enumerate(matrix.index):

        cluster = gene_clusters.get(gene)

        if prev_cluster is not None and cluster != prev_cluster:

            ax.hlines(
                i,
                *ax.get_xlim(),
                colors="white",
                linewidth=1.5
            )

        prev_cluster = cluster

    # =====================================================
    # Draw cancer cluster boundaries
    # =====================================================

    prev_cluster = None

    for j, cancer in enumerate(matrix.columns):

        cluster = cancer_clusters.get(cancer)

        if prev_cluster is not None and cluster != prev_cluster:

            ax.vlines(
                j,
                *ax.get_ylim(),
                colors="white",
                linewidth=1.5
            )

        prev_cluster = cluster

    # =====================================================
    # Layout
    # =====================================================

    plt.tight_layout()

    # =====================================================
    # Save figure
    # =====================================================

    save_path = os.path.join(
        output_dir,
        f"{omics}_gene_cancer_bipartite_heatmap.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[SAVED] {save_path}")

    # =====================================================
    # Show figure
    # =====================================================

    plt.show()

    plt.close()

def plot_omics_bipartite_heatmap(
    matrix,
    gene_clusters,
    cancer_clusters,
    omics,
    output_dir,
    vmax_percentile=99,
    figsize=(14, 10),
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # =====================================================
    # Create output directory
    # =====================================================

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Plotting {omics}: {matrix.shape}")

    # =====================================================
    # Compute vmax for robust visualization
    # =====================================================

    vmax = np.percentile(
        matrix.values,
        vmax_percentile
    )

    # =====================================================
    # Create figure
    # =====================================================

    plt.figure(figsize=figsize)

    # =====================================================
    # Heatmap
    # =====================================================

    ax = sns.heatmap(
        matrix,
        # cmap="viridis",
        cmap="Blues",
        vmax=vmax,
        xticklabels=True,
        yticklabels=False,
        linewidths=0,
        linecolor=None,
        cbar_kws={
            "label": ""
        },
    )

    # =====================================================
    # Titles
    # =====================================================

    ax.set_title(
        f"{omics}",
        fontsize=32,
        pad=15
    )

    ax.set_xlabel(
        "",
        fontsize=28
    )

    ax.set_ylabel(
        "Gene",
        fontsize=32,
        labelpad=15
    )

    # =====================================================
    # Remove outer border/spines
    # =====================================================

    for spine in ax.spines.values():
        spine.set_visible(False)

    # =====================================================
    # Tick styling
    # =====================================================

    ax.tick_params(
        axis="x",
        labelrotation=90,
        labelsize=28,
        pad=4
    )

    ax.tick_params(
        axis="y",
        labelsize=18
    )

    # Make x tick labels centered/aligned nicely
    plt.setp(
        ax.get_xticklabels(),
        rotation=90,
        ha="center",
        va="top",
        fontsize=28,
        # # fontweight="bold"
    )

    # =====================================================
    # Colorbar styling
    # =====================================================

    cbar = ax.collections[0].colorbar

    cbar.ax.tick_params(
        labelsize=16
    )

    cbar.set_label(
        "",
        fontsize=18
    )

    # =====================================================
    # Layout
    # =====================================================

    plt.tight_layout()

    # =====================================================
    # Save figure
    # =====================================================

    save_path = os.path.join(
        output_dir,
        f"{omics}_gene_cancer_bipartite_heatmap.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"[SAVED] {save_path}")

    # =====================================================
    # Show figure
    # =====================================================

    plt.show()

    plt.close()


def plot_omics_bipartite_heatmap_not_good_contast(
    matrix,
    gene_clusters,
    cancer_clusters,
    omics,
    output_dir,
    vmax_percentile=99,
    figsize=(14, 10),
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import PowerNorm

    # =====================================================
    # Create output directory
    # =====================================================

    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Plotting {omics}: {matrix.shape}")

    # =====================================================
    # Better contrast scaling
    # =====================================================

    values = matrix.values.flatten()

    values = values[np.isfinite(values)]

    vmin = np.percentile(values, 2)

    vmax = np.percentile(values, vmax_percentile)

    # =====================================================
    # Figure
    # =====================================================

    plt.figure(figsize=figsize)

    # =====================================================
    # HEATMAP
    #
    # CHANGES FOR BETTER CONTRAST:
    #
    # 1. magma / inferno / rocket colormap
    # 2. PowerNorm boosts mid-range contrast
    # 3. dark background
    # 4. higher vmax robustness
    # =====================================================

    ax = sns.heatmap(

        matrix,

        cmap="magma",
        # alternatives:
        # cmap="inferno"
        # cmap="rocket"

        norm=PowerNorm(
            gamma=0.55,
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=True,

        yticklabels=False,

        linewidths=0,

        linecolor=None,

        cbar_kws={
            "label": "Interaction Weight"
        },
    )

    # =====================================================
    # DARK BACKGROUND
    # =====================================================

    ax.set_facecolor("black")

    # =====================================================
    # TITLES
    # =====================================================

    ax.set_title(
        f"{omics}",
        fontsize=24,
        pad=14,
        # # fontweight="bold",
    )

    ax.set_xlabel(
        "",
        fontsize=18,
    )

    ax.set_ylabel(
        "Gene",
        fontsize=20,
        # # fontweight="bold",
    )

    # =====================================================
    # REMOVE SPINES
    # =====================================================

    for spine in ax.spines.values():

        spine.set_visible(False)

    # =====================================================
    # X TICKS
    # =====================================================

    ax.tick_params(
        axis="x",
        labelrotation=0,
        labelsize=14,
        colors="black",
    )

    # =====================================================
    # COLORBAR
    # =====================================================

    cbar = ax.collections[0].colorbar

    cbar.ax.tick_params(
        labelsize=13
    )

    cbar.set_label(
        "Interaction Weight",
        fontsize=16,
        # # fontweight="bold",
    )

    # =====================================================
    # OPTIONAL:
    # stronger cancer labels
    # =====================================================

    plt.setp(
        ax.get_xticklabels(),
        # # fontweight="bold"
    )

    # =====================================================
    # LAYOUT
    # =====================================================

    plt.tight_layout()

    # =====================================================
    # SAVE
    # =====================================================

    save_path = os.path.join(
        output_dir,
        f"{omics}_gene_cancer_bipartite_heatmap.png"
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"[SAVED] {save_path}")

    # =====================================================
    # SHOW
    # =====================================================

    plt.show()

    plt.close()

    
def leiden_bipartite_clustering(matrix, resolution=1.0):
    """
    Leiden clustering on gene–cancer bipartite graph.
    Returns reordered matrix + cluster labels.
    """
    B = nx.Graph()

    B.add_nodes_from(matrix.index, bipartite=0, type="gene")
    B.add_nodes_from(matrix.columns, bipartite=1, type="cancer")

    for g in matrix.index:
        for c in matrix.columns:
            w = matrix.loc[g, c]
            if not np.isnan(w) and w > 0:
                B.add_edge(g, c, weight=float(w))

    edges = [(u, v, float(d["weight"])) for u, v, d in B.edges(data=True)]
    ig_B = ig.Graph.TupleList(edges, edge_attrs=["weight"], directed=False)

    partition = la.find_partition(
        ig_B,
        la.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
    )

    cluster_map = dict(zip(ig_B.vs["name"], partition.membership))

    gene_clusters = {n: cluster_map[n] for n in matrix.index if n in cluster_map}
    cancer_clusters = {n: cluster_map[n] for n in matrix.columns if n in cluster_map}

    gene_order = sorted(gene_clusters, key=lambda x: gene_clusters[x])
    cancer_order = sorted(cancer_clusters, key=lambda x: cancer_clusters[x])

    reordered = matrix.loc[gene_order, cancer_order]

    return reordered, gene_clusters, cancer_clusters

# def split_gene_cancer_by_omics(gene_cancer_weighted):
#     """
#     Splits gene × (OMICS:CANCER) matrix into separate
#     gene × cancer matrices per omics type.
#     """
#     omics_types = ["MF", "METH", "GE", "CNA"]
#     omics_matrices = {}

#     for omics in omics_types:
#         cols = [
#             c for c in gene_cancer_weighted.columns
#             if c.startswith(f"{omics}:")
#         ]
#         if len(cols) == 0:
#             continue

#         mat = gene_cancer_weighted[cols].copy()
#         mat.columns = [c.split(": ")[1] for c in mat.columns]  # keep cancer only
#         mat = mat.loc[mat.sum(axis=1) > 0]  # drop zero rows

#         if mat.shape[0] > 0 and mat.shape[1] > 0:
#             omics_matrices[omics] = mat

#     return omics_matrices

def split_gene_cancer_by_omics(gene_cancer_weighted):
    """
    Split a gene × (OMICS: CANCER) matrix into separate
    gene × cancer matrices for each omics type.

    Expected column format:
        GE: BLCA
        MF: BRCA
        MIRNA: LUAD
        ...

    Returns
    -------
    omics_matrices : dict
        {
            "GE":     gene × cancer DataFrame,
            "MF":     gene × cancer DataFrame,
            "MIRNA":  gene × cancer DataFrame,
        }
    """

    import numpy as np
    import pandas as pd

    # ---------------------------------------------------
    # Detect omics types automatically
    # ---------------------------------------------------
    omics_types = sorted({
        c.split(": ")[0]
        for c in gene_cancer_weighted.columns
        if ": " in c
    })

    print("Detected omics types:", omics_types)

    omics_matrices = {}

    # ---------------------------------------------------
    # Process each omics type
    # ---------------------------------------------------
    for omics in omics_types:

        # -----------------------------------------------
        # Select columns for this omics
        # -----------------------------------------------
        cols = [
            c for c in gene_cancer_weighted.columns
            if c.startswith(f"{omics}:")
        ]

        if len(cols) == 0:
            continue

        # -----------------------------------------------
        # Subset matrix
        # -----------------------------------------------
        mat = gene_cancer_weighted[cols].copy()

        # -----------------------------------------------
        # Rename columns:
        # "GE: BLCA" -> "BLCA"
        # -----------------------------------------------
        mat.columns = [
            c.split(": ")[1]
            for c in mat.columns
        ]

        # -----------------------------------------------
        # Remove invalid values
        # -----------------------------------------------
        mat = mat.replace(
            [np.inf, -np.inf],
            np.nan
        ).fillna(0)

        # -----------------------------------------------
        # Drop all-zero genes
        # -----------------------------------------------
        mat = mat.loc[
            mat.sum(axis=1) > 0
        ]

        # -----------------------------------------------
        # Save
        # -----------------------------------------------
        if mat.shape[0] > 0 and mat.shape[1] > 0:

            omics_matrices[omics] = mat

            print(
                f"{omics}: "
                f"{mat.shape[0]} genes × "
                f"{mat.shape[1]} cancers"
            )

    return omics_matrices


def split_gene_cancer_by_omics(gene_cancer_weighted):
    """
    Split gene × (OMICS:CANCER) matrix into
    separate gene × cancer matrices for each omics type.

    Expected columns:
        GE: BLCA
        MF: BRCA
        MIRNA: LUAD
        etc.
    """

    import pandas as pd

    # ---------------------------------------------------
    # Actual omics types in your dataframe
    # ---------------------------------------------------
    omics_types = [
        "GE",
        "MF",
        "MIRNA",
    ]

    omics_matrices = {}

    # ---------------------------------------------------
    # Split by omics prefix
    # ---------------------------------------------------
    for omics in omics_types:

        cols = [
            c for c in gene_cancer_weighted.columns
            if c.startswith(f"{omics}:")
        ]

        print(f"{omics} columns:", len(cols))

        # Skip if absent
        if len(cols) == 0:
            continue

        # ---------------------------------------------------
        # Extract matrix
        # ---------------------------------------------------
        mat = gene_cancer_weighted[cols].copy()

        # ---------------------------------------------------
        # Keep ONLY cancer names
        # Example:
        #   GE: BLCA  -> BLCA
        # ---------------------------------------------------
        mat.columns = [
            c.split(": ")[1]
            for c in mat.columns
        ]

        # ---------------------------------------------------
        # Remove all-zero rows
        # ---------------------------------------------------
        mat = mat.loc[
            mat.sum(axis=1) > 0
        ]

        # ---------------------------------------------------
        # Remove NaN / inf
        # ---------------------------------------------------
        mat = mat.replace(
            [float("inf"), float("-inf")],
            0
        )

        mat = mat.fillna(0)

        # ---------------------------------------------------
        # Save
        # ---------------------------------------------------
        if mat.shape[0] > 0 and mat.shape[1] > 0:

            omics_matrices[omics] = mat

            print(
                f"{omics} matrix shape:",
                mat.shape
            )

    return omics_matrices


def plot_gene_cancer_biclustering_heatmap_with_gene_clusters_ori(
    gene_cancer_biclustered,
    gene_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Plot gene–cancer biclustering heatmap with gene cluster color bar.

    Parameters
    ----------
    gene_cancer_biclustered : np.ndarray or pd.DataFrame
        Matrix of shape (genes × cancers), already biclustered.
    gene_clusters : list or np.ndarray
        Leiden cluster ID per gene, ordered to match rows.
    output_dir : str
        Directory to save figure.
    """

    os.makedirs(output_dir, exist_ok=True)

    # # -----------------------------
    # # Align gene Leiden labels to heatmap rows
    # # -----------------------------
    # gene_leiden_labels = [
    #     gene_clusters[g] for g in gene_cancer_biclustered.index
    # ]

    # -------------------------
    # Prepare gene cluster colors
    # -------------------------
    gene_cluster_colors = np.array([
        to_rgb(CLUSTER_COLORS[c]) for c in gene_clusters
    ])

    # -------------------------
    # Plot
    # -------------------------
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=[0.03, 0.97],
        wspace=0.02
    )

    ax_cluster = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])

    # Gene cluster bar
    ax_cluster.imshow(gene_cluster_colors[:, None, :], aspect="auto")
    ax_cluster.set_axis_off()

    # Heatmap
    sns.heatmap(
        gene_cancer_biclustered,
        ax=ax_main,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=True,
    )

    ax_main.set_title(title)
    ax_main.set_xlabel("Cancer Type")
    ax_main.set_ylabel("Genes")

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Heatmap saved to {save_path}")


def plot_gene_cancer_biclustering_heatmap_with_gene_clusters_ori_(
    gene_cancer_biclustered,
    gene_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Gene–cancer biclustering heatmap with flush gene cluster bar (Leiden style).
    """

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Prepare gene cluster colors (row-aligned)
    # -------------------------
    gene_cluster_colors = np.array([
        to_rgb(CLUSTER_COLORS[c]) for c in gene_clusters
    ])

    # -------------------------
    # Layout (flush bar)
    # -------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 2,
        width_ratios=[0.022, 0.978],  # very thin bar
        wspace=0.0
    )

    ax_bar = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])

    # -------------------------
    # Gene cluster bar (LEFT)
    # -------------------------
    ax_bar.imshow(gene_cluster_colors[:, None, :], aspect="auto")
    ax_bar.set_axis_off()

    # -------------------------
    # Heatmap
    # -------------------------
    sns.heatmap(
        gene_cancer_biclustered,
        ax=ax_main,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=True,
        rasterized=True,
    )

    ax_main.set_title(title)
    ax_main.set_xlabel("Cancer Type")
    # ax_main.set_ylabel("Genes")

    # -------------------------
    # Hard alignment (key part)
    # -------------------------
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    bar_pos = ax_bar.get_position()

    ax_bar.set_position([
        bar_pos.x0,
        main_pos.y0,
        bar_pos.width,
        main_pos.height
    ])
    
    # -------------------------
    # Gene label aligned with cluster bar
    # -------------------------
    ax_bar.text(
        -0.6,                     # push text left of the bar
        0.5,                      # vertically centered
        "Genes",
        rotation=90,
        va="center",
        ha="center",
        fontsize=20,
        transform=ax_bar.transAxes
    )


    # -------------------------
    # Save
    # -------------------------
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_with_gene_clusters_reds(
    gene_cancer_biclustered,
    gene_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Gene–cancer biclustering heatmap with Leiden-style cluster bars
    and short, thin legend (matching plot_leiden_saliency_heatmap).
    """

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Cluster colors
    # -------------------------
    gene_cluster_colors = np.array([
        to_rgb(CLUSTER_COLORS[c]) for c in gene_clusters
    ])

    # -------------------------
    # Layout (bar | heatmap | colorbar)
    # -------------------------
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        1, 3,
        width_ratios=[0.02, 0.92, 0.06],
        wspace=0.0,
    )

    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    
    

    # -------------------------
    # Gene cluster bar (LEFT)
    # -------------------------
    ax_bar.imshow(gene_cluster_colors[:, None, :], aspect="auto")
    ax_bar.set_axis_off()

    # -------------------------
    # Heatmap
    # -------------------------
    hm = sns.heatmap(
        gene_cancer_biclustered,
        ax=ax_main,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )
    
    ax_main.set_xticklabels(
        ax_main.get_xticklabels(),
        fontsize=16,
        rotation=45,
        ha="right"
    )



    # ax_main.set_title(title, fontsize=18)
    # ax_main.set_xlabel("Cancer Type", fontsize=16)
    ax_main.set_ylabel("")

    # -------------------------
    # Colorbar styling (Leiden style)
    # -------------------------
    cbar = hm.collections[0].colorbar
    # cbar.set_label("Saliency", fontsize=22, labelpad=16)
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)

    # -------------------------
    # External gene label (like saliency plot)
    # -------------------------
    ax_bar.text(
        -1.6, 0.5,
        "Genes",
        rotation=90,
        ha="center",
        va="center",
        fontsize=22,
        transform=ax_bar.transAxes,
    )

    # -------------------------
    # Alignment + resize colorbar
    # -------------------------
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # Flush gene bar with heatmap
    ax_bar.set_position([
        ax_bar.get_position().x0,
        main_pos.y0,
        ax_bar.get_position().width,
        main_pos.height,
    ])

    # Short, thin colorbar (same logic as saliency plot)
    new_h = main_pos.height * 0.35
    new_w = cbar_pos.width * 0.35

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])


    # -------------------------
    # Reduce outer margins
    # -------------------------
    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)
    
    # -------------------------
    # Save
    # -------------------------
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_with_gene_clusters(
    gene_cancer_biclustered,
    gene_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Blues",  # <-- changed to blue
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Gene–cancer biclustering heatmap with Leiden-style cluster bars
    and short, thin legend (matching plot_leiden_saliency_heatmap).
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.gridspec import GridSpec

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Cluster colors
    # -------------------------
    gene_cluster_colors = np.array([
        to_rgb(CLUSTER_COLORS[c]) for c in gene_clusters
    ])

    # -------------------------
    # Layout (bar | heatmap | colorbar)
    # -------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        1, 3,
        width_ratios=[0.02, 0.92, 0.06],
        wspace=0.0,
    )

    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # -------------------------
    # Gene cluster bar (LEFT)
    # -------------------------
    ax_bar.imshow(gene_cluster_colors[:, None, :], aspect="auto")
    ax_bar.set_axis_off()

    # -------------------------
    # Heatmap
    # -------------------------
    hm = sns.heatmap(
        gene_cancer_biclustered,
        ax=ax_main,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xticklabels(
        ax_main.get_xticklabels(),
        fontsize=16,
        rotation=45,
        ha="right"
    )
    ax_main.set_ylabel("")

    # -------------------------
    # Colorbar styling (Leiden style)
    # -------------------------
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.outline.set_visible(False)

    # -------------------------
    # External gene label
    # -------------------------
    ax_bar.text(
        -1.6, 0.5,
        "Genes",
        rotation=90,
        ha="center",
        va="center",
        fontsize=22,
        transform=ax_bar.transAxes,
    )

    # -------------------------
    # Alignment + resize colorbar
    # -------------------------
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # Flush gene bar with heatmap
    ax_bar.set_position([
        ax_bar.get_position().x0,
        main_pos.y0,
        ax_bar.get_position().width,
        main_pos.height,
    ])

    # # Short, thin colorbar
    # new_h = main_pos.height * 0.35
    # new_w = ax_bar.get_position().width

    # ax_cbar.set_position([
    #     cbar_pos.x0 + 0.02,
    #     main_pos.y0 + (main_pos.height - new_h) / 2,
    #     new_w,
    #     new_h,
    # ])

    # --- Match colorbar width to gene cluster bar ---

    bar_pos = ax_bar.get_position()   # gene cluster bar size

    new_h = main_pos.height * 0.35
    new_w = bar_pos.width             # <-- exact same width

    ax_cbar.set_position([
        cbar_pos.x0 + 0.01,            # small gap from heatmap
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # -------------------------
    # Reduce outer margins
    # -------------------------
    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)

    # -------------------------
    # Save
    # -------------------------
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_with_gene_clusters__(
    gene_cancer_biclustered,
    gene_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Blues",
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Gene–cancer biclustering heatmap with Leiden-style cluster bars
    and short thin colorbar (matched to plot_leiden_saliency_heatmap).
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.gridspec import GridSpec

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================
    # Cluster colors
    # ==========================================================
    gene_cluster_colors = np.array([
        to_rgb(CLUSTER_COLORS[c]) for c in gene_clusters
    ])

    # ==========================================================
    # Layout (bar | heatmap | colorbar)
    # ==========================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        1, 3,
        width_ratios=[0.015, 0.94, 0.045],
        wspace=0.02,
    )

    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # ==========================================================
    # Gene cluster bar (LEFT)
    # ==========================================================
    ax_bar.imshow(gene_cluster_colors[:, None, :], aspect="auto")
    ax_bar.set_axis_off()

    # ==========================================================
    # Heatmap
    # ==========================================================
    hm = sns.heatmap(
        gene_cancer_biclustered,
        ax=ax_main,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xticklabels(
        ax_main.get_xticklabels(),
        fontsize=16,
        rotation=45,
        ha="right"
    )
    ax_main.set_ylabel("")

    # ==========================================================
    # Colorbar styling (IDENTICAL to Leiden heatmap)
    # ==========================================================
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)

    # ==========================================================
    # External gene label
    # ==========================================================
    ax_bar.text(
        -1.6, 0.5,
        "Genes",
        rotation=90,
        ha="center",
        va="center",
        fontsize=22,
        transform=ax_bar.transAxes,
    )

    # ==========================================================
    # Alignment + short thin colorbar
    # ==========================================================
    fig.canvas.draw()

    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # Flush gene bar with heatmap
    ax_bar.set_position([
        ax_bar.get_position().x0,
        main_pos.y0,
        ax_bar.get_position().width,
        main_pos.height,
    ])

    # Short + thin colorbar (centered)
    new_h = main_pos.height * 0.35
    # new_w = cbar_pos.width * 0.35
    new_w = ax_bar.get_position().width

    ax_cbar.set_position([
        cbar_pos.x0 + 0.01,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # ==========================================================
    # Reduce outer margins
    # ==========================================================
    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.97,
        bottom=0.12
    )

    # ==========================================================
    # Save
    # ==========================================================
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"Heatmap saved to {save_path}")


def plot_gene_cancer_biclustering_heatmap_with_gene_clusters_(
    gene_cancer_biclustered,
    gene_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Gene–cancer biclustering heatmap with Leiden-style cluster bars
    and short, thin legend.
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import to_rgb

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Cluster colors
    # -------------------------
    gene_cluster_colors = np.array([
        to_rgb(CLUSTER_COLORS[c]) for c in gene_clusters
    ])

    # -------------------------
    # Layout (bar | heatmap | colorbar)
    # -------------------------
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        1, 3,
        width_ratios=[0.015, 0.94, 0.045],   # thinner bar + thinner colorbar
        wspace=0.02,
    )

    ax_bar  = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])

    # -------------------------
    # Gene cluster bar (LEFT)
    # -------------------------
    ax_bar.imshow(gene_cluster_colors[:, None, :], aspect="auto")
    ax_bar.set_axis_off()

    # -------------------------
    # Heatmap
    # -------------------------
    hm = sns.heatmap(
        gene_cancer_biclustered,
        ax=ax_main,
        cmap=cmap,
        xticklabels=True,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xticklabels(
        ax_main.get_xticklabels(),
        fontsize=16,
        rotation=45,
        ha="right"
    )

    ax_main.set_ylabel("")

    # -------------------------
    # Colorbar styling
    # -------------------------
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)

    # -------------------------
    # External gene label
    # -------------------------
    ax_bar.text(
        -1.6, 0.5,
        "Genes",
        rotation=90,
        ha="center",
        va="center",
        fontsize=22,
        transform=ax_bar.transAxes,
    )

    # -------------------------
    # Align bar and resize colorbar
    # -------------------------
    fig.canvas.draw()

    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    # Flush gene bar with heatmap
    ax_bar.set_position([
        ax_bar.get_position().x0,
        main_pos.y0,
        ax_bar.get_position().width,
        main_pos.height,
    ])

    # Shorter & thinner colorbar
    new_h = main_pos.height * 0.35
    new_w = cbar_pos.width * 0.35

    ax_cbar.set_position([
        cbar_pos.x0 + 0.01,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # -------------------------
    # Reduce outer margins
    # -------------------------
    plt.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.12)

    # -------------------------
    # Save
    # -------------------------
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_reds(
    gene_cancer_biclustered,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    show_xticklabels=True,
    show_yticklabels=False,
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
):
    """
    Plot and save a gene–cancer biclustering heatmap.

    Parameters
    ----------
    gene_cancer_biclustered : array-like or DataFrame
        Biclustered gene–cancer matrix (genes x cancers).
    output_dir : str
        Directory to save the figure.
    filename : str, optional
        Output image filename.
    figsize : tuple, optional
        Figure size.
    cmap : str, optional
        Matplotlib/Seaborn colormap.
    show_xticklabels : bool, optional
        Whether to show x-axis tick labels.
    show_yticklabels : bool, optional
        Whether to show y-axis tick labels.
    title : str, optional
        Plot title.
    """

    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=figsize)
    sns.heatmap(
        gene_cancer_biclustered,
        cmap=cmap,
        xticklabels=show_xticklabels,
        yticklabels=show_yticklabels,
    )

    plt.title(title)
    plt.xlabel("Cancer Type")
    plt.ylabel("Genes")
    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_(
    gene_cancer_biclustered,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    show_xticklabels=True,
    show_yticklabels=False,
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
    vmax_percentile=99
):
    """
    Plot and save a gene–cancer biclustering heatmap with log-scaled colorbar.
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatter

    os.makedirs(output_dir, exist_ok=True)

    # -------------------------
    # Prepare matrix
    # -------------------------
    if hasattr(gene_cancer_biclustered, "values"):
        mat = gene_cancer_biclustered.values
    else:
        mat = np.asarray(gene_cancer_biclustered)

    # Avoid log(0)
    mat_plot = np.clip(mat, 1e-6, None)

    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=figsize)

    hm = sns.heatmap(
        mat_plot,
        cmap=cmap,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        xticklabels=show_xticklabels,
        yticklabels=show_yticklabels,
        cbar=True,
        rasterized=True,
    )

    # -------------------------
    # Colorbar formatting (MATCHES Leiden plot)
    # -------------------------
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(LogFormatter())
    cbar.ax.tick_params(labelsize=14)
    cbar.outline.set_visible(False)

    plt.title(title)
    plt.xlabel("Cancer Type")
    plt.ylabel("Genes")

    plt.tight_layout()

    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.show()

    print(f"Heatmap saved to {save_path}")


def plot_gene_cancer_biclustering_heatmap_red_pas_ori(
    gene_cancer_biclustered,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    show_xticklabels=True,
    show_yticklabels=False,
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
    vmax_percentile=99
):
    """
    Plot and save a gene–cancer biclustering heatmap
    with log-scaled colorbar and real cancer names
    on the x-axis.
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatter

    # ---------------------------------------------------
    # Create output directory
    # ---------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------
    # Prepare matrix
    # ---------------------------------------------------
    if hasattr(gene_cancer_biclustered, "values"):
        mat = gene_cancer_biclustered.values
    else:
        mat = np.asarray(gene_cancer_biclustered)

    # ---------------------------------------------------
    # Avoid zeros for log scaling
    # ---------------------------------------------------
    mat_plot = np.clip(mat, 1e-6, None)

    # ---------------------------------------------------
    # Color scaling
    # ---------------------------------------------------
    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # ---------------------------------------------------
    # REAL cancer labels
    # ---------------------------------------------------
    if hasattr(gene_cancer_biclustered, "columns"):
        cancer_labels = list(gene_cancer_biclustered.columns)
    else:
        cancer_labels = np.arange(mat_plot.shape[1])

    # ---------------------------------------------------
    # Gene labels
    # ---------------------------------------------------
    if hasattr(gene_cancer_biclustered, "index"):
        gene_labels = list(gene_cancer_biclustered.index)
    else:
        gene_labels = np.arange(mat_plot.shape[0])

    # ---------------------------------------------------
    # Plot
    # ---------------------------------------------------
    plt.figure(figsize=figsize)

    hm = sns.heatmap(
        mat_plot,

        cmap=cmap,

        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=(
            cancer_labels
            if show_xticklabels
            else False
        ),

        yticklabels=(
            gene_labels
            if show_yticklabels
            else False
        ),

        cbar=True,

        rasterized=True,
    )

    # ---------------------------------------------------
    # Colorbar formatting
    # ---------------------------------------------------
    cbar = hm.collections[0].colorbar

    cbar.ax.yaxis.set_major_formatter(
        LogFormatter()
    )

    cbar.ax.tick_params(
        labelsize=14
    )

    cbar.outline.set_visible(False)

    cbar.set_label(
        "",
        fontsize=14
    )

    # ---------------------------------------------------
    # X-axis formatting
    # ---------------------------------------------------
    plt.xticks(
        rotation=0,
        ha="right",
        fontsize=12
    )

    # ---------------------------------------------------
    # Y-axis formatting
    # ---------------------------------------------------
    plt.yticks(
        fontsize=8
    )

    # ---------------------------------------------------
    # Labels
    # ---------------------------------------------------
    plt.xlabel(
        "Cancer Type",
        fontsize=14
    )

    plt.ylabel(
        "Genes",
        fontsize=14
    )

    plt.title(
        title,
        fontsize=16
    )

    # ---------------------------------------------------
    # Tight layout
    # ---------------------------------------------------
    plt.tight_layout()

    # ---------------------------------------------------
    # Save
    # ---------------------------------------------------
    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_red_separate_cell(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(14, 10),
    cmap="Reds",
    show_xticklabels=True,
    show_yticklabels=False,
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
    vmax_percentile=99
):
    """
    Plot gene–cancer biclustering heatmap with:

    - left gene cluster color bar
    - top cancer cluster color bar
    - log-scaled heatmap
    - real cancer labels
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from matplotlib.colors import (
        LogNorm,
        to_rgb
    )

    from matplotlib.ticker import LogFormatter

    from matplotlib.gridspec import GridSpec

    # ---------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ---------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------
    # MATRIX
    # ---------------------------------------------------
    if hasattr(gene_cancer_biclustered, "values"):
        mat = gene_cancer_biclustered.values
    else:
        mat = np.asarray(gene_cancer_biclustered)

    # ---------------------------------------------------
    # LABELS
    # ---------------------------------------------------
    if hasattr(gene_cancer_biclustered, "columns"):
        cancer_labels = list(
            gene_cancer_biclustered.columns
        )
    else:
        cancer_labels = list(
            np.arange(mat.shape[1])
        )

    if hasattr(gene_cancer_biclustered, "index"):
        gene_labels = list(
            gene_cancer_biclustered.index
        )
    else:
        gene_labels = list(
            np.arange(mat.shape[0])
        )

    # ---------------------------------------------------
    # LOG SCALE
    # ---------------------------------------------------
    mat_plot = np.clip(mat, 1e-6, None)

    vmin = np.percentile(
        mat_plot,
        5
    )

    vmax = np.percentile(
        mat_plot,
        vmax_percentile
    )

    # ---------------------------------------------------
    # DEFAULT COLORS
    # ---------------------------------------------------
    unique_gene_clusters = sorted(
        set(gene_clusters.values())
    )

    unique_cancer_clusters = sorted(
        set(cancer_clusters.values())
    )

    palette = sns.color_palette(
        "tab20",
        max(
            len(unique_gene_clusters),
            len(unique_cancer_clusters),
            20
        )
    )

    CLUSTER_COLORS = {
        c: palette[i % len(palette)]
        for i, c in enumerate(
            sorted(
                set(unique_gene_clusters + unique_cancer_clusters)
            )
        )
    }

    # ---------------------------------------------------
    # GENE CLUSTER COLORS
    # ---------------------------------------------------
    gene_colors = np.array([
        to_rgb(
            CLUSTER_COLORS[
                gene_clusters.get(g, -1)
            ]
        )
        for g in gene_labels
    ])

    # ---------------------------------------------------
    # CANCER CLUSTER COLORS
    # ---------------------------------------------------
    cancer_colors = np.array([
        to_rgb(
            CLUSTER_COLORS[
                cancer_clusters.get(c, -1)
            ]
        )
        for c in cancer_labels
    ])

    # ---------------------------------------------------
    # FIGURE LAYOUT
    # ---------------------------------------------------
    fig = plt.figure(
        figsize=figsize
    )

    gs = GridSpec(
        2,
        3,

        height_ratios=[0.035, 0.965],

        width_ratios=[0.022, 0.92, 0.058],

        hspace=0.0,

        wspace=0.0
    )

    ax_top = fig.add_subplot(gs[0, 1])

    ax_left = fig.add_subplot(gs[1, 0])

    ax_main = fig.add_subplot(gs[1, 1])

    ax_cbar = fig.add_subplot(gs[1, 2])

    # ---------------------------------------------------
    # HEATMAP
    # ---------------------------------------------------
    hm = sns.heatmap(
        mat_plot,

        ax=ax_main,

        cmap=cmap,

        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=(
            cancer_labels
            if show_xticklabels
            else False
        ),

        yticklabels=(
            gene_labels
            if show_yticklabels
            else False
        ),

        cbar=True,

        cbar_ax=ax_cbar,

        rasterized=True,
    )

    # ---------------------------------------------------
    # X TICKS
    # ---------------------------------------------------
    ax_main.set_xticklabels(
        ax_main.get_xticklabels(),

        rotation=0,

        ha="center",

        fontsize=12
    )

    # ---------------------------------------------------
    # Y TICKS
    # ---------------------------------------------------
    ax_main.set_yticklabels(
        ax_main.get_yticklabels(),

        fontsize=8
    )

    # ---------------------------------------------------
    # REMOVE AXIS LABELS
    # ---------------------------------------------------
    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # ---------------------------------------------------
    # COLORBAR FORMAT
    # ---------------------------------------------------
    cbar = hm.collections[0].colorbar

    cbar.ax.yaxis.set_major_formatter(
        LogFormatter()
    )

    cbar.ax.tick_params(
        labelsize=12
    )

    cbar.outline.set_visible(False)

    cbar.set_label(
        "Weighted Saliency",
        fontsize=14
    )

    # ---------------------------------------------------
    # LEFT GENE CLUSTER BAR
    # ---------------------------------------------------
    ax_left.imshow(
        gene_colors[:, None, :],

        aspect="auto"
    )

    ax_left.set_axis_off()

    ax_left.text(
        -1.8,
        0.5,

        "Genes",

        ha="center",

        va="center",

        rotation=90,

        transform=ax_left.transAxes,

        fontsize=18,
    )

    # ---------------------------------------------------
    # TOP CANCER CLUSTER BAR
    # ---------------------------------------------------
    ax_top.imshow(
        cancer_colors[None, :, :],

        aspect="auto"
    )

    ax_top.set_axis_off()

    ax_top.text(
        0.5,
        1.8,

        "Cancers",

        ha="center",

        va="center",

        transform=ax_top.transAxes,

        fontsize=18,
    )

    # ---------------------------------------------------
    # TITLE
    # ---------------------------------------------------
    plt.suptitle(
        title,
        fontsize=18,
        y=1.02
    )

    # ---------------------------------------------------
    # ALIGNMENT
    # ---------------------------------------------------
    fig.canvas.draw()

    main_pos = ax_main.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height
    ])

    # ---------------------------------------------------
    # SHORTER COLORBAR
    # ---------------------------------------------------
    cbar_pos = ax_cbar.get_position()

    new_h = main_pos.height * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (
            main_pos.height - new_h
        ) / 2,
        ax_left.get_position().width,
        new_h
    ])

    # ---------------------------------------------------
    # TIGHT LAYOUT
    # ---------------------------------------------------
    plt.tight_layout()

    # ---------------------------------------------------
    # SAVE
    # ---------------------------------------------------
    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,

        dpi=300,

        bbox_inches="tight",

        pad_inches=0.02
    )

    plt.show()

    print(f"Heatmap saved to {save_path}")

def plot_gene_cancer_biclustering_heatmap_omics_heatmap_pas(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(16, 12),
    cmap="Blues",
    show_xticklabels=True,
    show_yticklabels=False,
    title="Gene–Cancer Biclustering (Leiden Bipartite)",
    vmax_percentile=99,
):
    """
    Plot Leiden-style gene–cancer biclustering heatmap with:
    - top cancer cluster color bar
    - left gene cluster color bar
    - log-scaled colorbar
    - real cancer names on x-axis
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm, to_rgb
    from matplotlib.ticker import LogFormatter
    from matplotlib.gridspec import GridSpec

    # ---------------------------------------------------
    # Create output directory
    # ---------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------
    # Extract names
    # ---------------------------------------------------
    gene_names = list(gene_cancer_biclustered.index)
    cancer_names = list(gene_cancer_biclustered.columns)

    # ---------------------------------------------------
    # Sort by cluster
    # ---------------------------------------------------
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters.get(gene_names[i], -1)
    )

    cancer_order = sorted(
        range(len(cancer_names)),
        key=lambda i: cancer_clusters.get(cancer_names[i], -1)
    )

    # ---------------------------------------------------
    # Reorder matrix
    # ---------------------------------------------------
    reordered = gene_cancer_biclustered.iloc[
        gene_order,
        cancer_order
    ]

    mat = reordered.values

    ordered_gene_names = reordered.index.tolist()
    ordered_cancer_names = reordered.columns.tolist()

    # ---------------------------------------------------
    # Avoid zeros for log scaling
    # ---------------------------------------------------
    mat_plot = np.clip(mat, 1e-6, None)

    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # ---------------------------------------------------
    # Cluster colors
    # ---------------------------------------------------
    gene_colors = np.array([
        to_rgb(
            CLUSTER_COLORS.get(
                gene_clusters.get(g, -1),
                "#B0B0B0"
            )
        )
        for g in ordered_gene_names
    ])

    cancer_colors = np.array([
        to_rgb(
            CLUSTER_COLORS.get(
                cancer_clusters.get(c, -1),
                "#B0B0B0"
            )
        )
        for c in ordered_cancer_names
    ])

    # ---------------------------------------------------
    # Figure layout
    # ---------------------------------------------------
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2,
        3,

        height_ratios=[0.035, 0.965],

        width_ratios=[0.022, 0.92, 0.058],

        hspace=0.0,
        wspace=0.0
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # ---------------------------------------------------
    # Main heatmap
    # ---------------------------------------------------
    hm = sns.heatmap(
        mat_plot,

        ax=ax_main,

        cmap=cmap,

        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=(
            ordered_cancer_names
            if show_xticklabels
            else False
        ),

        yticklabels=(
            ordered_gene_names
            if show_yticklabels
            else False
        ),

        cbar=True,

        cbar_ax=ax_cbar,

        rasterized=True,
    )

    # ---------------------------------------------------
    # X labels
    # ---------------------------------------------------
    ax_main.set_xticklabels(
        ax_main.get_xticklabels(),
        fontsize=16,
        rotation=0,
        ha="center"
    )

    # ---------------------------------------------------
    # Y labels
    # ---------------------------------------------------
    ax_main.set_yticklabels(
        ax_main.get_yticklabels(),
        fontsize=7
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # ---------------------------------------------------
    # Colorbar styling
    # ---------------------------------------------------
    cbar = hm.collections[0].colorbar

    cbar.ax.yaxis.set_major_formatter(
        LogFormatter()
    )

    cbar.ax.tick_params(
        labelsize=18
    )

    cbar.outline.set_visible(False)

    cbar.set_label(
        "Weighted Saliency",
        fontsize=18
    )

    # ---------------------------------------------------
    # Left gene-cluster color bar
    # ---------------------------------------------------
    ax_left.imshow(
        gene_colors[:, None, :],
        aspect="auto"
    )

    ax_left.set_axis_off()

    ax_left.text(
        -1.8,
        0.5,

        "Genes",

        ha="center",
        va="center",

        rotation=90,

        transform=ax_left.transAxes,

        fontsize=22,
    )

    # ---------------------------------------------------
    # Top cancer-cluster color bar
    # ---------------------------------------------------
    ax_top.imshow(
        cancer_colors[None, :, :],
        aspect="auto"
    )

    ax_top.set_axis_off()

    ax_top.text(
        0.5,
        1.8,

        "Cancers",

        ha="center",
        va="center",

        transform=ax_top.transAxes,

        fontsize=22,
    )

    # ---------------------------------------------------
    # Align bars to heatmap
    # ---------------------------------------------------
    fig.canvas.draw()

    main_pos = ax_main.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height
    ])

    # ---------------------------------------------------
    # Shorter colorbar
    # ---------------------------------------------------
    cbar_pos = ax_cbar.get_position()

    new_h = main_pos.height * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        ax_left.get_position().width,
        new_h
    ])

    # ---------------------------------------------------
    # Title
    # ---------------------------------------------------
    plt.suptitle(
        title,
        fontsize=18,
        y=0.98
    )

    # ---------------------------------------------------
    # Save
    # ---------------------------------------------------
    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.show()

    print(f"Heatmap saved to {save_path}")
    
def plot_gene_cancer_biclustering_heatmap_(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(16, 12),
    cmap="Blues",
    show_xticklabels=True,
    show_yticklabels=False,
    vmax_percentile=99,
):
    """
    Plot Leiden-style gene–cancer biclustering heatmap with:

    - top cancer cluster color bar
    - left gene cluster color bar
    - log-scaled colorbar
    - real cancer names on x-axis

    IMPORTANT:
    This assumes columns are REAL cancer names:
        BLCA, BRCA, COAD, ESCA, LUAD, LUSC, STAD

    NOT:
        GE: BLCA
        MF: BLCA
        MIRNA: BLCA

    Omics-specific matrices should already be split BEFORE calling this.
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm, to_rgb
    from matplotlib.ticker import LogFormatter
    from matplotlib.gridspec import GridSpec

    # =====================================================
    # OUTPUT DIR
    # =====================================================
    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # EXTRACT NAMES
    # =====================================================
    gene_names = list(gene_cancer_biclustered.index)
    cancer_names = list(gene_cancer_biclustered.columns)

    # =====================================================
    # FILTER VALID CLUSTERS
    # =====================================================
    gene_names = [
        g for g in gene_names
        if g in gene_clusters
    ]

    cancer_names = [
        c for c in cancer_names
        if c in cancer_clusters
    ]

    # =====================================================
    # SUBSET MATRIX
    # =====================================================
    gene_cancer_biclustered = gene_cancer_biclustered.loc[
        gene_names,
        cancer_names
    ]

    # =====================================================
    # SORT BY CLUSTER
    # =====================================================
    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )

    cancer_order = sorted(
        range(len(cancer_names)),
        key=lambda i: cancer_clusters[cancer_names[i]]
    )

    # =====================================================
    # REORDER MATRIX
    # =====================================================
    reordered = gene_cancer_biclustered.iloc[
        gene_order,
        cancer_order
    ]

    ordered_gene_names = reordered.index.tolist()
    ordered_cancer_names = reordered.columns.tolist()

    mat = reordered.values

    # =====================================================
    # LOG SCALE
    # =====================================================
    mat_plot = np.clip(mat, 1e-6, None)

    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # =====================================================
    # CLUSTER COLORS
    # =====================================================
    gene_colors = np.array([
        to_rgb(
            CLUSTER_COLORS.get(
                gene_clusters[g],
                "#B0B0B0"
            )
        )
        for g in ordered_gene_names
    ])

    cancer_colors = np.array([
        to_rgb(
            CLUSTER_COLORS.get(
                cancer_clusters[c],
                "#B0B0B0"
            )
        )
        for c in ordered_cancer_names
    ])

    # =====================================================
    # FIGURE LAYOUT
    # =====================================================
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2,
        3,

        height_ratios=[0.035, 0.965],

        width_ratios=[0.022, 0.92, 0.058],

        hspace=0.0,
        wspace=0.0
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # =====================================================
    # HEATMAP
    # =====================================================
    hm = sns.heatmap(
        mat_plot,

        ax=ax_main,

        cmap=cmap,

        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=(
            ordered_cancer_names
            if show_xticklabels
            else False
        ),

        yticklabels=(
            ordered_gene_names
            if show_yticklabels
            else False
        ),

        cbar=True,

        cbar_ax=ax_cbar,

        rasterized=True,
    )

    # =====================================================
    # X LABELS
    # =====================================================
    ax_main.set_xticklabels(
        ordered_cancer_names,
        fontsize=16,
        rotation=45,
        ha="center"
    )

    # =====================================================
    # Y LABELS
    # =====================================================
    if show_yticklabels:
        ax_main.set_yticklabels(
            ordered_gene_names,
            fontsize=6
        )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # =====================================================
    # COLORBAR STYLE
    # =====================================================
    cbar = hm.collections[0].colorbar

    cbar.ax.yaxis.set_major_formatter(
        LogFormatter()
    )

    cbar.ax.tick_params(
        labelsize=18
    )

    cbar.outline.set_visible(False)

    cbar.set_label(
        "Weighted Saliency",
        fontsize=18
    )

    # =====================================================
    # LEFT GENE CLUSTER BAR
    # =====================================================
    ax_left.imshow(
        gene_colors[:, None, :],
        aspect="auto"
    )

    ax_left.set_axis_off()

    ax_left.text(
        -1.8,
        0.5,

        "Genes",

        ha="center",
        va="center",

        rotation=90,

        transform=ax_left.transAxes,

        fontsize=22,
    )

    # =====================================================
    # TOP CANCER CLUSTER BAR
    # =====================================================
    ax_top.imshow(
        cancer_colors[None, :, :],
        aspect="auto"
    )

    ax_top.set_axis_off()

    ax_top.text(
        0.5,
        1.8,

        "Cancers",

        ha="center",
        va="center",

        transform=ax_top.transAxes,

        fontsize=22,
    )

    # =====================================================
    # ALIGN BARS
    # =====================================================
    fig.canvas.draw()

    main_pos = ax_main.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height
    ])

    # =====================================================
    # SHORTEN COLORBAR
    # =====================================================
    cbar_pos = ax_cbar.get_position()

    new_h = main_pos.height * 0.5

    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        ax_left.get_position().width,
        new_h
    ])

    # =====================================================
    # SAVE
    # =====================================================
    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.show()

    print(f"Heatmap saved to {save_path}")


def plot_gene_cancer_biclustering_heatmap_title(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(20, 16),
    cmap="Blues",
    show_xticklabels=True,
    show_yticklabels=False,
    title="Cancer–Gene Leiden Biclustering",
    vmax_percentile=99,
):
    """
    Plot Leiden-style gene–cancer biclustering heatmap with:

    - top cancer cluster color bar
    - left gene cluster color bar
    - log-scaled colorbar
    - large publication-style fonts
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm, to_rgb
    from matplotlib.ticker import LogFormatter
    from matplotlib.gridspec import GridSpec

    # =====================================================
    # GLOBAL FONT CONFIG
    # =====================================================

    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 26,
        "axes.labelsize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 18,
    })

    # =====================================================
    # OUTPUT DIR
    # =====================================================

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # EXTRACT NAMES
    # =====================================================

    gene_names = list(gene_cancer_biclustered.index)
    cancer_names = list(gene_cancer_biclustered.columns)

    # =====================================================
    # FILTER VALID CLUSTERS
    # =====================================================

    gene_names = [
        g for g in gene_names
        if g in gene_clusters
    ]

    cancer_names = [
        c for c in cancer_names
        if c in cancer_clusters
    ]

    # =====================================================
    # SUBSET MATRIX
    # =====================================================

    gene_cancer_biclustered = gene_cancer_biclustered.loc[
        gene_names,
        cancer_names
    ]

    # =====================================================
    # SORT BY CLUSTER
    # =====================================================

    gene_order = sorted(
        range(len(gene_names)),
        key=lambda i: gene_clusters[gene_names[i]]
    )

    cancer_order = sorted(
        range(len(cancer_names)),
        key=lambda i: cancer_clusters[cancer_names[i]]
    )

    # =====================================================
    # REORDER MATRIX
    # =====================================================

    reordered = gene_cancer_biclustered.iloc[
        gene_order,
        cancer_order
    ]

    ordered_gene_names = reordered.index.tolist()
    ordered_cancer_names = reordered.columns.tolist()

    mat = reordered.values

    # =====================================================
    # LOG SCALE
    # =====================================================

    mat_plot = np.clip(mat, 1e-6, None)

    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # =====================================================
    # CLUSTER COLORS
    # =====================================================

    gene_colors = np.array([
        to_rgb(
            CLUSTER_COLORS.get(
                gene_clusters[g],
                "#B0B0B0"
            )
        )
        for g in ordered_gene_names
    ])

    cancer_colors = np.array([
        to_rgb(
            CLUSTER_COLORS.get(
                cancer_clusters[c],
                "#B0B0B0"
            )
        )
        for c in ordered_cancer_names
    ])

    # =====================================================
    # FIGURE LAYOUT
    # =====================================================

    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2,
        3,
        height_ratios=[0.04, 0.96],
        width_ratios=[0.03, 0.91, 0.06],
        hspace=0.0,
        wspace=0.0
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # =====================================================
    # HEATMAP
    # =====================================================

    hm = sns.heatmap(
        mat_plot,

        ax=ax_main,

        cmap=cmap,

        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=False,

        yticklabels=(
            ordered_gene_names
            if show_yticklabels
            else False
        ),

        cbar=True,

        cbar_ax=ax_cbar,

        rasterized=True,
    )

    # =====================================================
    # FORCE XTICKS TO CELL CENTERS
    # =====================================================

    ax_main.set_xticks(
        np.arange(len(ordered_cancer_names)) + 0.5
    )

    ax_main.set_xticklabels(
        ordered_cancer_names,
        fontsize=44,
        rotation=90,
        ha="center",
        va="top",
        rotation_mode="default"
    )

    ax_main.tick_params(
        axis="x",
        pad=8
    )


    # =====================================================
    # COLORBAR STYLE
    # =====================================================

    cbar = hm.collections[0].colorbar

    cbar.ax.yaxis.set_major_formatter(
        LogFormatter()
    )

    cbar.ax.tick_params(
        labelsize=22
    )

    cbar.outline.set_visible(False)

    cbar.set_label(
        "",
        fontsize=34,
        # # fontweight="bold",
        labelpad=20
    )
    # =====================================================
    # LEFT GENE CLUSTER BAR
    # =====================================================

    ax_left.imshow(
        gene_colors[:, None, :],
        aspect="auto"
    )

    ax_left.set_axis_off()

    ax_left.text(
        -1.0,
        0.5,

        "Genes",

        ha="center",
        va="center",

        rotation=90,

        transform=ax_left.transAxes,

        fontsize=48,
        # # fontweight="bold",
    )

    # =====================================================
    # TOP CANCER CLUSTER BAR
    # =====================================================

    ax_top.imshow(
        cancer_colors[None, :, :],
        aspect="auto"
    )

    ax_top.set_axis_off()

    ax_top.text(
        0.5,
        2.2,

        # "Caners",
        "",

        ha="center",
        va="center",

        transform=ax_top.transAxes,

        fontsize=48,
        # # fontweight="bold",
    )

    # =====================================================
    # ALIGN BARS
    # =====================================================

    fig.canvas.draw()

    main_pos = ax_main.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height
    ])

    # =====================================================
    # SHORTEN + THIN COLORBAR
    # =====================================================

    cbar_pos = ax_cbar.get_position()

    new_h = main_pos.height * 0.55

    new_w = 0.012

    ax_cbar.set_position([

        cbar_pos.x0 + 0.01,

        main_pos.y0 + (
            main_pos.height - new_h
        ) / 2,

        new_w,

        new_h
    ])


    # =====================================================
    # SAVE
    # =====================================================

    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05
    )

    plt.show()
    plt.close()

    print(f"[SAVED] {save_path}")

def plot_gene_cancer_biclustering_heatmap(
    gene_cancer_biclustered,
    gene_clusters,
    cancer_clusters,
    output_dir,
    filename="gene_cancer_leiden_bipartite.png",
    figsize=(20, 16),
    cmap="Blues",
    show_xticklabels=True,
    show_yticklabels=False,
    # title="Cancer–Gene Leiden Biclustering",
    title="",
    vmax_percentile=99,
    dpi=300,
    **kwargs
):
    """
    Publication-style Leiden biclustering heatmap.

    Features:
    - top cancer cluster color bar
    - left gene cluster color bar
    - logarithmic color scaling
    - publication-style fonts
    - configurable title
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    from matplotlib.colors import LogNorm, to_rgb
    from matplotlib.ticker import LogFormatter
    from matplotlib.gridspec import GridSpec

    # =====================================================
    # GLOBAL FONT CONFIG
    # =====================================================

    plt.rcParams.update({

        "font.size": 22,

        "axes.titlesize": 40,

        "axes.labelsize": 30,

        "xtick.labelsize": 28,

        "ytick.labelsize": 20,
    })

    # =====================================================
    # OUTPUT DIR
    # =====================================================

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # =====================================================
    # EXTRACT NAMES
    # =====================================================

    gene_names = list(
        gene_cancer_biclustered.index
    )

    cancer_names = list(
        gene_cancer_biclustered.columns
    )

    # =====================================================
    # FILTER VALID CLUSTERS
    # =====================================================

    gene_names = [

        g for g in gene_names

        if g in gene_clusters
    ]

    cancer_names = [

        c for c in cancer_names

        if c in cancer_clusters
    ]

    # =====================================================
    # SUBSET MATRIX
    # =====================================================

    gene_cancer_biclustered = (
        gene_cancer_biclustered.loc[
            gene_names,
            cancer_names
        ]
    )

    # =====================================================
    # SORT BY CLUSTER
    # =====================================================

    gene_order = sorted(

        range(len(gene_names)),

        key=lambda i:
            gene_clusters[gene_names[i]]
    )

    cancer_order = sorted(

        range(len(cancer_names)),

        key=lambda i:
            cancer_clusters[cancer_names[i]]
    )

    # =====================================================
    # REORDER MATRIX
    # =====================================================

    reordered = gene_cancer_biclustered.iloc[
        gene_order,
        cancer_order
    ]

    ordered_gene_names = reordered.index.tolist()

    ordered_cancer_names = reordered.columns.tolist()

    mat = reordered.values.astype(float)

    # =====================================================
    # REMOVE INVALID VALUES
    # =====================================================

    mat = np.nan_to_num(
        mat,
        nan=1e-6,
        posinf=1e-6,
        neginf=1e-6
    )

    mat = np.clip(
        mat,
        1e-6,
        None
    )

    # =====================================================
    # LOG SCALE RANGE
    # =====================================================

    vmin = np.percentile(
        mat,
        5
    )

    vmax = np.percentile(
        mat,
        vmax_percentile
    )

    if vmax <= vmin:
        vmax = vmin * 10

    # =====================================================
    # DEFAULT CLUSTER COLORS
    # =====================================================

    DEFAULT_CLUSTER_COLORS = {

        0: '#0077B6',
        1: '#0000FF',
        2: '#00B4D8',
        3: '#48EAC4',
        4: '#F1C0E8',
        5: '#B9FBC0',
        6: '#32CD32',
        7: '#bee1e6',
        8: '#8A2BE2',
        9: '#E377C2'
    }

    # =====================================================
    # USE GLOBAL COLORS IF AVAILABLE
    # =====================================================

    try:
        cluster_colors = CLUSTER_COLORS
    except:
        cluster_colors = DEFAULT_CLUSTER_COLORS

    # =====================================================
    # GENE COLORS
    # =====================================================

    gene_colors = np.array([

        to_rgb(

            cluster_colors.get(
                gene_clusters[g],
                "#B0B0B0"
            )
        )

        for g in ordered_gene_names
    ])

    # =====================================================
    # CANCER COLORS
    # =====================================================

    cancer_colors = np.array([

        to_rgb(

            cluster_colors.get(
                cancer_clusters[c],
                "#B0B0B0"
            )
        )

        for c in ordered_cancer_names
    ])

    # =====================================================
    # FIGURE LAYOUT
    # =====================================================

    fig = plt.figure(
        figsize=figsize
    )

    gs = GridSpec(

        2,
        3,

        height_ratios=[0.045, 0.955],

        width_ratios=[0.03, 0.92, 0.05],

        hspace=0.0,

        wspace=0.0
    )

    ax_top  = fig.add_subplot(gs[0, 1])

    ax_left = fig.add_subplot(gs[1, 0])

    ax_main = fig.add_subplot(gs[1, 1])

    ax_cbar = fig.add_subplot(gs[1, 2])

    # =====================================================
    # MAIN HEATMAP
    # =====================================================

    hm = sns.heatmap(

        mat,

        ax=ax_main,

        cmap=cmap,

        norm=LogNorm(
            vmin=vmin,
            vmax=vmax
        ),

        xticklabels=False,

        yticklabels=(
            ordered_gene_names
            if show_yticklabels
            else False
        ),

        cbar=True,

        cbar_ax=ax_cbar,

        rasterized=True,
    )

    # =====================================================
    # TITLE
    # =====================================================

    ax_main.set_title(

        title,

        fontsize=50,

        pad=28
    )

    # =====================================================
    # XTICKS
    # =====================================================

    if show_xticklabels:

        ax_main.set_xticks(
            np.arange(len(ordered_cancer_names)) + 0.5
        )

        ax_main.set_xticklabels(

            ordered_cancer_names,

            fontsize=34,

            rotation=90,

            ha="center",

            va="top"
        )

        ax_main.tick_params(
            axis="x",
            pad=8
        )

    # =====================================================
    # REMOVE AXIS LABELS
    # =====================================================

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # =====================================================
    # COLORBAR STYLE
    # =====================================================

    cbar = hm.collections[0].colorbar

    cbar.ax.yaxis.set_major_formatter(
        LogFormatter()
    )

    cbar.ax.tick_params(
        labelsize=24
    )

    cbar.outline.set_visible(False)

    # =====================================================
    # LEFT GENE CLUSTER BAR
    # =====================================================

    ax_left.imshow(

        gene_colors[:, None, :],

        aspect="auto"
    )

    ax_left.set_axis_off()

    ax_left.text(

        -0.9,

        0.5,

        "Genes",

        ha="center",

        va="center",

        rotation=90,

        transform=ax_left.transAxes,

        fontsize=44
    )

    # =====================================================
    # TOP CANCER CLUSTER BAR
    # =====================================================

    ax_top.imshow(

        cancer_colors[None, :, :],

        aspect="auto"
    )

    ax_top.set_axis_off()

    # =====================================================
    # ALIGN COLOR BARS
    # =====================================================

    fig.canvas.draw()

    main_pos = ax_main.get_position()

    ax_left.set_position([

        ax_left.get_position().x0,

        main_pos.y0,

        ax_left.get_position().width,

        main_pos.height
    ])

    ax_top.set_position([

        main_pos.x0,

        ax_top.get_position().y0,

        main_pos.width,

        ax_top.get_position().height
    ])

    # =====================================================
    # SHORTER COLORBAR
    # =====================================================

    cbar_pos = ax_cbar.get_position()

    new_h = main_pos.height * 0.55

    new_w = 0.012

    ax_cbar.set_position([

        cbar_pos.x0 + 0.01,

        main_pos.y0 + (
            main_pos.height - new_h
        ) / 2,

        new_w,

        new_h
    ])

    # =====================================================
    # SAVE
    # =====================================================

    save_path = os.path.join(
        output_dir,
        filename
    )

    plt.savefig(

        save_path,

        dpi=dpi,

        bbox_inches="tight",

        pad_inches=0.05
    )

    print(f"\n[SAVED] {save_path}")

    plt.show()

    plt.close()
    
def plot_leiden_cancer_gene_heatmap(
    gene_cancer_weighted,
    gene_names,
    cancer_names,
    gene_clusters,
    cancer_clusters,
    output_path,
    args,
    vmax_percentile=99,
    figsize=(16, 12),
):
    """
    Leiden-ordered cancer–gene heatmap with cluster bars and log scaling.
    X-axis: cancers
    Y-axis: genes
    """

    # -------------------------
    # Filter to genes/cancers with clusters
    # -------------------------
    gene_names_filtered = [g for g in gene_names if g in gene_clusters]
    cancer_names_filtered = [c for c in cancer_names if c in cancer_clusters]

    # Subset matrix (genes x cancers)
    mat = gene_cancer_weighted.loc[
        gene_names_filtered, cancer_names_filtered
    ].values

    # -------------------------
    # Sort by Leiden cluster
    # -------------------------
    gene_order = sorted(
        range(len(gene_names_filtered)),
        key=lambda i: gene_clusters[gene_names_filtered[i]]
    )
    cancer_order = sorted(
        range(len(cancer_names_filtered)),
        key=lambda i: cancer_clusters[cancer_names_filtered[i]]
    )

    # Reorder and transpose → (cancers x genes)
    mat = mat[gene_order][:, cancer_order].T

    # -------------------------
    # Output filename
    # -------------------------
    leiden_heatmap_file = (
        f"leiden_cancer_gene_heatmap"
        f"_dim{getattr(args, 'in_feats', 'NA')}"
        f"_lay{getattr(args, 'num_layers', 'NA')}"
        f"_hid{getattr(args, 'hidden_feats', 'NA')}"
        f"_epo{getattr(args, 'num_epochs', 'NA')}.png"
    )

    # -------------------------
    # Log scaling
    # -------------------------
    mat_plot = np.clip(mat, 1e-6, None)
    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # -------------------------
    # Cluster color bars
    # -------------------------
    gene_colors = np.array([
        to_rgb(CLUSTER_COLORS[gene_clusters[gene_names_filtered[i]]])
        for i in gene_order
    ])
    cancer_colors = np.array([
        to_rgb(CLUSTER_COLORS[cancer_clusters[cancer_names_filtered[i]]])
        for i in cancer_order
    ])

    # -------------------------
    # Layout
    # -------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        2, 3,
        width_ratios=[0.022, 0.92, 0.058],
        height_ratios=[0.03, 0.97],
        wspace=0.0,
        hspace=0.0,
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # -------------------------
    # Heatmap
    # -------------------------
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap="Blues",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        xticklabels=False,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    cbar = hm.collections[0].colorbar
    cbar.set_label("Saliency", fontsize=22, labelpad=14)
    cbar.ax.yaxis.set_major_formatter(LogFormatter())
    cbar.ax.tick_params(labelsize=18)

    # -------------------------
    # Cluster bars
    # -------------------------
    # Top bar → cancers (x-axis)
    ax_top.imshow(cancer_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    # Left bar → genes (y-axis)
    ax_left.imshow(gene_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    # -------------------------
    # Axis labels
    # -------------------------
    ax_top.text(
        0.5, 1.9, "Cancers",
        ha="center", va="bottom",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    ax_left.text(
        -1.8, 0.5, "Genes",
        ha="center", va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # -------------------------
    # Alignment & colorbar resize
    # -------------------------
    fig.canvas.draw()
    main_pos = ax_main.get_position()
    cbar_pos = ax_cbar.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height,
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height,
    ])

    new_h = main_pos.height * 0.5
    new_w = cbar_pos.width * 0.5
    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        new_w,
        new_h,
    ])

    # -------------------------
    # Save
    # -------------------------
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, leiden_heatmap_file),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()



def plot_leiden_gene_cancer_heatmap(
    gene_cancer_weighted,
    gene_names,
    cancer_names,
    gene_clusters,
    cancer_clusters,
    output_path,
    args,
    vmax_percentile=99,
    figsize=(12, 12),
):
    """
    Leiden-ordered gene–cancer heatmap with
    - left gene-cluster bar
    - top cancer-cluster bar
    - log-scaled colorbar
    """

    import os
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, to_rgb
    from matplotlib.ticker import LogFormatter
    from matplotlib.gridspec import GridSpec

    # -------------------------
    # Filter valid genes / cancers
    # -------------------------
    gene_names_filtered = [g for g in gene_names if g in gene_clusters]
    cancer_names_filtered = [c for c in cancer_names if c in cancer_clusters]

    mat = gene_cancer_weighted.loc[
        gene_names_filtered, cancer_names_filtered
    ].values

    # -------------------------
    # Sort by Leiden cluster
    # -------------------------
    gene_order = sorted(
        range(len(gene_names_filtered)),
        key=lambda i: gene_clusters[gene_names_filtered[i]]
    )
    cancer_order = sorted(
        range(len(cancer_names_filtered)),
        key=lambda i: cancer_clusters[cancer_names_filtered[i]]
    )

    mat = mat[gene_order][:, cancer_order]

    ordered_cancer_names = [cancer_names_filtered[i] for i in cancer_order]

    # -------------------------
    # Filename
    # -------------------------
    leiden_heatmap_file = (
        f"leiden_gene_cancer_heatmap"
        f"_dim{getattr(args, 'in_feats', 'NA')}"
        f"_lay{getattr(args, 'num_layers', 'NA')}"
        f"_hid{getattr(args, 'hidden_feats', 'NA')}"
        f"_epo{getattr(args, 'num_epochs', 'NA')}.png"
    )

    # -------------------------
    # Log scaling
    # -------------------------
    mat_plot = np.clip(mat, 1e-6, None)
    vmin = np.percentile(mat_plot, 5)
    vmax = np.percentile(mat_plot, vmax_percentile)

    # -------------------------
    # Cluster colors
    # -------------------------
    gene_colors = np.array([
        to_rgb(CLUSTER_COLORS[gene_clusters[gene_names_filtered[i]]])
        for i in gene_order
    ])

    cancer_colors = np.array([
        to_rgb(CLUSTER_COLORS[cancer_clusters[cancer_names_filtered[i]]])
        for i in cancer_order
    ])

    # -------------------------
    # Layout (top + left bars)
    # -------------------------
    fig = plt.figure(figsize=figsize)

    gs = GridSpec(
        2, 3,
        height_ratios=[0.035, 0.965],     # top bar, heatmap
        width_ratios=[0.022, 0.92, 0.058],  # left bar, heatmap, colorbar
        hspace=0.0,
        wspace=0.0
    )

    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])
    ax_cbar = fig.add_subplot(gs[1, 2])

    # -------------------------
    # Heatmap
    # -------------------------
    hm = sns.heatmap(
        mat_plot,
        ax=ax_main,
        cmap="Blues",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        xticklabels=ordered_cancer_names,
        yticklabels=False,
        cbar=True,
        cbar_ax=ax_cbar,
        rasterized=True,
    )

    # ax_main.set_xticklabels(
    #     ax_main.get_xticklabels(),
    #     fontsize=16,
    #     rotation=45,
    #     ha="right"
    # )

    # -------------------------
    # Custom centered x tick labels
    # -------------------------

    n_cols = len(ordered_cancer_names)

    ax_main.set_xticks(
        np.arange(n_cols) + 0.5
    )

    ax_main.set_xticklabels(
        ordered_cancer_names,
        fontsize=16,
        rotation=90,
        ha="center",
        va="top"
    )

    ax_main.tick_params(
        axis="x",
        length=0
    )

    ax_main.set_xlabel("")
    ax_main.set_ylabel("")

    # -------------------------
    # Colorbar styling
    # -------------------------
    cbar = hm.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(LogFormatter())
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)

    # -------------------------
    # Left gene cluster bar
    # -------------------------
    ax_left.imshow(gene_colors[:, None, :], aspect="auto")
    ax_left.set_axis_off()

    ax_left.text(
        -1.8, 0.5,
        "Genes",
        ha="center",
        va="center",
        rotation=90,
        transform=ax_left.transAxes,
        fontsize=22,
    )

    # -------------------------
    # Top cancer cluster bar
    # -------------------------
    ax_top.imshow(cancer_colors[None, :, :], aspect="auto")
    ax_top.set_axis_off()

    ax_top.text(
        0.5, 1.8,
        "Cancers",
        ha="center",
        va="center",
        transform=ax_top.transAxes,
        fontsize=22,
    )

    # -------------------------
    # Align bars & shorten colorbar
    # -------------------------
    fig.canvas.draw()
    main_pos = ax_main.get_position()

    ax_left.set_position([
        ax_left.get_position().x0,
        main_pos.y0,
        ax_left.get_position().width,
        main_pos.height
    ])

    ax_top.set_position([
        main_pos.x0,
        ax_top.get_position().y0,
        main_pos.width,
        ax_top.get_position().height
    ])

    cbar_pos = ax_cbar.get_position()
    new_h = main_pos.height * 0.5
    ax_cbar.set_position([
        cbar_pos.x0 + 0.02,
        main_pos.y0 + (main_pos.height - new_h) / 2,
        ax_left.get_position().width,
        new_h
    ])

    # -------------------------
    # Save
    # -------------------------
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(
        os.path.join(output_path, leiden_heatmap_file),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.show()

def build_gene_cancer_matrix(
    saliency_dict,
    top_k=300,
    normalize=True
):
    """
    saliency_dict:
        {
          cancer_name: {gene: saliency_score}
        }
    """

    # Collect top genes across cancers
    all_genes = set()
    for cancer, scores in saliency_dict.items():
        top_genes = sorted(scores.items(), key=lambda x: -abs(x[1]))[:top_k]
        all_genes.update([g for g, _ in top_genes])

    all_genes = sorted(list(all_genes))
    cancers = sorted(list(saliency_dict.keys()))

    mat = np.zeros((len(all_genes), len(cancers)))

    gene2idx = {g: i for i, g in enumerate(all_genes)}
    cancer2idx = {c: j for j, c in enumerate(cancers)}

    for cancer, scores in saliency_dict.items():
        for gene, val in scores.items():
            if gene in gene2idx:
                mat[gene2idx[gene], cancer2idx[cancer]] = val

    df = pd.DataFrame(mat, index=all_genes, columns=cancers)

    if normalize:
        df = (df - df.mean()) / (df.std() + 1e-8)

    return df

from sklearn.cluster import SpectralBiclustering

def gene_cancer_biclustering(
    df,
    n_gene_clusters=6,
    n_cancer_clusters=4,
    random_state=0
):
    model = SpectralBiclustering(
        n_clusters=(n_gene_clusters, n_cancer_clusters),
        method="log",
        random_state=random_state
    )

    model.fit(df.values)

    gene_order = np.argsort(model.row_labels_)
    cancer_order = np.argsort(model.column_labels_)

    biclustered_df = df.iloc[gene_order, cancer_order]

    return biclustered_df, model


def plot_gene_cancer_heatmap(
    df,
    figsize=(10, 12),
    vmax=3,
    vmin=-3,
    cmap="coolwarm",
    output=None
):
    plt.figure(figsize=figsize)

    sns.heatmap(
        df,
        cmap=cmap,
        center=0,
        vmax=vmax,
        vmin=vmin,
        xticklabels=True,
        yticklabels=False,
        cbar_kws={"label": "Gene Saliency"}
    )

    plt.xlabel("Cancer Type")
    plt.ylabel("Genes")
    plt.title("Gene–Cancer Saliency Biclustering")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300)
    plt.show()



def plot_leiden_saliency_heatmap(saliency_df, args, output_dir):
    df = saliency_df.sort_values("Leiden_Cluster")

    plt.figure(figsize=(6, 10))
    sns.heatmap(
        df[["Cancer_Driver_Saliency"]],
        cmap="Reds",
        cbar_kws={"label": "Saliency"}
    )

    plt.ylabel("Genes (Leiden clustered)")
    plt.xlabel("Cancer Driver Task")
    plt.title(
        f"Leiden Saliency Heatmap\n"
        f"{args.model_type} | {args.net_type}"
    )

    out_path = os.path.join(
        output_dir,
        f"leiden_gene_cancer_saliency_{args.model_type}_{args.net_type}.pdf"
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Leiden saliency heatmap saved to {out_path}")

def compute_gene_saliency_(model, graph, features, target_nodes):
    model.eval()
    features.requires_grad_(True)

    logits = model(graph, features).squeeze()
    scores = torch.sigmoid(logits)

    # Focus on predicted cancer drivers (or all test nodes)
    target_score = scores[target_nodes].sum()
    target_score.backward()

    saliency = features.grad.abs().detach().cpu()
    return saliency

def compute_gene_saliency(model, graph, features, mask):
    """
    Node-level saliency for gene classification
    """
    model.eval()
    features = features.clone().detach().requires_grad_(True)

    logits = model(graph, features).squeeze()
    probs = torch.sigmoid(logits)

    target = probs[mask].sum()
    target.backward()

    saliency = features.grad.abs().detach().cpu()
    gene_saliency = saliency.sum(dim=1).numpy()

    return gene_saliency

def select_topk_genes(gene_saliency, nodes, top_k=1000):
    gene_names = list(nodes.keys())

    # df = pd.DataFrame({
    #     "Gene": gene_names,
    #     "Saliency": gene_saliency
    # })

    # import pandas as pd
    # import numpy as np
    # import torch

    # If saliency is a torch tensor
    if torch.is_tensor(saliency_scores):
        saliency_scores = saliency_scores.detach().cpu().numpy()

    df = pd.DataFrame({
        "Gene": pd.Series(gene_names),
        "Saliency": pd.Series(saliency_scores)
    })

    df = df.sort_values(
        "Saliency", ascending=False
    ).head(top_k).reset_index(drop=True)

    return df

def get_topk_saliency_df(gene_saliency_score, nodes, top_k=1000):
    gene_names = list(nodes.keys())

    # saliency_df = pd.DataFrame({
    #     "Gene": gene_names,
    #     "Saliency": gene_saliency_score
    # })

    # import pandas as pd
    # import numpy as np
    # import torch

    # If saliency is a torch tensor
    if torch.is_tensor(saliency_scores):
        saliency_scores = saliency_scores.detach().cpu().numpy()

    saliency_df = pd.DataFrame({
        "Gene": pd.Series(gene_names),
        "Saliency": pd.Series(saliency_scores)
    })

    saliency_df = saliency_df.sort_values(
        by="Saliency",
        ascending=False
    ).head(top_k).reset_index(drop=True)

    return saliency_df


def leiden_cluster_gene_saliency(
    saliency_df,
    nodes,
    similarity_threshold=0.3,
    resolution=1.0
):
    X = saliency_df[["Saliency"]].values
    sim = cosine_similarity(X)

    edges, weights = [], []
    n = sim.shape[0]

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                edges.append((i, j))
                weights.append(sim[i, j])

    g = ig.Graph(edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution
    )

    # saliency_df["LeidenCluster"] = partition.membership
    leiden_map = dict(zip(nodes, partition.membership))
    saliency_df["LeidenCluster"] = saliency_df["Gene"].map(leiden_map).fillna(-1).astype(int)
    return saliency_df

def plot_leiden_gene_saliency_heatmap(
    saliency_df,
    args,
    output_dir
):
    df = saliency_df.sort_values("LeidenCluster")

    heatmap_data = df[["Saliency"]].values

    plt.figure(figsize=(5, 10))
    sns.heatmap(
        heatmap_data,
        cmap="Reds",
        yticklabels=False,
        cbar_kws={"label": "Gene Saliency"}
    )

    plt.xlabel("Cancer Driver Prediction")
    plt.ylabel("Top-K Genes (Leiden clustered)")
    plt.title(
        f"Leiden Gene Saliency Heatmap\n"
        f"{args.model_type} | {args.net_type}"
    )

    out_path = os.path.join(
        output_dir,
        f"leiden_gene_saliency_topK_{len(df)}_"
        f"{args.model_type}_{args.net_type}.pdf"
    )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"✔ Leiden saliency heatmap saved to:\n{out_path}")

def load_gene_set(file_path):
    """
    Load a gene list from a file and return as a set.
    
    Args:
    - file_path: Path to the file containing genes, one per line.
    
    Returns:
    - Set of gene names.
    """
    with open(file_path, 'r') as f:
        return set(line.strip() for line in f)

def save_predictions_to_csv(predicted_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save the predicted genes with their sources to a CSV file.
    
    Args:
    - predicted_genes: List of tuples (gene, score, sources) to save.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    os.makedirs(output_dir, exist_ok=True)
    predicted_genes_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_driver_genes_epo{num_epochs}.csv')
    df_predictions = pd.DataFrame(predicted_genes, columns=["Gene", "Score", "Confirmed Sources"])
    df_predictions.to_csv(predicted_genes_csv_path, index=False)
    print(f"Predicted driver genes with confirmed sources saved to {predicted_genes_csv_path}")

def save_confirmed_predictions_to_csv(confirmed_predictions, output_dir, model_type, net_type, num_epochs):
    """
    Save confirmed predicted genes to a CSV file.
    
    Args:
    - confirmed_predictions: List of tuples (gene, score, sources).
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    confirmed_predictions_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_confirmed_predicted_genes_epo{num_epochs}.csv')
    df_confirmed = pd.DataFrame(confirmed_predictions, columns=["Gene", "Score", "Source"])
    df_confirmed.to_csv(confirmed_predictions_csv_path, index=False)
    print(f"Confirmed predicted genes saved to {confirmed_predictions_csv_path}")

def save_predicted_known_drivers(predicted_driver_genes, output_dir, model_type, net_type, num_epochs):
    """
    Save predicted known cancer driver genes to a CSV file.
    
    Args:
    - predicted_driver_genes: List of predicted cancer driver genes.
    - output_dir: Directory to save the CSV file.
    - model_type, net_type, num_epochs: For naming the output file.
    """
    predicted_drivers_csv_path = os.path.join(output_dir, f'{model_type}_{net_type}_predicted_known_drivers_epo{num_epochs}.csv')
    df = pd.DataFrame(predicted_driver_genes, columns=["Gene"])
    df.to_csv(predicted_drivers_csv_path, index=False)
    print(f"Predicted known driver genes saved to {predicted_drivers_csv_path}")

def process_predictions(ranking, args, drivers_file_path, oncokb_file_path, depmap_file_path, ncg_file_path, intogen_file_path, node_names, non_labeled_nodes):
    """
    Process and save the predicted driver genes, confirmed sources, and known drivers.
    
    Args:
    - ranking: List of tuples (gene, score) representing ranked predictions.
    - args: Argument object containing model and network type, and score threshold.
    - drivers_file_path, oncokb_file_path, depmap_file_path, ncg_file_path, intogen_file_path: Paths to the confirmation gene files.
    - node_names, non_labeled_nodes: Information about node names and indices for matching.
    """
    # Load data from the confirmation files
    oncokb_genes = load_gene_set(oncokb_file_path)
    depmap_genes = load_gene_set(depmap_file_path)
    ncg_genes = load_gene_set(ncg_file_path)
    intogen_genes = load_gene_set(intogen_file_path)

    # Threshold for the score
    score_threshold = args.score_threshold

    confirmed_predictions = []
    predicted_genes = []

    for node, score in ranking:
        if score >= score_threshold:
            sources = []  # Accumulate sources confirming the gene
            if node in oncokb_genes:
                sources.append("OncoKB")
            if node in depmap_genes:
                sources.append("DepMap")
            if node in ncg_genes:
                sources.append("NCG")
            if node in intogen_genes:
                sources.append("IntOGen")
            if sources:  # If the gene is confirmed by at least one source
                confirmed_predictions.append((node, score, ", ".join(sources)))
            predicted_genes.append((node, score, ", ".join(sources) if sources else ""))

    # Save predictions to a CSV file
    save_predictions_to_csv(predicted_genes, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)
    save_confirmed_predictions_to_csv(confirmed_predictions, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

    # Load known cancer driver genes
    with open(drivers_file_path, 'r') as f:
        known_drivers = set(line.strip() for line in f)

    # Collect predicted cancer driver genes that match the known drivers
    predicted_driver_genes = [node_names[i] for i in non_labeled_nodes if node_names[i] in known_drivers]

    # Save the predicted known cancer driver genes to a CSV file
    save_predicted_known_drivers(predicted_driver_genes, 'results/gene_prediction/', args.model_type, args.net_type, args.num_epochs)

def save_overall_metrics(total_time, average_time_per_epoch, average_cpu_usage, average_gpu_usage, args, output_dir):
    """
    Save the overall performance metrics to a CSV file.

    Args:
    - total_time: Total training time in seconds.
    - average_time_per_epoch: Average time per epoch in seconds.
    - average_cpu_usage: Average CPU usage in MB.
    - average_gpu_usage: Average GPU usage in MB.
    - args: Argument object containing model and network type.
    - output_dir: The directory where the results will be saved.
    """
    # Save overall metrics
    df_overall_metrics = pd.DataFrame([{
        "Model Type": args.model_type,
        "Total Time": f"{total_time:.4f}s",
        "Average Time per Epoch": f"{average_time_per_epoch:.4f}s",
        "Average CPU Usage (MB)": f"{average_cpu_usage:.2f}",
        "Average GPU Usage (MB)": f"{average_gpu_usage:.2f}"
    }])
    
    # Define path to save the CSV
    overall_metrics_csv_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_overall_performance_epo{args.num_epochs}.csv')
    
    # Save to CSV
    df_overall_metrics.to_csv(overall_metrics_csv_path, index=False)
    print(f"Overall performance metrics saved to {overall_metrics_csv_path}")

def calculate_and_save_prediction_stats(non_labeled_nodes, labels, node_names, scores, args):
    """
    Calculate prediction statistics and save them to a CSV file.

    Parameters:
    - non_labeled_nodes: List of nodes without labels
    - labels: List of ground truth labels for the nodes
    - node_names: List of node names corresponding to the nodes
    - scores: List of predicted scores for the nodes
    - args: Arguments containing model and network type, score threshold, and number of epochs
    """
    # Calculate statistics
    non_labeled_nodes_count = len(non_labeled_nodes)
    ground_truth_driver_nodes = [i for i, label in enumerate(labels) if label == 1]
    ground_truth_non_driver_nodes = [i for i, label in enumerate(labels) if label == 0]

    predicted_driver_nodes = [node_names[i] for i in non_labeled_nodes if scores[i] >= args.score_threshold]

    # Prepare data to save to CSV
    stats_output_file = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_prediction_stats_{args.num_epochs}.csv')
    
    with open(stats_output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Non-Labeled Nodes Count', 'Driver Genes', 'Non-Driver Genes', 'Total Testing Nodes', 'Predicted Driver Genes'])
        csvwriter.writerow([
            non_labeled_nodes_count,
            len(ground_truth_driver_nodes),
            len(ground_truth_non_driver_nodes),
            len(ground_truth_driver_nodes) + len(ground_truth_non_driver_nodes),
            len(predicted_driver_nodes)
        ])

    print(f"Prediction statistics saved to {stats_output_file}")

def plot_degree_distributions(sorted_degree_counts_above, sorted_degree_counts_below, args, output_dir):
    """
    Generates a box plot comparing interaction degrees of PCGs vs. Other Genes with KCGs.
    
    Parameters:
    - sorted_degree_counts_above: List of degrees for PCGs.
    - sorted_degree_counts_below: List of degrees for other genes.
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """

    print("Generating box plot for degree distributions...")

    degree_data = [sorted_degree_counts_above, sorted_degree_counts_below]

    plt.figure(figsize=(3, 4))

    # Create the box plot
    boxplot = plt.boxplot(
        degree_data,
        vert=True,
        patch_artist=True,  # Allows customization of box color
        flierprops=dict(marker='o', markerfacecolor='grey', markeredgecolor='grey', markersize=5, alpha=0.2),  # Outliers
        boxprops=dict(color='black'),  # Box border color
        medianprops=dict(color='blue', linewidth=2),  # Median line style
        whiskerprops=dict(color='black', linewidth=1.5),  # Whiskers
        capprops=dict(color='black', linewidth=1.5)  # Caps
    )

    # Customize frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # Remove top frame line
    ax.spines['right'].set_visible(False)  # Remove right frame line

    # X-axis labels
    plt.xticks([1, 2], ['PCGs', 'Other'], fontsize=8)  
    plt.yticks(fontsize=8)
    plt.ylabel('Interaction Degrees with KCGs', fontsize=10, labelpad=10) 

    # Assign different colors to box plots
    colors = ['green', 'skyblue']
    for patch, color in zip(boxplot['boxes'], colors):
        patch.set_facecolor(color)

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    output_plot_path = os.path.join(output_dir, f'{args.model_type}_{args.net_type}_degree_distributions_epo{args.num_epochs}.png')
    plt.savefig(output_plot_path, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    print(f"Box plot saved to {output_plot_path}")

def generate_kde_and_curves(logits, node_names, degree_counts_above, degree_counts_below, labels, train_mask, args):
    """
    Generates KDE plot comparing ACGNN score ranks with KCG interaction ranks, 
    computes Spearman correlation, and saves the KDE plot.
    Also computes and saves ROC and PR curves.

    Parameters:
    - logits: Tensor of model outputs before applying sigmoid.
    - node_names: List of node names.
    - degree_counts_above, degree_counts_below: Dictionaries mapping nodes to degree counts.
    - labels: Ground truth labels.
    - train_mask: Boolean mask indicating training samples.
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save plots.
    """

    print("Preparing data for KDE plot...")
    
    # Convert logits to probabilities
    scores = torch.sigmoid(logits).cpu().numpy()

    # Compute degree ranks
    degrees = [
        degree_counts_above.get(node_names[i], 0) +
        degree_counts_below.get(node_names[i], 0)
        for i in range(len(node_names))
    ]

    # Create DataFrame
    plot_data = pd.DataFrame({
        "Prob_pos_ranked": pd.Series(scores).rank(),
        "Degree_ranked": pd.Series(degrees).rank()
    })

    # KDE Plot
    print("Generating KDE plot...")
    plt.figure(figsize=(4, 4))
    sns.kdeplot(
        x=plot_data["Prob_pos_ranked"],
        y=plot_data["Degree_ranked"],
        cmap="Reds", fill=True,
        alpha=0.7, levels=50, thresh=0.05
    )

    # Spearman correlation
    correlation, p_value = scipy.stats.spearmanr(
        plot_data["Prob_pos_ranked"], plot_data["Degree_ranked"]
    )
    
    # print('correlation = ', correlation)

    # Labels and formatting
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('ACGNN score rank', fontsize=10, labelpad=10)
    plt.ylabel('KCG interaction rank', fontsize=12, labelpad=10)

    # Add correlation text
    legend_text = f"Spearman R: {correlation:.4f}\nP-value: {p_value:.3e}"
    plt.text(
        0.05, 0.95, legend_text,
        fontsize=8, transform=plt.gca().transAxes,
        verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    kde_output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_kde_plot_epo{args.num_epochs}.png')
    plt.savefig(kde_output_path, bbox_inches='tight')
    print(f"KDE plot saved to {kde_output_path}")

    plt.tight_layout()
    plt.show()

    # Extract labeled scores and labels
    labeled_scores = scores[train_mask.cpu().numpy()]
    labeled_labels = labels[train_mask.cpu().numpy()]

    # Convert to NumPy arrays if necessary
    labeled_scores_np = labeled_scores.cpu().detach().numpy() if isinstance(labeled_scores, torch.Tensor) else labeled_scores
    labeled_labels_np = labeled_labels.cpu().detach().numpy() if isinstance(labeled_labels, torch.Tensor) else labeled_labels

    # Save ROC and PR curves
    output_file_roc = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_epo{args.num_epochs}_roc_curves.png')
    output_file_pr = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_epo{args.num_epochs}_pr_curves.png')

    plot_roc_curve(labeled_labels_np, labeled_scores_np, output_file_roc)
    plot_pr_curve(labeled_labels_np, labeled_scores_np, output_file_pr)

    print(f"ROC curve saved to {output_file_roc}")
    print(f"PR curve saved to {output_file_pr}")
    
    return correlation

def plot_model_performance_ori(args):
    """
    Generates and saves a scatter plot comparing AUROC and AUPRC values 
    for different models across multiple networks.

    Parameters:
    - models: List of model names.
    - networks: List of network names.
    - auroc: 2D list of AUROC scores (rows: models, cols: networks).
    - auprc: 2D list of AUPRC scores (rows: models, cols: networks).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the plot.
    """


    # Define models and networks
    models = ["ACGNN", "HGDC", "EMOGI", "MTGCN", "GCN", "GAT", "GraphSAGE", "GIN", "Chebnet"]
    networks = ["CPDB", "STRING", "HIPPIE"]

    # AUPRC values for ONGene and OncoKB for each model (rows: models, cols: networks)
    auroc = [
        [0.9652, 0.9578, 0.9297],  # ACGNN ACGNN & 0.9652 & 0.9783 & 0.9578 & 0.9738 & 0.9297 & 0.9597 \\
        [0.6776, 0.7133, 0.6525],  # HGDC
        [0.6735, 0.8184, 0.6672],  # EMOGI
        [0.6862, 0.7130, 0.6762],  # MTGCN
        [0.6915, 0.6688, 0.6708],  # GCN
        [0.6670, 0.8166, 0.6478],  # GAT
        [0.6664, 0.6166, 0.6571],  # GraphSAGE
        [0.5836, 0.5173, 0.5844],  # GIN
        [0.8017, 0.8777, 0.7409]   # Chebnet
    ]

    auprc = [
        [0.9783, 0.9738, 0.9597],  # ACGNN
        [0.7288, 0.7740, 0.7634],  # HGDC
        [0.7230, 0.8737, 0.7960],  # EMOGI
        [0.7712, 0.7878, 0.7785],  # MTGCN
        [0.7730, 0.7681, 0.7675],  # GCN
        [0.7086, 0.8791, 0.7496],  # GAT
        [0.7522, 0.7182, 0.7624],  # GraphSAGE
        [0.6405, 0.5918, 0.6791],  # GIN
        [0.8622, 0.9159, 0.8443]   # Chebnet
    ]

    # Compute averages for each model
    avg_auroc = np.mean(auroc, axis=1)
    avg_auprc = np.mean(auprc, axis=1)

    # Define colors for models and unique shapes for networks
    colors = ['red', 'grey', 'blue', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink']
    network_markers = ['P', '^', 's']  # One shape for each network
    avg_marker = 'o'  # Marker for average points

    # Create the plot
    plt.figure(figsize=(8, 7))

    # Plot individual points for each model and network
    for i, model in enumerate(models):
        for j, network in enumerate(networks):
            plt.scatter(auprc[i][j], auroc[i][j], color=colors[i], 
                        marker=network_markers[j], s=90, alpha=0.6)

    # Add average points for each model
    for i, model in enumerate(models):
        plt.scatter(avg_auprc[i], avg_auroc[i], color=colors[i], marker=avg_marker, 
                    s=240, edgecolor='none', alpha=0.5)

    # Create legends for models (colors) and networks (shapes)
    model_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                            markersize=14, label=models[i], alpha=0.5) for i in range(len(models))]
    network_legend = [Line2D([0], [0], marker=network_markers[i], color='k', linestyle='None', 
                            markersize=8, label=networks[i]) for i in range(len(networks))]

    # Add legends
    network_legend_artist = plt.legend(handles=network_legend, loc='lower right', title="Networks", fontsize=12, title_fontsize=14, frameon=True)
    plt.gca().add_artist(network_legend_artist)
    plt.legend(handles=model_legend, loc='upper left', fontsize=12, frameon=True)

    # Labels and title
    plt.ylabel("AUPRC", fontsize=14)
    plt.xlabel("AUROC", fontsize=14)

    # Save the plot
    comp_output_path = os.path.join('results/gene_prediction/', f'{args.model_type}_{args.net_type}_comp_plot_epo{args.num_epochs}.png')
    plt.savefig(comp_output_path, bbox_inches='tight')
    
    print(f"Comparison plot saved to {comp_output_path}")

    # Show plot
    plt.tight_layout()
    plt.show()

def plot_model_performance(args):
    """
    Generates and saves a scatter plot comparing AUROC and AUPRC values 
    for different models across multiple networks.
    """

    # ----------------------------
    # Models and networks
    # ----------------------------
    models = [
        "ACGNN", "DMGNN", "MOGAT",
        "HGDC", "EMOGI", "MTGCN",
        "GCN", "GAT", "GraphSAGE", "GIN", "Chebnet"
    ]
    networks = ["CPDB", "STRING", "HIPPIE"]

    # ----------------------------
    # AUROC (rows: models, cols: networks)
    # ----------------------------
    auroc = [
        [0.9652, 0.9578, 0.9297],  # ACGNN
        [0.7012, 0.9135, 0.7834],  # DMGNN
        [0.5598, 0.6429, 0.5664],  # MOGAT
        [0.6776, 0.7133, 0.6525],  # HGDC
        [0.6735, 0.8185, 0.6672],  # EMOGI
        [0.6862, 0.7130, 0.6762],  # MTGCN
        [0.6915, 0.6688, 0.6708],  # GCN
        [0.6670, 0.8166, 0.6478],  # GAT
        [0.6664, 0.6166, 0.6571],  # GraphSAGE
        [0.5836, 0.5173, 0.5844],  # GIN
        [0.8017, 0.8777, 0.7409],  # Chebnet
    ]

    # ----------------------------
    # AUPRC
    # ----------------------------
    auprc = [
        [0.9783, 0.9738, 0.9597],  # ACGNN
        [0.9441, 0.9793, 0.8810],  # DMGNN
        [0.8687, 0.7081, 0.6672],  # MOGAT
        [0.7288, 0.7740, 0.7634],  # HGDC
        [0.7230, 0.8737, 0.7960],  # EMOGI
        [0.7712, 0.7878, 0.7785],  # MTGCN
        [0.7730, 0.7681, 0.7675],  # GCN
        [0.7086, 0.8791, 0.7496],  # GAT
        [0.7522, 0.7182, 0.7624],  # GraphSAGE
        [0.6405, 0.5918, 0.6791],  # GIN
        [0.8622, 0.9159, 0.8443],  # Chebnet
    ]

    # ----------------------------
    # Averages
    # ----------------------------
    avg_auroc = np.mean(auroc, axis=1)
    avg_auprc = np.mean(auprc, axis=1)

    # ----------------------------
    # Plot styling
    # ----------------------------
    colors = [
        'red', 'darkorange', 'olive',
        'grey', 'blue', 'green',
        'purple', 'orange', 'cyan', 'brown', 'pink'
    ]
    network_markers = ['P', '^', 's']
    avg_marker = 'o'

    plt.figure(figsize=(8, 7))

    # ----------------------------
    # Individual network points
    # ----------------------------
    for i, model in enumerate(models):
        for j, network in enumerate(networks):
            plt.scatter(
                auprc[i][j], auroc[i][j],
                color=colors[i],
                marker=network_markers[j],
                s=90,
                alpha=0.6
            )

    # ----------------------------
    # Average points
    # ----------------------------
    for i, model in enumerate(models):
        plt.scatter(
            avg_auprc[i], avg_auroc[i],
            color=colors[i],
            marker=avg_marker,
            s=240,
            edgecolor='none',
            alpha=0.5
        )

    # ----------------------------
    # Legends
    # ----------------------------
    model_legend = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=colors[i],
               markersize=14, label=models[i], alpha=0.5)
        for i in range(len(models))
    ]

    network_legend = [
        Line2D([0], [0], marker=network_markers[i],
               color='k', linestyle='None',
               markersize=8, label=networks[i])
        for i in range(len(networks))
    ]

    net_leg = plt.legend(
        handles=network_legend,
        loc='lower right',
        title="Networks",
        fontsize=12,
        title_fontsize=14,
        frameon=True
    )
    plt.gca().add_artist(net_leg)

    plt.legend(
        handles=model_legend,
        loc='upper left',
        fontsize=11,
        frameon=True
    )

    # ----------------------------
    # Axes
    # ----------------------------
    plt.ylabel("AUPRC", fontsize=14)
    plt.xlabel("AUROC", fontsize=14)

    # ----------------------------
    # Save
    # ----------------------------
    comp_output_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_comp_plot_epo{args.num_epochs}.png'
    )
    plt.savefig(comp_output_path, bbox_inches='tight', dpi=300)
    print(f"Comparison plot saved to {comp_output_path}")

    plt.tight_layout()
    plt.show()

def save_model_details(model, args, model_csv_path, in_feats, hidden_feats, out_feats):
    """
    Extracts model details and saves them to a CSV file.

    Parameters:
    - model: The neural network model.
    - args: Arguments containing model configuration.
    - model_csv_path: File path to save the model details.
    - in_feats: Number of input features.
    - hidden_feats: Number of hidden layer features.
    - out_feats: Number of output features.
    """
    # Count layers and parameters
    num_layers = sum(1 for _ in model.children())  # Count layers
    total_params = sum(p.numel() for p in model.parameters())  # Count parameters

    # Detect attention layers
    attention_layer_nodes = None
    for layer in model.children():
        if hasattr(layer, 'heads'):  # Assuming attention layers have 'heads' attribute
            attention_layer_nodes = layer.heads

    # Detect residual connections
    has_residual = any(isinstance(layer, nn.Identity) for layer in model.modules())

    # Prepare data for CSV
    model_data = {
        "Method": [args.model_type],
        "Number of Layers": [num_layers],
        "Input Layer Nodes": [in_feats],
        "Hidden Layer Nodes": [hidden_feats],
        "Attention Layer Nodes": [attention_layer_nodes if attention_layer_nodes else "N/A"],
        "Output Layer Nodes": [out_feats],
        "Total Parameters": [total_params],
        "Residual Connection": ["Yes" if has_residual else "No"]
    }

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(model_data)
    df.to_csv(model_csv_path, index=False)
    print(f"Model architecture saved to {model_csv_path}")

def save_predicted_scores_ori(scores, labels, nodes, args, save_path):
    """
    Saves predicted scores and labels to a CSV file.

    Parameters:
    - scores: List of predicted scores.
    - labels: List of ground-truth labels.
    - nodes: Dictionary of node names.
    - args: Arguments containing model configuration.
    """
    # Initialize variables to calculate average scores and standard deviations
    label_scores = {0: [], 1: [], 2: [], 3: []}  # Groups for each label

    # # Define CSV file path
    # csv_file_path = os.path.join(
    #     'results/gene_prediction/',
    #     f'{args.model_type}_{args.net_type}_predicted_scores_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    # )

    # # Ensure directory exists
    # os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Save results to CSV
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node Name', 'Score', 'Label'])  # Header

        for i, score in enumerate(scores):
            label = int(labels[i].item())  # Ensure label is an integer
            
            if label in [1, 0]:  # Ground-truth labels
                writer.writerow([list(nodes.keys())[i], score, label])
                label_scores[label].append(score)
            elif label == -1 and score >= args.score_threshold:  # Predicted driver genes
                writer.writerow([list(nodes.keys())[i], score, 2])
                label_scores[2].append(score)
            else:  # Non-labeled nodes or other
                writer.writerow([list(nodes.keys())[i], score, 3])
                label_scores[3].append(score)

    print(f"Predicted scores and labels saved to {save_path}")

    return label_scores  # Returning for further analysis if needed

import csv
import numpy as np
import os

def save_predicted_scores(scores, labels, nodes, args, save_path):
    """
    Saves predicted scores and labels to a CSV file.

    Parameters:
    - scores: array-like of predicted scores
    - labels: array-like of ground-truth labels
    - nodes: dict {name: id} OR list/np.ndarray of node names
    - args: arguments containing model configuration
    """

    # -------------------------
    # Normalize node names
    # -------------------------
    if isinstance(nodes, dict):
        node_names = list(nodes.keys())
    elif isinstance(nodes, (list, np.ndarray)):
        node_names = list(nodes)
    else:
        raise TypeError(f"Unsupported type for nodes: {type(nodes)}")

    assert len(node_names) == len(scores), \
        "Mismatch between number of nodes and scores"

    # -------------------------
    # Containers for stats
    # -------------------------
    label_scores = {0: [], 1: [], 2: [], 3: []}

    # -------------------------
    # Save CSV
    # -------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node Name', 'Score', 'Label'])

        for i, score in enumerate(scores):
            label = int(labels[i].item())

            node_name = node_names[i]

            if label in [0, 1]:  # Ground-truth
                writer.writerow([node_name, score, label])
                label_scores[label].append(score)

            elif label == -1 and score >= args.score_threshold:  # Predicted driver
                writer.writerow([node_name, score, 2])
                label_scores[2].append(score)

            else:  # Non-driver / other
                writer.writerow([node_name, score, 3])
                label_scores[3].append(score)

    print(f"Predicted scores and labels saved to {save_path}")
    return label_scores

def save_average_scores(label_scores, args):
    """
    Calculates and saves the average score, standard deviation, and number of nodes per label.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    # Define CSV file path
    average_scores_file = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_by_label_threshold{args.score_threshold}_epo{args.num_epochs}.csv'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(average_scores_file), exist_ok=True)

    # Save average scores to CSV
    with open(average_scores_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Label', 'Average Score', 'Standard Deviation', 'Number of Nodes'])  # Header

        for label, scores_list in label_scores.items():
            if scores_list:  # Check if the list is not empty
                avg_score = np.mean(scores_list)
                std_dev = np.std(scores_list)
                num_nodes = len(scores_list)
            else:
                avg_score = 0.0  # Default if no nodes in the label group
                std_dev = 0.0
                num_nodes = 0

            writer.writerow([label, avg_score, std_dev, num_nodes])

    print(f"Average scores by label saved to {average_scores_file}")

def plot_average_scores(label_scores, args):
    """
    Plots average scores with error bars and saves the figure.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    labels_list = []
    avg_scores = []
    std_devs = []

    for label, scores_list in label_scores.items():
        if scores_list:
            labels_list.append(label)
            avg_scores.append(np.mean(scores_list))
            std_devs.append(np.std(scores_list))

    if not labels_list:
        print("No valid scores to plot.")
        return

    # Define plot save path
    plot_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_average_scores_with_error_bars_threshold{args.score_threshold}_epo{args.num_epochs}.png'
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    # Create plot
    plt.figure(figsize=(8, 6))
    plt.bar(labels_list, avg_scores, yerr=std_devs, capsize=5, color='skyblue', alpha=0.7)
    plt.xlabel('Label')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Label with Error Bars')
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Save and close plot
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Error bar plot saved to {plot_path}")

def plot_score_distributions(label_scores, args):
    """
    Plots score distributions for each label and saves the figures.

    Parameters:
    - label_scores: Dictionary with labels as keys and lists of scores as values.
    - args: Arguments containing model configuration.
    """
    for label, scores_list in label_scores.items():
        if scores_list:
            plt.figure(figsize=(8, 6))
            plt.hist(scores_list, bins=20, alpha=0.7, color='#98f5e1', edgecolor='black')

            # Set labels and tick sizes
            plt.xlabel('Score', fontsize=16)
            plt.ylabel('Frequency', fontsize=16)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)

            # Customize grid and tick appearance
            plt.tick_params(axis='both', which='major', length=6, width=2, direction='inout', grid_color='gray', grid_alpha=0.5)
            plt.grid(axis='y', linestyle='--', alpha=0.6)

            # Define plot save path
            plot_path = os.path.join(
                'results/gene_prediction/',
                f'{args.model_type}_{args.net_type}_score_distribution_label{label}_threshold{args.score_threshold}_epo{args.num_epochs}.png'
            )

            # Ensure directory exists
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)

            # Save and close plot
            plt.savefig(plot_path)
            plt.close()

            print(f"Score distribution for label {label} saved to {plot_path}")

def save_performance_metrics_ori(epoch_times, cpu_usages, gpu_usages, args):
    """
    Saves performance metrics per epoch, including time per epoch, CPU, and GPU usage.

    Parameters:
    - epoch_times: List of epoch durations (in seconds).
    - cpu_usages: List of CPU memory usage per epoch (in MB).
    - gpu_usages: List of GPU memory usage per epoch (in MB).
    - args: Arguments containing model and training configuration.
    - output_dir: Directory to save the metrics CSV file.
    """

    # Compute total and average performance metrics
    total_time = sum(epoch_times)
    avg_time_per_epoch = total_time / args.num_epochs
    avg_cpu_usage = sum(cpu_usages) / args.num_epochs
    avg_gpu_usage = sum(gpu_usages) / args.num_epochs

    # Create DataFrame with per-epoch metrics
    df_metrics = pd.DataFrame({
        "Epoch": range(1, args.num_epochs + 1),
        "Time per Epoch (s)": epoch_times,
        "CPU Usage (MB)": cpu_usages,
        "GPU Usage (MB)": gpu_usages
    })

    # Define CSV path
    metrics_csv_path = os.path.join(
        'results/gene_prediction/',
        f'{args.model_type}_{args.net_type}_performance_metrics_epo{args.num_epochs}.csv'
    )

    # Save to CSV
    df_metrics.to_csv(metrics_csv_path, index=False)

    print(f"Epoch performance metrics saved to {metrics_csv_path}")

    # Print summary statistics
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f} MB")
    print(f"Average GPU Usage: {avg_gpu_usage:.2f} MB")



def save_performance_metrics(epoch_times, cpu_usages, gpu_usages, args):
    """
    Saves per-epoch performance metrics (time, CPU, GPU).
    """

    # -------------------------
    # Align lengths safely
    # -------------------------
    n_epochs = min(len(epoch_times), len(cpu_usages), len(gpu_usages))

    if n_epochs == 0:
        print("Warning: No performance metrics to save.")
        return

    epoch_times = epoch_times[:n_epochs]
    cpu_usages = cpu_usages[:n_epochs]
    gpu_usages = gpu_usages[:n_epochs]

    # -------------------------
    # Compute summary stats
    # -------------------------
    total_time = sum(epoch_times)
    avg_time_per_epoch = total_time / n_epochs
    avg_cpu_usage = sum(cpu_usages) / n_epochs
    avg_gpu_usage = sum(gpu_usages) / n_epochs

    # -------------------------
    # Create DataFrame
    # -------------------------
    df_metrics = pd.DataFrame({
        "Epoch": range(1, n_epochs + 1),
        "Time per Epoch (s)": epoch_times,
        "CPU Usage (MB)": cpu_usages,
        "GPU Usage (MB)": gpu_usages,
    })

    # -------------------------
    # Save CSV
    # -------------------------
    os.makedirs("results/gene_prediction", exist_ok=True)

    metrics_csv_path = os.path.join(
        "results/gene_prediction",
        f"{args.model_type}_{args.net_type}_performance_metrics_epo{n_epochs}.csv",
    )

    df_metrics.to_csv(metrics_csv_path, index=False)

    print(f"Epoch performance metrics saved to {metrics_csv_path}")

    # -------------------------
    # Print summary
    # -------------------------
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f} MB")
    print(f"Average GPU Usage: {avg_gpu_usage:.2f} MB")
