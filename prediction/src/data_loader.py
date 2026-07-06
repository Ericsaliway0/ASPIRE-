import json
import networkx as nx
import dgl
import torch


def load_graph_data(file_path):

    # -----------------------------
    # LOAD JSON
    # -----------------------------
    with open(file_path, 'r') as file:
        data = json.load(file)

    G_nx = nx.DiGraph()
    edge_type_mapping = {}

    # -----------------------------
    # LOAD NODES
    # -----------------------------
    id_to_name = {}
    node_features_dict = {}
    node_labels_dict = {}

    for node in data["nodes"]:

        nid = node["id"]
        name = str(node["name"])

        emb = node.get("embedding", [0.0] * 128)

        label = node.get("label", 0)
        label = 0 if label is None else label

        id_to_name[nid] = name

        G_nx.add_node(name)

        node_features_dict[name] = emb
        node_labels_dict[name] = label

    # -----------------------------
    # LOAD EDGES
    # -----------------------------
    for edge in data["edges"]:

        s_id = edge["source"]
        t_id = edge["target"]
        rel_type = str(edge["type"])

        s_name = id_to_name[s_id]
        t_name = id_to_name[t_id]

        if rel_type not in edge_type_mapping:
            edge_type_mapping[rel_type] = len(edge_type_mapping)

        G_nx.add_edge(
            s_name,
            t_name,
            type=edge_type_mapping[rel_type]
        )

    # -----------------------------
    # CONSISTENT NODE ORDER
    # -----------------------------
    node_names = list(G_nx.nodes())

    # -----------------------------
    # FEATURES (ALIGNED)
    # -----------------------------
    node_features = torch.tensor(
        [node_features_dict[n] for n in node_names],
        dtype=torch.float32
    )

    node_labels = torch.tensor(
        [node_labels_dict[n] for n in node_names],
        dtype=torch.float32
    )

    # -----------------------------
    # CONVERT TO DGL
    # -----------------------------
    G_dgl = dgl.from_networkx(
        G_nx,
        edge_attrs=['type']
    )

    G_dgl.ndata['feat'] = node_features
    G_dgl.ndata['label'] = node_labels

    # 🔥 preserve real gene names
    G_dgl.gene_names = node_names

    # -----------------------------
    # RETURN 4 OBJECTS
    # -----------------------------
    return G_dgl, node_features, node_labels, node_names