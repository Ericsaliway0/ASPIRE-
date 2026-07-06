import json
import networkx as nx

class Network:
    def __init__(self, json_path):
        # Initialize directed graph
        self.graph_nx = nx.DiGraph()

        # Load JSON data
        with open(json_path, "r") as f:
            self.pathways_json = json.load(f)

        # Build the graph
        self.build_graph_from_json()

    def build_graph_from_json(self):
        def traverse(node, parent_gene=None):
            current_gene = node.get("stId")  # Already gene name or ID
            gene_name = node.get("name", current_gene)
            gene_type = node.get("type", "Gene")

            # Add node with attributes
            self.graph_nx.add_node(current_gene, stId=current_gene, name=gene_name, type=gene_type)

            # Add edge from parent to current node
            if parent_gene:
                self.graph_nx.add_edge(parent_gene, current_gene)

            # Recursively traverse children
            for child in node.get("children", []):
                traverse(child, current_gene)

        for top_node in self.pathways_json:
            traverse(top_node)

    def get_gene_info(self, gene):
        if gene in self.graph_nx:
            return dict(self.graph_nx.nodes[gene])
        return None

    def display_graph(self):
        for node in self.graph_nx.nodes:
            info = self.get_gene_info(node)
            print(f"{node}: {info}")

    def save_name_to_id(self, file_path="name_to_id.txt"):
        name_to_id = {node: self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes}
        with open(file_path, 'w') as f:
            for name, stid in name_to_id.items():
                f.write(f"{name}: {stid}\n")

    def save_sorted_stids(self, file_path="sorted_stids.txt"):
        stids = sorted([self.graph_nx.nodes[node]['stId'] for node in self.graph_nx.nodes])
        with open(file_path, 'w') as f:
            for stid in stids:
                f.write(f"{stid}\n")
