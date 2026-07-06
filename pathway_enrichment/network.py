import urllib.request
import json
from collections import defaultdict, namedtuple
from datetime import datetime
import networkx as nx


class Network:
    
    Info = namedtuple('Info', ['name', 'species', 'type', 'diagram'])

    def __init__(self, ea_result=None, kge=None):
        self.txt_url = 'https://reactome.org/download/current/ReactomePathwaysRelation.txt'
        self.json_url = 'https://reactome.org/ContentService/data/eventsHierarchy/9606'

        if kge is not None:
            self.kge = kge
        else:
            self.kge = datetime.now().strftime('%Y-%b-%d-%H-%M')

        self.txt_adjacency = self.parse_txt()
        self.json_adjacency, self.pathway_info = self.parse_json()

        if ea_result is not None:
            self.weights = self.set_weights(ea_result)
        else:
            self.weights = None

        self.name_to_id = self.set_name_to_id()
        self.graph_nx = self.to_networkx()

        self.save_name_to_id()
        self.save_sorted_stids()

    def parse_txt(self):
        txt_adjacency = defaultdict(list)
        found = False

        with urllib.request.urlopen(self.txt_url) as f:
            lines = f.readlines()

            for line in lines:
                line = line.decode('utf-8')
                stid1, stid2 = line.strip().split()

                if 'R-HSA' not in stid1:
                    if found:
                        break
                    else:
                        continue

                found = True
                txt_adjacency[stid1].append(stid2)

        return dict(txt_adjacency)

    def parse_json(self):
        with urllib.request.urlopen(self.json_url) as f:
            tree_list = json.load(f)

        json_adjacency = defaultdict(list)
        pathway_info = {}

        for tree in tree_list:
            self.recursive(tree, json_adjacency, pathway_info)

        return dict(json_adjacency), pathway_info

    def recursive(self, tree, json_adjacency, pathway_info):
        id = tree['stId']

        try:
            pathway_info[id] = Network.Info(
                tree['name'],
                tree['species'],
                tree['type'],
                tree.get('diagram', None)
            )
        except KeyError:
            pathway_info[id] = Network.Info(
                tree.get('name', 'NA'),
                tree.get('species', 'NA'),
                tree.get('type', 'NA'),
                None
            )

        children = tree.get('children', [])
        for child in children:
            json_adjacency[id].append(child['stId'])
            self.recursive(child, json_adjacency, pathway_info)

    def set_weights(self, ea_result):
        weights = {}

        for stid in self.pathway_info.keys():
            if stid in ea_result:
                entry = ea_result[stid]
                weights[stid] = {
                    'p_value': entry.get('p_value', 1.0),
                    'significance': entry.get('significance', 'not-found')
                }
            else:
                weights[stid] = {
                    'p_value': 1.0,
                    'significance': 'not-found'
                }

        return weights

    def set_node_attributes(self):
        stids = {}
        names = {}
        weights = {}
        significances = {}

        for stid in self.pathway_info.keys():
            stids[stid] = stid
            names[stid] = self.pathway_info[stid].name

            if self.weights is None:
                weights[stid] = 1.0
                significances[stid] = 'not-found'
            else:
                weights[stid] = self.weights[stid]['p_value']
                significances[stid] = self.weights[stid]['significance']

        return stids, names, weights, significances

    def to_networkx(self, type='json'):
        graph_nx = nx.DiGraph()

        graph = self.json_adjacency if type == 'json' else self.txt_adjacency

        for key, values in graph.items():
            for value in values:
                graph_nx.add_edge(key, value)

        stids, names, weights, significances = self.set_node_attributes()

        nx.set_node_attributes(graph_nx, stids, 'stId')
        nx.set_node_attributes(graph_nx, names, 'name')
        nx.set_node_attributes(graph_nx, weights, 'weight')
        nx.set_node_attributes(graph_nx, significances, 'significance')

        return graph_nx

    def add_significance_by_stid(self, stid_list):
        for stid in stid_list:
            if stid in self.graph_nx.nodes:
                self.graph_nx.nodes[stid]['significance'] = 'significant'
                self.graph_nx.nodes[stid]['weight'] = 0.0

    def save_name_to_id(self):
        file_path = 'reactome_embedding/data/emb/info/name_to_id.txt'

        with open(file_path, 'w') as f:
            for name, id in self.name_to_id.items():
                f.write(f"{name}: {id}\n")

    def save_sorted_stids(self):
        file_path = 'reactome_embedding/data/emb/info/sorted_stids.txt'

        stids = sorted(self.pathway_info.keys())

        with open(file_path, 'w') as f:
            for stid in stids:
                f.write(f"{stid}\n")

    def set_name_to_id(self):
        name_to_id = {}

        for id, info in self.pathway_info.items():
            name_to_id[info.name] = id

        return name_to_id


# ============================
# 🔥 MAIN USAGE
# ============================

def extract_stId_name_pairs(graph):
    stId_name_pairs = []

    for n in graph.graph_nx.nodes():
        node = graph.graph_nx.nodes[n]

        stId_name_pairs.append((
            node.get('stId', n),
            node.get('name', 'NA'),
            node.get('weight', 1.0),
            node.get('significance', 'not-found')
        ))

    return stId_name_pairs


# ============================
# Example pipeline
# ============================

if __name__ == "__main__":
    
    # Example: fake enrichment result (ea_result)
    ea_result = {
        "R-HSA-12345": {"p_value": 0.001, "significance": "significant"},
        "R-HSA-67890": {"p_value": 0.05, "significance": "marginal"}
    }

    # Build network
    graph = Network(ea_result=ea_result)

    # Extract pairs
    stId_name_pairs = extract_stId_name_pairs(graph)

    # Print a few
    for item in stId_name_pairs[:10]:
        print(item)