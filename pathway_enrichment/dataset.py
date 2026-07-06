import os
import pickle
import subprocess

import dgl
from dgl.data import DGLDataset


class PathwayDataset(DGLDataset):
    def __init__(self, root='reactome_embedding/data/emb'):

    # def __init__(self, root='data'):
        raw_dir = os.path.join(root, 'raw')
        save_dir = os.path.join(root, 'processed')
        model_dir = os.path.join(root, 'models')

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # store them separately if you need to reuse later
        self._raw_dir = raw_dir
        self._save_dir = save_dir
        super().__init__(name='pathway_graph', raw_dir=raw_dir, save_dir=save_dir)
        

    def has_cache(self):
        return len(os.listdir(self.save_dir)) == len(os.listdir(self.raw_dir))

    def __len__(self):
        return len(os.listdir(self.save_dir))

    def __getitem__(self, idx):
        names = sorted(os.listdir(self.save_dir))
        name = names[idx]
        (graph,), _ = dgl.load_graphs(os.path.join(self.save_dir, name))
        return graph, name

    def process(self):
        for cnt, graph_file in enumerate(os.listdir(self.raw_dir)):
            graph_path = os.path.join(self.raw_dir, graph_file)
            nx_graph = pickle.load(open(graph_path, 'rb'))
            for node in nx_graph.nodes:
                if nx_graph.nodes[node]['significance'] == 'significant':
                    nx_graph.nodes[node]['significance'] = 1.0
                else:
                    nx_graph.nodes[node]['significance'] = 0.0
            dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['weight', 'significance'])
            save_path = os.path.join(self.save_dir, f'{graph_file[:-4]}.dgl')
            dgl.save_graphs(save_path, dgl_graph)
