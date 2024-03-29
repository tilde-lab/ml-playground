import pandas as pd
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data

class CrystalGraphVectorsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.total = pd.read_csv(
            '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/rep_vectors_str_200.csv'
        )
        self.seebeck = pd.read_csv(
            '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/seebeck_200.csv'
        )
        self.data = pd.concat([self.seebeck["Seebeck coefficient"], self.total], axis=1).values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        atoms = self.data[idx][1]
        distance = self.data[idx][2]
        seebeck = self.data[idx][0]

        graph = self.build_graph([atoms, distance])
        return graph, seebeck

    def build_graph(self, crystal_data):
        """
        Makes graph.
        """
        atoms, distance = crystal_data

        # create list with features for every node
        x_vector = []

        # add coordinates to every node
        for i, d in enumerate(eval(distance)):
            x_vector.append([])
            x_vector[i].append(eval(atoms)[i])
            x_vector[i].append(d)

        node_features = torch.tensor(x_vector)

        edge_index = []

        for i in range(len(eval(atoms))):
            for j in range(i + 1, len(eval(atoms))):
                # graph is undirected, so we duplicate edge
                if i != j:
                    edge_index.append([i, j])
                    edge_index.append([j, i])

        graph_data = Data(x=node_features, edge_index=torch.tensor(edge_index).t().contiguous())
        return graph_data



