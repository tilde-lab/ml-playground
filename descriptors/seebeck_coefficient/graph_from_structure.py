import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import json
import periodictable

with open('/Users/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/one_hot.json', "r") as f:
    dict_one_hot = json.load(f)

class MolecularGraphDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = self.build_graph
        self.file_path = '/Users/alina/PycharmProjects/ml-playground/descriptors/seebeck_coefficient/data/'
        self.excel_file_path = self.file_path + 'normalized_median_structure_and_seebeck' + ".xlsx"
        self.data_excel = pd.read_excel(self.excel_file_path)
        self.data = self.data_excel.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph, seebeck = self.transform(self.data[idx])
        return graph, seebeck

    def atom_to_one_hot(self, atom):
        '''
        Return one-hot vector for specific atom.
        '''
        element = periodictable.elements.symbol(atom)
        atomic_number = element.number
        return atomic_number

    def build_graph(self, mol_data):
        '''
        Makes graph.
        Saves atom coordinates as node attributes, calculates the length of edges (distance between atoms).
        Graph is fully connected - all atoms are connected by edges.
        '''
        _, formula, seebeck, phase_id, cell_abc_str, _, basis_noneq, els_noneq = mol_data
        els_noneq = eval(els_noneq)

        # makes one-hot tensor
        one_hot = [[self.atom_to_one_hot(atom)] for atom in els_noneq]

        # add coordinates to every node
        for i, atom in enumerate(els_noneq):
            one_hot[i].append(eval(basis_noneq)[i][0])
            one_hot[i].append(eval(basis_noneq)[i][1])
            one_hot[i].append(eval(basis_noneq)[i][2])

        node_features = torch.tensor(one_hot)

        edge_index = []
        edge_attr = []

        # to calculate distance between all atoms
        for i in range(len(eval(basis_noneq))):
            for j in range(i + 1, len(eval(basis_noneq))):
                distance = torch.norm(torch.tensor(eval(basis_noneq)[i]) - torch.tensor(eval(basis_noneq)))

                # graph is undirected, so we duplicate edge
                edge_index.append([i, j])
                edge_index.append([j, i])

                edge_attr.append(distance)
                edge_attr.append(distance)

        graph_data = Data(x=node_features, edge_index=torch.tensor(edge_index).t().contiguous(),
                          edge_attr=torch.tensor(edge_attr))
        return graph_data, seebeck
