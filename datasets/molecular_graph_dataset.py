import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import periodictable
from data_massage.mendeleev_table.periods import periods

class MolecularGraphDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = self.build_graph
        self.file_path = \
            '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/'
        self.excel_file_path = self.file_path + "K_I_C_B_prop_ALL.xlsx"
        self.data_excel = pd.read_excel(self.excel_file_path)
        self.data = self.data_excel.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph, seebeck = self.transform(self.data[idx])
        return graph, seebeck

    def atom_to_ordenal(self, atom):
        """
        Return ordinal number for specific atom.
        """
        element = periodictable.elements.symbol(atom)
        atomic_number = element.number
        return atomic_number

    def get_atoms_period_number(self, atom):
        for idx, period in enumerate(periods):
            if atom in period:
                return idx+1
        print(f'INCORRECT atoms name: {atom}')

    def build_graph(self, mol_data):
        """
        Makes graph.
        Saves atom coordinates as node attributes, calculates the length of edges (distance between atoms).
        Graph is fully connected - all atoms are connected by edges.
        """
        # k, i, c, b - see keys in 'props.json'
        _, formula, seebeck, phase_id, cell_abc_str, _, basis_noneq, els_noneq, k, i, c, b = mol_data
        els_noneq = eval(els_noneq)
        basis_noneq = eval(basis_noneq)

        # create list with features for every node
        x_vector = [[self.get_atoms_period_number(atom)] for atom in els_noneq]

        # add coordinates to every node
        for i, atom in enumerate(els_noneq):
            x_vector[i].append(basis_noneq[i][0])
            x_vector[i].append(basis_noneq[i][1])
            x_vector[i].append(basis_noneq[i][2])
            x_vector[i].append(k)
            x_vector[i].append(i)
            x_vector[i].append(c)
            x_vector[i].append(b)

        node_features = torch.tensor(x_vector)

        edge_index = []
        edge_attr = []

        # to calculate distance between all atoms
        for i in range(len(basis_noneq)):
            for j in range(i + 1, len(basis_noneq)):
                distance = torch.norm(torch.tensor(basis_noneq[i]) - torch.tensor(basis_noneq))

                # graph is undirected, so we duplicate edge
                edge_index.append([i, j])
                edge_index.append([j, i])

                edge_attr.append(distance)
                edge_attr.append(distance)

        graph_data = Data(x=node_features, edge_index=torch.tensor(edge_index).t().contiguous(),
                          edge_attr=torch.tensor(edge_attr))
        return graph_data, seebeck
