import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import pickle


file_path = '/root/projects/ml-playground/descriptors/seebeck_coefficient/data/not_repetitive_phase_id/'

excel_file_path = file_path + 'structure_and_seebeck_uniq' + ".xlsx"
data = pd.read_excel(excel_file_path)

structure_list = data.values.tolist()


class MolecularGraphDataset(Dataset):
    def __init__(self, structure_list):
        super().__init__()
        self.transform = self.build_graph
        self.data = structure_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph = self.transform(self.data[idx])
        return graph

    def atom_to_one_hot(self, atom):
        '''
        Makes one-hot vector for specific atom.
        '''
        atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                 'Kr',
                 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                 'Xe',
                 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                 'Hf',
                 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac',
                 'Th',
                 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh',
                 'Hs',
                 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

        one_hot = [1 if atom == a else 0 for a in atoms]
        return one_hot

    def build_graph(self, mol_data):
        '''
        Makes graph.
        Saves atom coordinates as node attributes, calculates the length of edges (distance between atoms).
        Graph is fully connected - all atoms are connected by edges.
        '''
        _, formula, _, phase_id, cell_abc_str, _, basis_noneq, els_noneq = mol_data
        els_noneq = eval(els_noneq)

        # makes one-hot tensor
        one_hot = [self.atom_to_one_hot(atom) for atom in els_noneq]

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
        return graph_data


mol_dataset = MolecularGraphDataset(structure_list)
