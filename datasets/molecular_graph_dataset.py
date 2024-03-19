import pandas as pd
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
import periodictable
from data_massage.mendeleev_table import get_periodic_number
from descriptors.utils import get_APF, get_Wiener
from mpds_client import MPDSDataRetrieval
from ase import Atoms


class MolecularGraphDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = self.build_graph
        self.file_path = \
            '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/'
        self.excel_file_path = self.file_path + "I_C_PEER_INITIO.xlsx"
        self.data_excel = pd.read_excel(self.excel_file_path)
        self.data = self.data_excel.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        graph, seebeck, p, c, apf, wiener  = self.transform(self.data[idx])
        return graph, seebeck, p, c, apf, wiener

    def atom_to_ordinal(self, atom):
        """
        Returns ordinal number for specific atom.
        """
        element = periodictable.elements.symbol(atom)
        atomic_number = element.number
        return atomic_number

    def calculate_apf_and_wiener(self, items):
        """
        Calculates APF and Wiener for each graph.
        """
        if items[4] != 1:
            crystal = MPDSDataRetrieval.compile_crystal(items, "ase")
            if not crystal:
                return None, None
        elif items[4] == 1:
            crystal = Atoms(symbols=items[6], positions=items[5], cell=items[3])
            if not crystal:
                return None, None

        return get_APF(crystal), get_Wiener(crystal)

    def build_graph(self, mol_data):
        """
        Makes graph.
        Saves atom coordinates as node attributes, calculates the length of edges (distance between atoms).
        Graph is fully connected - all atoms are connected by edges.
        """
        # k, i(p), c, b - see keys in 'props.json'
        ph, formula, seebeck, entry, cell_abc_str, sg_n, basis_noneq, els_noneq, p, c = mol_data
        els_noneq = eval(els_noneq)
        basis_noneq = eval(basis_noneq)

        # create list with features for every node
        x_vector = [[get_periodic_number(atom)] for atom in els_noneq]

        # add coordinates to every node
        for i, atom in enumerate(els_noneq):
            x_vector[i].append(basis_noneq[i][0])
            x_vector[i].append(basis_noneq[i][1])
            x_vector[i].append(basis_noneq[i][2])
            x_vector[i].append(p)
            x_vector[i].append(c)

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

        apf, wiener = self.calculate_apf_and_wiener(
            [ph, entry, formula, eval(cell_abc_str), sg_n, basis_noneq, els_noneq]
        )

        graph_data = Data(x=node_features, edge_index=torch.tensor(edge_index).t().contiguous(),
                          edge_attr=torch.tensor(edge_attr))
        return graph_data, seebeck, p, c, apf, wiener
