import numpy as np
from ase.data import chemical_symbols
from ase import Atoms
import pandas as pd
from data_massage.mendeleev_table import periodic_numbers
from torch.utils.data import Dataset
import torch

class CrystalVectorsDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = self.build
        self.file_path = \
            '/root/projects/ml-playground/data_massage/seebeck_coefficient_and_structure/data/26_3/'
        self.excel_file_path = self.file_path + "ordered_str_200.csv"
        self.data_csv = pd.read_csv(self.excel_file_path)
        self.data = self.data_csv.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vectors_str, seebeck = self.transform(self.data[idx])
        return vectors_str, seebeck

    def get_bv_descriptor(self, ase_obj, kappa=None, overreach=False):
        """
        From ASE object obtain a vectorized atomic structure
        populated to a certain fixed relatively big volume
        defined by kappa
        """
        if not kappa: kappa = 18
        if overreach: kappa *= 2

        norms = np.array([np.linalg.norm(vec) for vec in ase_obj.get_cell()])
        multiple = np.ceil(kappa / norms).astype(int)
        ase_obj = ase_obj.repeat(multiple)
        com = ase_obj.get_center_of_mass()
        ase_obj.translate(-com)
        del ase_obj[
            [atom.index for atom in ase_obj if np.sqrt(np.dot(atom.position, atom.position)) > kappa]
        ]

        ase_obj.center()
        ase_obj.set_pbc((False, False, False))
        sorted_seq = np.argsort(np.fromiter((np.sqrt(np.dot(x, x)) for x in ase_obj.positions), np.float))
        ase_obj = ase_obj[sorted_seq]

        elements, positions = [], []
        for atom in ase_obj:
            elements.append(periodic_numbers[chemical_symbols.index(atom.symbol)] - 1)
            positions.append(
                int(round(np.sqrt(atom.position[0] ** 2 + atom.position[1] ** 2 + atom.position[2] ** 2) * 10)))

        return np.array([elements, positions])

    def build(self, struct):
        crystal = Atoms(symbols=eval(struct[6]), positions=eval(struct[5]), cell=eval(struct[3]))
        vectors_str = self.get_bv_descriptor(crystal)
        seebeck = struct[2]

        return vectors_str, seebeck


