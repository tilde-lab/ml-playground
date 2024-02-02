import numpy as np
from ase.data import chemical_symbols, covalent_radii
from ase import neighborlist
from scipy import sparse
from scipy.sparse import csr_matrix
import math

# Defining descriptors


def get_APF(ase_obj):
    volume = 0.0
    for atom in ase_obj:
        volume += (
            4 / 3 * np.pi * covalent_radii[chemical_symbols.index(atom.symbol)] ** 3
        )
    return volume / abs(np.linalg.det(ase_obj.cell))


def get_Wiener(ase_obj):
    return np.sum(ase_obj.get_all_distances()) * 0.5


def get_Randic(ase_obj):
    cutoff = neighborlist.natural_cutoffs(ase_obj)
    neighborList = neighborlist.NeighborList(
        cutoff, self_interaction=False, bothways=True
    )
    neighborList.update(ase_obj)
    matrix = neighborList.get_connectivity_matrix()  # connectivity matrix
    matrix1 = csr_matrix(matrix)  # converting sparse matrix to csr matrix
    (
        A,
        B,
    ) = matrix1.nonzero()  # adjacent vertices at corresponding positions of both arrays
    Sum = matrix1.sum(axis=0)
    sum1 = Sum.tolist()  # degree of each vertex
    sum1 = sum1[0]
    natoms = len(sum1)  # number of atoms in unit cell
    Final = []
    for a in range(natoms):
        k = 1 / math.sqrt(sum1[A[a]] * sum1[B[a]])
        Final.append(k)
    Randic_no = np.sum(Final)  # connectivity index
    return Randic_no


# Evaluation metrics: R2 coefficient


def get_R2score(prediction, total):
    return sum((prediction - total["Value"].mean()) ** 2) / sum(
        (total["Value"] - total["Value"].mean()) ** 2
    )
