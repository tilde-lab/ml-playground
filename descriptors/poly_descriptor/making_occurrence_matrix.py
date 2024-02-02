import numpy as np
import pandas as pd
from scipy import linalg
import pickle

np.set_printoptions(threshold=np.inf)


# To get number of compounds with point symbols in the ToposPro file
def get_no_of_compounds(fp):
    cc = 0
    temp = [" "]
    token = ["###########"]
    for ii in fp:
        temp.append(ii.rstrip("\n"))
        if token[0] in ii:
            cc += 1
    return int(cc / 2)


# Driver code
fp = open("point_symbol_info")
k = get_no_of_compounds(fp)
print(k)
fp.close()

# Used bash script to get specific lines with elements and point symbols from saved text file of ToposPro


# Refining file to filter out irrelevant data
def refine_file(fp):
    fp1 = open("final_file.txt", "w")
    new_lines = [" "]
    for ii in fp:
        line = ii.rstrip("\n")
        line = line.split(" ")
        line[0] = "".join(c for c in line[0] if c.isalpha())
        fp1.write(line[0] + " " + line[1] + "\n")
    fp1.close()


# Used bash scripts to sort and get unique values of elements (rows) and pont_symbols (columns) taken from final_file.txt file

# Getting rows and columns for the 2D array
row_file = open("rows.txt", "r")
column_file = open("columns.txt", "r")
row = row_file.read().rsplit("\n")
row = row[:-1]  # to remove empty item in the end of the list
row = np.array(row)
column = column_file.read().rsplit("\n")
column = column[:-1]  # to remove empty item in the end of the list
column = np.array(column)
row_file.close()
column_file.close()

# Creating the occurrence matrix
file = open("count.txt")
matrix = np.empty([len(row), len(column)])
for index, ii in enumerate(file):
    line = ii.rstrip("\n")
    line = line.split(" ")
    better_line = [x for x in line if x != ""]
    for i in range(len(row)):
        for j in range(len(column)):
            if row[i] == better_line[1] and column[j] == better_line[2]:
                matrix[i][j] = better_line[0]
file.close()


# Normalization
factor = np.sqrt(np.sum(matrix, axis=1))
norm_matrix = matrix / factor[:, None]

# Single Value Decomposition
U, D, Vt = linalg.svd(norm_matrix)

# Descriptor formation
sigma = np.zeros(
    (min(len(row), len(column)), min(len(row), len(column))), dtype=float, order="C"
)
for i in range(min(len(row), len(column))):
    sigma[i, i] = D[i]
descriptor = np.dot(U, sigma)
print(descriptor)

pickle.dump(descriptor, open("descriptor.p", "wb"))
