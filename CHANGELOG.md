# Changelog

All notable changes to this project will be documented in this file.
## 2023.10.18 - 2023.11.18
- Added [crystal visualization](https://github.com/tilde-lab/ml-playground/commit/3943392f4b25eea5f66c205517fdeb34e16b17a5) by ASE.
- Added [implementation Atomization energy and Equation of state](https://github.com/tilde-lab/ml-playground/commit/0642958e1701c9814e3f48d1601c3a65e292c1b7).
- Added [functions for work with ASE-database](https://github.com/tilde-lab/ml-playground/commit/050504c795d65e054cb325446063c2bdbc02e469).
- Added [surface adsorption](https://github.com/tilde-lab/ml-playground/commit/6b2f623303871a580c879ecc0ad278f6c2fd1478).


## 2023.11.19 - 2023.12.19
- [Metis](https://github.com/basf/metis-gui) installation completed. Metis(gui, bff, backend) has been studied, errors found and recorded in 'issue'.
- Completed tutorial about MPDS.

## 2024.01.29 - 2024.02.29
- Added example for using [Pcrystal](https://github.com/tilde-lab/ml-playground/commit/4fdac2c4e5c5fec08f989c2eb8d3f393d6f59e7f) and [Metis-client](https://github.com/tilde-lab/ml-playground/commit/ff607197b75dbf843226191f17cdd51604a7fdb9).
- Added [data processing from MPDS](https://github.com/tilde-lab/ml-playground/commit/78b2c55af381762ba40a76a551e40495ee678ed2).
- Added [GCN model](https://github.com/tilde-lab/ml-playground/blob/master/models/GCN/gcn_regression_model.py). Trained on several types of data to predict Seebeck coefficient.

## 2024.03.01 - 2024.03.29
- Added and trained [GAT](https://github.com/tilde-lab/ml-playground/commit/61e73261661967ab63af72c7ed33dfc0636ab19e), [Transformer](https://github.com/tilde-lab/ml-playground/commit/ece88e86203bfdb33dc5dc0a0f43f9c7bb8e64a3) models.
- Trained GCN models to predict Seebeck coefficient.
- Added getting and preparation AB_INITIO structures from MPDS.
- Added [ordering to disordered structures](https://github.com/tilde-lab/ml-playground/blob/master/data_massage/seebeck_coefficient_and_structure/structure_to_vectors.py).
- Added [CrystalGraphVectorsDataset](https://github.com/tilde-lab/ml-playground/blob/master/datasets/vectors_graph_dataset.py).

## 2024.03.30 - 2024.04.29
- Complete refactoring done, updated structure in the [ml-selection repo](https://github.com/tilde-lab/ml-selection/commits/master/).
- Added tests.
- Added processing plyhedra. Added Polyhedra dataset. Trained models on it.
- Added CNN model.

