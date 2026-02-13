# HTS-2026

[![GitHub license](https://img.shields.io/github/license/houtazak/HTS-2026)](https://github.com/houtazak/HTS-2026) [![GitHub release](https://img.shields.io/github/release/houtazak/HTS-2026.svg)](https://github.com/houtazak/HTS-2026/releases/) [![GitHub stars](https://img.shields.io/github/stars/houtazak/HTS-2026)](https://github.com/houtazak/HTS-2026/stargazers)
[![DOI](https://zenodo.org/badge/1157299604.svg)](https://zenodo.org/doi/10.5281/zenodo.18634454)

This repository contains FEM implementations of H-formulation for 2D and 3D High Temperature Superconductors (HTS) simulations using NGSolve, presented at HTS workshop 2026.

## 1) Quickstart

### Installation
Install the required packages in a new Python environment (version 3.13 or later, see `requirements.txt`). 

For doing so, open a terminal, create and activate a dedicated Python environment, using for instance `conda` (requires the installation of [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) ) : 

`conda create -n myenv python=3.13`

with `myenv` being replaced by your environment name.
Then, go to the folder where the code is located :

`cd C:\path\to\the\folder`

replacing `C:\path\to\the\folder` by the path to the local folder containing the code. The, install the required packages using 

`pip install -r requirements.txt`

Then, you should be able to run the scripts.

### Run the scripts
Then, execute one of the scripts on your favorite IDE within your newly created `myenv` environment :
- `2D_tape_external_field.py`
- `2D_tape_transport_current.py`
- `3D_bulk_external_field.py`

## 2) Contents

```
├── utils/   # meshes and nonlinear solver
│   ├── geometry.py
│   ├── solver.py
│   ├── mesh_comsol_2D.mphtxt
│   └── mesh_comsol_3D.mphtxt
│
| # Scripts to execute
├── 2D_tape_external_field.py 
├── 2D_tape_transport_current.py
├── 3D_bulk_external_field.py
│
| # Installation and instructions
├── requirements.txt # numpy, matplotlib, ngsolve
├── README.md
|
├── AUTHORS # Z. Houta, T. Cherrière, L. Quéval
└── LICENSE # LGPL
```
## 3) Citation

Please use Zenodo's DOI.

## 4) License

Copyright (C) 2026 Zakaria HOUTA (zakaria.houta@centralesupelec.fr), Théodore CHERRIERE (theodore.cherriere@centralesupelec.fr), Loïc QUEVAL (loic.queval@centralesupelec.fr)


This code is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License and of the GNU General Public License along with this code. If not, see <http://www.gnu.org/licenses/>. Please read their terms carefully and use this copy of the code only if you accept them.
