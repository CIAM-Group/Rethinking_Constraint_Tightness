

## [NeurIPS2025] Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees

This repository contains the code implementation of paper [Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees](https://openreview.net/forum?id=Lwn1rLB8t7). 
In this paper, We revisit the training setting of existing NCO models and find that models 
trained under these settings may overft certain specific constraint tightness. 
They suffer from severe performance degradation when applied to VRP instances with 
out-of-domain tightness degrees.
We propose a simple yet efficient training scheme that enables the NCO model to 
be efficient on a broad spectrum of constraint tightness degrees. 
Moreover, we propose a multi-expert module to enable the NCO model to 
learn a more effective policy for coping with diverse constraint tightness degrees.

### Dependencies
```bash
Python=3.8.20
matplotlib==3.5.2
numpy==1.23.3
pandas==1.5.1
pytz==2022.1
torch==2.4.1
torchaudio==0.12.1
torchvision==0.13.1
tqdm==4.64.1
```
Also see `packages.txt` for the remaining packages. If any package is missing, just install it following the prompts. 

### Datasets and pre-trained models
The training and test datasets can be downloaded from Google Drive:
```bash
https://drive.google.com/drive/folders/1ustzGdPht3CGS9HcLIJ2rjltUIivbPxN?usp=sharing
```

### Implementation

#### CVRP

```bash
1. Testing:
python CVRP/LEHD_main/CVRP/test_multi_capacity.py

2. Training:
python CVRP/LEHD_main/CVRP/train.py

3. Generate the CVRP instances with different capacity values and demand distributions:
see CVRP/Generate_data/README.md


```

#### CVRPTW

```bash
1. Testing:
cd CVRPTW/Routing-MVMoE-main-TW/
bash test.sh

2. Training:
cd CVRPTW/Routing-MVMoE-main-TW/
python train.py

3. Generate the CVRPTW instances with different TW tightness degrees:
cd CVRPTW/Generate_data
bash generate_data.sh
```


## Citation


```

@inproceedings{
2025rethinking,
title={Rethinking Neural Combinatorial Optimization for Vehicle Routing Problems with Different Constraint Tightness Degrees},
author={Fu Luo and Yaoxin Wu and and Zhi Zheng and Zhenkun Wang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=Lwn1rLB8t7}
}

```
****


## Acknowledgements

- https://github.com/RoyalSkye/Routing-MVMoE

- https://github.com/CIAM-Group/NCO_code/tree/main/single_objective/LEHD

- https://github.com/yd-kwon/POMO/tree/master


