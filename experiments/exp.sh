#!/bin/bash

# Run the Python script with the specified dataset
python experiment_main.py --name_dataset Cora --filename 2d_sampling_Cora --out_dim 2
python experiment_main.py --name_dataset Cora --filename 3d_sampling_Cora --out_dim 3
python experiment_main.py --name_dataset Pubmed --filename 2d_sampling_Pubmed --out_dim 2
python experiment_main.py --name_dataset Pubmed --filename 3d_sampling_Pubmed --out_dim 3
python experiment_main.py --name_dataset Citeseer --filename 2d_sampling_Citeseer --out_dim 2
python experiment_main.py --name_dataset Citeseer --filename 3d_sampling_Citeseer --out_dim 3

# In terminal
# chmod +x exp.sh
# bash exp.sh