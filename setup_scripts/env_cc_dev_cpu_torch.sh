#!/bin/bash

set -e

source ~/.bashrc

module load StdEnv/2023
module load python/3.10

virtualenv LLM-DEMO

source LLM-DEMO/bin/activate
pip install --no-index --upgrade pip

pip install --no-index torch==2.2.0 \
    --no-index rdkit==2022.3.5 \
    --no-index torch_geometric==2.4.0 \
    --no-index torch_scatter==2.1.2 \
    --no-index torch_sparse==0.6.18 \
    --no-index torch_cluster==1.6.3 \
    --no-index torch_spline_conv==1.2.2 \
    --no-index cython==3.0.8 \
    --no-index cysignals \
    --no-index pandas \
    --no-index wandb \
    --no-index tqdm 
    
# this is only needed on CC because there is no internet 
# pip install wandb[sweeps]
pip install -I -e .