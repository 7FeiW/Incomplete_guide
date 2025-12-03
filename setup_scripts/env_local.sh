#!/bin/bash

set -e

source ~/.bashrc

conda create -n LLM-DEMO-GPU
conda activate LLM-DEMO-GPU
conda install python=3.10

pip install torch \
    pandas \
    wandb \
    tqdm \
    scipy \
    matplotlib \
    unsloth==2024.10.1
    
# this is only needed on CC because there is no internet 
# pip install wandb[sweeps]
pip install -e .