#!/bin/bash

set -e

source ~/.bashrc

conda create -n LLM-DEMO-GPU
conda activate LLM-DEMO-GPU
conda install python=3.10

pip install torch
pip install pandas
pip install wandb
pip install tqdm
pip install scipy
pip install matplotlib
pip install unsloth==2024.10.1

# this is only needed on CC because there is no internet 
# pip install wandb[sweeps]
pip install -e .