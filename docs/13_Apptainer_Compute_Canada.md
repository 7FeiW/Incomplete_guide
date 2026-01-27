# Using Apptainer on Compute Canada

This guide covers practical workflows for using **Apptainer** (formerly Singularity) on Compute Canada clusters (Cedar, Béluga, Graham, Narval, Niagara). It focuses on containerizing research workflows, GPU applications, and running reproducible experiments on HPC infrastructure.

## Table of Contents

### Part 1: Apptainer Basics
1. [What is Apptainer?](#what-is-apptainer)
2. [Why Use Apptainer on Compute Canada?](#why-use-apptainer-on-compute-canada)
3. [Apptainer vs Docker](#apptainer-vs-docker)

### Part 2: Getting Started
4. [Compute Canada Cluster Overview](#compute-canada-cluster-overview)
5. [Loading Apptainer Module](#loading-apptainer-module)
6. [Basic Apptainer Commands](#basic-apptainer-commands)

### Part 3: Building Containers
7. [Building from Docker Hub](#building-from-docker-hub)
8. [Writing Definition Files](#writing-definition-files)
9. [Building for GPU Applications](#building-for-gpu-applications)
10. [Building ML/DL Containers](#building-mldl-containers)

### Part 4: Running Containers
11. [Running Interactive Sessions](#running-interactive-sessions)
12. [Running Batch Jobs with SLURM](#running-batch-jobs-with-slurm)
13. [GPU Jobs with Apptainer](#gpu-jobs-with-apptainer)
14. [Binding Directories and Data](#binding-directories-and-data)

### Part 5: Best Practices
15. [Storage and Performance](#storage-and-performance)
16. [Reproducibility and Versioning](#reproducibility-and-versioning)
17. [Common Issues and Solutions](#common-issues-and-solutions)

### Part 6: Domain-Specific Examples
18. [Machine Learning Workflows](#machine-learning-workflows)
   - 18.1 [Complete PyTorch Training Pipeline](#complete-pytorch-training-pipeline)
   - 18.2 [Large Language Model (LLM) Inference](#large-language-model-llm-inference)
19. [Computational Biology Examples](#computational-biology-examples)
20. [Python Research Workflows](#python-research-workflows)

---

# Part 1: Apptainer Basics

## What is Apptainer?

**Apptainer** (formerly Singularity) is a container platform designed for HPC environments. Unlike Docker, Apptainer:
- Runs without root privileges (perfect for shared clusters)
- Uses a single-file container format (.sif files)
- Supports GPU pass-through seamlessly
- Integrates with HPC schedulers (SLURM)
- Maintains user permissions inside containers

**Key terminology**:
- **Container**: A packaged environment with all dependencies
- **Image/SIF file**: The container file (e.g., `pytorch.sif`)
- **Definition file**: Recipe for building containers (`.def` file)
- **Sandbox**: Writable directory-based container for development

---

## Why Use Apptainer on Compute Canada?

**Benefits**:
1. **Reproducibility**: Package entire software stack (Python, CUDA, libraries) in one file
2. **Portability**: Same container runs on Cedar, Béluga, Graham, Narval
3. **Dependency management**: Avoid module conflicts and version issues
4. **GPU support**: Direct access to cluster GPUs without driver conflicts
5. **Custom environments**: Install any software without admin rights
6. **Collaboration**: Share exact environments with collaborators
7. **Bypass HPC restrictions**: Install packages freely (no `--no-index` limitations)
8. **GPU compatibility**: Isolate from H100:20G vs H100:30G driver variations

**Use cases**:
- Deep learning with specific PyTorch/TensorFlow versions
- **Large language models (LLMs)**: vLLM, Transformers with strict CUDA requirements
- Molecular dynamics with custom-compiled GROMACS
- Bioinformatics pipelines with complex dependencies
- Python environments with conflicting package requirements
- Reproducible experiments for publications

---

## Apptainer vs Docker

| Feature | Apptainer | Docker |
|---------|-----------|--------|
| Root privileges | Not required | Required |
| HPC support | Excellent | Poor |
| File format | Single .sif file | Layered images |
| GPU support | Native | Requires nvidia-docker |
| User permissions | Preserved | Root inside container |
| SLURM integration | Built-in | Manual |
| Compute Canada | Fully supported | Not allowed |

**Can I use Docker images?** Yes! Apptainer can convert Docker images:
```bash
apptainer build mycontainer.sif docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

---

# Part 2: Getting Started

## Compute Canada Cluster Overview

**Available clusters**:

| Cluster | Location | Best For | GPU Types |
|---------|----------|----------|-----------|
| **Cedar** | Simon Fraser | General purpose, GPUs | P100, V100 |
| **Béluga** | McGill, ÉTS, Polytechnique | CPU-intensive, some GPUs | V100 |
| **Graham** | Waterloo | General purpose, GPUs | P100, V100 |
| **Narval** | ÉTS | Large memory, GPUs | A100 |
| **Niagara** | Toronto | Large parallel jobs | No GPUs |

**Storage**:
- **Home** (`$HOME`): 50 GB, backed up, slow I/O (for code, scripts)
- **Project** (`$PROJECT`): Shared, backed up, medium I/O (for data, containers)
- **Scratch** (`$SCRATCH`): Fast, NOT backed up, purged after 60 days (for temporary data)

**Important**: Store `.sif` container files in `$PROJECT` or `$HOME`, NOT `$SCRATCH` (auto-deleted).

---

## Loading Apptainer Module

On Compute Canada, Apptainer is available as a module:

```bash
# Check available versions
module spider apptainer

# Load Apptainer
module load apptainer/1.1.8

# Verify installation
apptainer --version
```

**Add to your `.bashrc`** for automatic loading:
```bash
echo "module load apptainer/1.1.8" >> ~/.bashrc
source ~/.bashrc
```

---

## Basic Apptainer Commands

### Pulling Images from Docker Hub

```bash
# Pull PyTorch image
apptainer pull docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Rename during pull
apptainer pull pytorch.sif docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

### Running Commands in Containers

```bash
# Run command
apptainer exec mycontainer.sif python script.py

# Interactive shell
apptainer shell mycontainer.sif

# Run with GPU
apptainer exec --nv mycontainer.sif python train.py
```

### Inspecting Containers

```bash
# Show container metadata
apptainer inspect mycontainer.sif

# List files in container
apptainer exec mycontainer.sif ls /opt

# Check Python version
apptainer exec mycontainer.sif python --version
```

---

# Part 3: Building Containers

## Building from Docker Hub

**Method 1: Direct pull** (simplest)
```bash
# PyTorch
apptainer pull pytorch.sif docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# TensorFlow
apptainer pull tensorflow.sif docker://tensorflow/tensorflow:2.12.0-gpu

# NVIDIA CUDA base
apptainer pull cuda.sif docker://nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# Bioinformatics
apptainer pull biopython.sif docker://biocontainers/biopython:v1.79_cv1
```

**Method 2: Build with definition file** (for customization)
```bash
# Create pytorch.def
apptainer build pytorch_custom.sif pytorch.def
```

---

## Writing Definition Files

Definition files (`.def`) are recipes for building containers.

### Example 1: Basic Python Container

```def
Bootstrap: docker
From: python:3.10-slim

%post
    # Update and install system packages
    apt-get update && apt-get install -y \
        build-essential \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

    # Install Python packages
    pip install --no-cache-dir \
        numpy \
        pandas \
        scikit-learn \
        matplotlib \
        seaborn

%environment
    export LC_ALL=C
    export PYTHONUNBUFFERED=1

%runscript
    exec python "$@"

%labels
    Author your.name@institution.ca
    Version v1.0
    Description Basic Python scientific computing environment

%help
    This container provides Python 3.10 with numpy, pandas, scikit-learn.

    Usage:
        apptainer exec container.sif python script.py
```

**Build it**:
```bash
# On Compute Canada, use --fakeroot (no root needed)
apptainer build --fakeroot python_sci.sif python_sci.def
```

---

### Example 2: PyTorch ML Container

```def
Bootstrap: docker
From: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

%post
    # Install additional ML packages
    pip install --no-cache-dir \
        torchvision==0.15.0 \
        torchaudio==2.0.0 \
        transformers==4.28.0 \
        accelerate==0.18.0 \
        datasets==2.11.0 \
        wandb==0.15.0 \
        tensorboard==2.12.0 \
        optuna==3.1.0 \
        scikit-learn==1.2.2 \
        pandas==2.0.0 \
        matplotlib==3.7.1

    # Install development tools
    apt-get update && apt-get install -y \
        git \
        vim \
        htop \
        && rm -rf /var/lib/apt/lists/*

%environment
    export PYTHONUNBUFFERED=1
    export CUDA_VISIBLE_DEVICES=0

%runscript
    exec python "$@"

%labels
    Author researcher@university.ca
    Version v1.0
    Description PyTorch 2.0 with transformers and experiment tracking

%help
    PyTorch 2.0 container with GPU support.

    Usage:
        # CPU
        apptainer exec container.sif python train.py

        # GPU
        apptainer exec --nv container.sif python train.py
```

---

### Example 3: Computational Biology Container

```def
Bootstrap: docker
From: ubuntu:22.04

%post
    # Install system dependencies
    apt-get update && apt-get install -y \
        python3 \
        python3-pip \
        build-essential \
        wget \
        curl \
        git \
        libz-dev \
        libbz2-dev \
        liblzma-dev \
        && rm -rf /var/lib/apt/lists/*

    # Install BioPython and scientific stack
    pip3 install --no-cache-dir \
        biopython==1.81 \
        numpy==1.24.0 \
        pandas==2.0.0 \
        scipy==1.10.0 \
        matplotlib==3.7.0 \
        seaborn==0.12.0

    # Install bioinformatics tools
    pip3 install --no-cache-dir \
        pysam==0.21.0 \
        pyvcf3==1.0.3 \
        scikit-bio==0.5.8

    # Install command-line tools
    apt-get update && apt-get install -y \
        samtools \
        bcftools \
        bedtools \
        && rm -rf /var/lib/apt/lists/*

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8

%runscript
    exec python3 "$@"

%labels
    Author bioinformatics@lab.ca
    Version v1.0
    Description Bioinformatics pipeline container

%help
    Bioinformatics container with BioPython and common tools.

    Tools included:
    - BioPython, pysam, scikit-bio
    - samtools, bcftools, bedtools

    Usage:
        apptainer exec container.sif python3 analysis.py
```

---

## Building for GPU Applications

**Important**: Match CUDA version to cluster GPUs:
- Cedar/Graham P100/V100: CUDA 11.x
- Narval A100: CUDA 11.x or 12.x
- Béluga V100: CUDA 11.x

### GPU-Enabled PyTorch

```def
Bootstrap: docker
From: nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

%post
    # Install Python
    apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        && rm -rf /var/lib/apt/lists/*

    # Install PyTorch with CUDA 11.7
    pip3 install --no-cache-dir \
        torch==2.0.0+cu117 \
        torchvision==0.15.0+cu117 \
        torchaudio==2.0.0+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117

%environment
    export CUDA_VISIBLE_DEVICES=0

%test
    # Verify CUDA is available
    python3 -c "import torch; assert torch.cuda.is_available()"

%labels
    Author ml@researcher.ca
    CUDA 11.7
    PyTorch 2.0
```

**Build and test**:
```bash
apptainer build --fakeroot pytorch_gpu.sif pytorch_gpu.def

# Test GPU detection
apptainer exec --nv pytorch_gpu.sif python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## Building ML/DL Containers

### Complete ML Stack with Experiment Tracking

```def
Bootstrap: docker
From: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

%post
    # ML frameworks
    pip install --no-cache-dir \
        torch==2.0.0 \
        torchvision==0.15.0 \
        transformers==4.28.0 \
        accelerate==0.18.0

    # Experiment tracking
    pip install --no-cache-dir \
        mlflow==2.3.0 \
        wandb==0.15.0 \
        tensorboard==2.12.0

    # Hyperparameter tuning
    pip install --no-cache-dir \
        optuna==3.1.0 \
        ray[tune]==2.4.0

    # Data processing
    pip install --no-cache-dir \
        pandas==2.0.0 \
        numpy==1.24.0 \
        scikit-learn==1.2.2 \
        datasets==2.11.0

    # Visualization
    pip install --no-cache-dir \
        matplotlib==3.7.1 \
        seaborn==0.12.2 \
        plotly==5.14.0

%environment
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false

%labels
    Description Complete ML stack with experiment tracking
    PyTorch 2.0
    CUDA 11.7
```

---

# Part 4: Running Containers

## Running Interactive Sessions

### Request Interactive Session with GPU

```bash
# Request 1 GPU for 3 hours
salloc --account=def-advisor --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=3:00:00

# Load Apptainer
module load apptainer/1.1.8

# Run interactive shell
apptainer shell --nv /project/def-advisor/containers/pytorch.sif

# Inside container
python
>>> import torch
>>> torch.cuda.is_available()
True
```

### Run Script Interactively

```bash
# In interactive session
apptainer exec --nv /project/def-advisor/containers/pytorch.sif python train.py
```

---

## Running Batch Jobs with SLURM

### Example 1: Basic Python Job

**File**: `run_analysis.sh`
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --job-name=analysis
#SBATCH --output=%x-%j.out

# Load Apptainer
module load apptainer/1.1.8

# Set paths
CONTAINER=/project/def-advisor/containers/python_sci.sif
SCRIPT=$HOME/projects/analysis/process_data.py

# Run script in container
apptainer exec $CONTAINER python $SCRIPT --input $1 --output $2

echo "Job completed at $(date)"
```

**Submit job**:
```bash
chmod +x run_analysis.sh
sbatch run_analysis.sh input.csv output.csv
```

---

### Example 2: PyTorch Training Job

**File**: `train_model.sh`
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --gres=gpu:v100:1           # 1 V100 GPU
#SBATCH --cpus-per-task=6           # 6 CPU cores
#SBATCH --mem=32G                   # 32 GB RAM
#SBATCH --time=12:00:00             # 12 hours
#SBATCH --job-name=train_resnet
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Load modules
module load apptainer/1.1.8

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Paths
CONTAINER=/project/def-advisor/containers/pytorch_ml.sif
WORK_DIR=$HOME/projects/image_classification
DATA_DIR=/project/def-advisor/datasets/imagenet

# Bind directories
BIND_DIRS="$WORK_DIR:/workspace,$DATA_DIR:/data"

# Run training
apptainer exec \
    --nv \
    --bind $BIND_DIRS \
    $CONTAINER \
    python /workspace/train.py \
        --data-dir /data \
        --epochs 100 \
        --batch-size 256 \
        --lr 0.001 \
        --output /workspace/checkpoints

echo "Training completed at $(date)"
```

**Submit**:
```bash
mkdir -p logs
sbatch train_model.sh
```

---

### Example 3: Multi-GPU Distributed Training

**File**: `train_distributed.sh`
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --nodes=1                   # Single node
#SBATCH --gres=gpu:4                # 4 GPUs
#SBATCH --cpus-per-task=16          # 16 CPU cores
#SBATCH --mem=128G                  # 128 GB RAM
#SBATCH --time=24:00:00
#SBATCH --job-name=ddp_training
#SBATCH --output=logs/%x-%j.out

# Load modules
module load apptainer/1.1.8

# Set environment for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export WORLD_SIZE=4

# Container and paths
CONTAINER=/project/def-advisor/containers/pytorch_ml.sif
WORK_DIR=$HOME/projects/transformer_training

# Run with torchrun
apptainer exec \
    --nv \
    --bind $WORK_DIR:/workspace \
    $CONTAINER \
    torchrun \
        --nproc_per_node=4 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        /workspace/train_ddp.py \
            --batch-size 64 \
            --epochs 50

echo "Distributed training completed"
```

---

## GPU Jobs with Apptainer

### Always Use `--nv` Flag for GPUs

```bash
# WRONG - GPU not accessible
apptainer exec container.sif python train.py

# CORRECT - GPU accessible
apptainer exec --nv container.sif python train.py
```

### Verify GPU Access

```bash
# Check NVIDIA driver
apptainer exec --nv container.sif nvidia-smi

# Check PyTorch CUDA
apptainer exec --nv container.sif python -c "import torch; print(torch.cuda.is_available())"

# Check TensorFlow GPU
apptainer exec --nv container.sif python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Selecting Specific GPUs

```bash
# Use first GPU only
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv container.sif python train.py

# Use GPUs 2 and 3
CUDA_VISIBLE_DEVICES=2,3 apptainer exec --nv container.sif python train.py
```

---

## Binding Directories and Data

By default, Apptainer binds:
- `$HOME`
- `/tmp`
- `/proc`
- `/sys`
- `/dev`
- Current working directory

**Bind additional directories** with `--bind` or `-B`:

```bash
# Single bind
apptainer exec --bind /project/def-advisor/data:/data container.sif python script.py

# Multiple binds
apptainer exec \
    --bind /project/def-advisor/data:/data \
    --bind /scratch/username/temp:/temp \
    container.sif python script.py

# Short syntax (multiple binds)
apptainer exec -B /project/data:/data,/scratch/temp:/temp container.sif python script.py
```

### Best Practice: Bind Project and Scratch

```bash
#!/bin/bash
# SLURM script

# Define bind paths
PROJECT_DATA=/project/def-advisor/datasets
SCRATCH_OUTPUT=$SCRATCH/experiment_001
HOME_CODE=$HOME/projects/myproject

BIND_DIRS="$PROJECT_DATA:/data,$SCRATCH_OUTPUT:/output,$HOME_CODE:/code"

# Run with binds
apptainer exec \
    --nv \
    --bind $BIND_DIRS \
    $CONTAINER \
    python /code/train.py \
        --data /data \
        --output /output
```

---

# Part 5: Best Practices

## Storage and Performance

### Container Storage Location

**Store containers in `$PROJECT`** (shared, backed up):
```bash
mkdir -p $PROJECT/containers
cd $PROJECT/containers
apptainer pull pytorch.sif docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
```

**NOT in `$SCRATCH`** (auto-deleted after 60 days!)

### Data Organization

```
$HOME/
├── projects/
│   └── myproject/
│       ├── scripts/          # Python scripts
│       ├── configs/          # Config files
│       └── slurm_jobs/       # SLURM job scripts

$PROJECT/def-advisor/
├── containers/               # Container .sif files
│   ├── pytorch_v2.sif
│   ├── tensorflow_v2.sif
│   └── bio_pipeline.sif
├── datasets/                 # Long-term datasets
│   └── imagenet/
└── results/                  # Important results

$SCRATCH/
└── tmp_experiment_001/       # Temporary data (deleted after 60 days)
    ├── checkpoints/
    └── logs/
```

### Performance Tips

1. **Use Scratch for I/O-intensive work**:
   ```bash
   # Copy data to scratch for faster I/O
   cp -r $PROJECT/datasets/large_dataset $SCRATCH/

   # Run with scratch data
   apptainer exec --bind $SCRATCH:/data container.sif python train.py --data /data/large_dataset
   ```

2. **Avoid reading many small files from Project**:
   - Tar datasets before copying: `tar -czf dataset.tar.gz dataset/`
   - Extract on Scratch: `tar -xzf dataset.tar.gz -C $SCRATCH/`

3. **Use .sif files, not sandbox directories**:
   - Faster to read
   - Single file to manage
   - Better for HPC storage

---

## Reproducibility and Versioning

### Version Your Containers

```bash
# Tag with version and date
apptainer build pytorch_v2.0_20240115.sif pytorch.def

# Keep definition files in version control
git add pytorch.def
git commit -m "Add PyTorch 2.0 container definition"
```

### Document Container Contents

Add to definition file:
```def
%labels
    Author researcher@university.ca
    Version v1.0
    Date 2024-01-15
    Python 3.10
    PyTorch 2.0
    CUDA 11.7
    Packages numpy-1.24 pandas-2.0 transformers-4.28

%help
    This container provides PyTorch 2.0 with CUDA 11.7.

    Installed packages:
    - PyTorch 2.0.0
    - torchvision 0.15.0
    - transformers 4.28.0
    - numpy 1.24.0
    - pandas 2.0.0

    Usage:
        apptainer exec --nv container.sif python train.py
```

### Include Exact Package Versions

```def
%post
    # Pin exact versions for reproducibility
    pip install --no-cache-dir \
        torch==2.0.0 \
        torchvision==0.15.0 \
        numpy==1.24.0 \
        pandas==2.0.0
```

---

## Common Issues and Solutions

### Issue 1: "FATAL: container creation failed: mount /proc/self/fd/3->/usr/local/var/apptainer/mnt/session/rootfs error"

**Cause**: Insufficient space in `/tmp` or `$HOME`

**Solution**:
```bash
# Set Apptainer temp directory to scratch
export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp
mkdir -p $APPTAINER_TMPDIR

# Add to ~/.bashrc
echo "export APPTAINER_TMPDIR=$SCRATCH/apptainer_tmp" >> ~/.bashrc
```

---

### Issue 2: "CUDA driver version is insufficient"

**Cause**: Container CUDA version > cluster driver version

**Solution**: Use older CUDA version in container
```bash
# Check cluster CUDA driver
nvidia-smi

# Build with compatible CUDA version
apptainer pull docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime  # not 12.x
```

---

### Issue 3: "cannot verify /project signature: you must first execute 'module load StdEnv'"

**Cause**: StdEnv module not loaded

**Solution**:
```bash
# Load standard environment
module load StdEnv/2020

# Add to ~/.bashrc
echo "module load StdEnv/2020" >> ~/.bashrc
```

---

### Issue 4: Permission denied when accessing `/project` or `/scratch`

**Cause**: Directory not bound, or wrong bind path

**Solution**:
```bash
# Verify binds
apptainer exec --bind /project:/project container.sif ls /project

# Check SLURM job
echo $PROJECT
echo $SCRATCH
```

---

### Issue 5: Container runs but GPU not detected

**Cause**: Missing `--nv` flag

**Solution**:
```bash
# Add --nv for GPU access
apptainer exec --nv container.sif python train.py
```

---

# Part 6: Domain-Specific Examples

## Machine Learning Workflows

### Complete PyTorch Training Pipeline

**Definition file**: `pytorch_training.def`
```def
Bootstrap: docker
From: pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

%post
    pip install --no-cache-dir \
        torchvision==0.15.0 \
        transformers==4.28.0 \
        datasets==2.11.0 \
        tensorboard==2.12.0 \
        scikit-learn==1.2.2 \
        wandb==0.15.0

%environment
    export PYTHONUNBUFFERED=1
    export HF_HOME=/workspace/.cache/huggingface
```

**Training script**: `train_model.py`
```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST('/data', train=True, download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Save
    torch.save(model.state_dict(), '/output/model.pth')

if __name__ == '__main__':
    train()
```

**SLURM job**: `submit_training.sh`
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/train-%j.out

module load apptainer/1.1.8

CONTAINER=$PROJECT/containers/pytorch_training.sif
DATA_DIR=$PROJECT/datasets/mnist
OUTPUT_DIR=$SCRATCH/mnist_output

mkdir -p $OUTPUT_DIR

apptainer exec \
    --nv \
    --bind $DATA_DIR:/data,$OUTPUT_DIR:/output \
    $CONTAINER \
    python train_model.py
```

---

### Large Language Model (LLM) Inference

**Why Apptainer is ideal for LLM workloads on HPC**:

Apptainer solves two critical challenges when running LLMs on Compute Canada clusters:

1. **Python Package Limitations**: HPC systems often restrict `pip install` with `--no-index` flags, preventing installation of newer packages. Apptainer containers bypass this completely by packaging all dependencies inside the container.

2. **GPU Compatibility Issues**: Different H100 GPU variants (H100:20G vs H100:30G) can cause driver/CUDA version conflicts. Apptainer containers include the exact CUDA/cuDNN versions needed, isolating you from cluster-level GPU driver variations.

This is particularly important for LLM frameworks (vLLM, Transformers, Flash Attention) which have strict CUDA version requirements and frequent updates.

---

#### vLLM Inference Container

**Use case**: Deploy large language models (Llama, Mistral, Qwen) for efficient inference with vLLM.

This example is from the repository at [`examples/apptainer_examples/vllm_apptainer.def`](../examples/apptainer_examples/vllm_apptainer.def).

**Definition file**: `vllm_inference.def`
```def
Bootstrap: docker
From: verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2

%post
    # Install additional python packages
    pip install --no-cache-dir pandas==2.2.2 sentence_transformers==5.1.2 bitsandbytes==0.48.2
    # flashinfer_python==0.2.5 causes issues with latest vLLM, so not installing it here
    # These are for GPU and CPU monitoring
    pip install --no-cache-dir psutil

%environment
    # Prevent Python from using user site-packages from /home/<user>/.local
    export PYTHONNOUSERSITE=1
    # Suppress FutureWarning for pynvml
    export PYTHONWARNINGS="ignore::FutureWarning"
    # Prevent xalt bind
    export LD_PRELOAD=
```

**Why these environment settings matter**:
- `PYTHONNOUSERSITE=1`: Prevents conflicts with user-installed packages in `~/.local/lib/python*/site-packages`, ensuring container uses only its packaged dependencies
- `PYTHONWARNINGS="ignore::FutureWarning"`: Suppresses noisy warnings from GPU monitoring libraries
- `LD_PRELOAD=`: Disables XALT (Compute Canada's job tracking tool) which can interfere with custom CUDA libraries

---

**Build the container**:
```bash
# On a compute node with internet (not login node)
salloc --time=1:00:00 --mem=16G --cpus-per-task=4

module load apptainer/1.1.8

apptainer build vllm_inference.sif vllm_inference.def
```

**Why this base image?**
- `verlai/verl` includes compatible versions of:
  - **vLLM 0.10.0**: Optimized LLM inference engine (PagedAttention, continuous batching)
  - **Transformers 4.55.4**: Latest Hugging Face models support
  - **Megatron-Core 0.13.0**: Tensor/pipeline parallelism for multi-GPU
  - **TransformerEngine 2.2**: FP8 quantization for H100 GPUs

This combination is tested to work together, avoiding version conflicts common with manual installs on HPC.

---

**Inference script**: `run_inference.py`
```python
import torch
from vllm import LLM, SamplingParams

def run_inference(model_name, prompts):
    """
    Run LLM inference with vLLM optimizations.

    Args:
        model_name: HuggingFace model ID or local path
        prompts: List of input prompts
    """
    # Initialize LLM with vLLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),  # Use all GPUs
        dtype="auto",  # Auto-detect best dtype (fp16/bf16/fp8)
        max_model_len=4096,  # Context length
        gpu_memory_utilization=0.9  # Use 90% of GPU memory
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512
    )

    # Generate
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}\n")

if __name__ == "__main__":
    model_name = "/scratch/models/Meta-Llama-3-8B-Instruct"
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to compute fibonacci numbers:"
    ]

    run_inference(model_name, prompts)
```

---

**SLURM job for single-GPU inference**:
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --gres=gpu:h100:1          # Single H100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/inference-%j.out

module load apptainer/1.1.8

# Paths
CONTAINER=$PROJECT/containers/vllm_inference.sif
MODEL_DIR=$SCRATCH/models
WORK_DIR=$HOME/projects/llm_inference

# Run inference
apptainer exec \
    --nv \
    --bind $MODEL_DIR:/scratch/models,$WORK_DIR:/workspace \
    $CONTAINER \
    python /workspace/run_inference.py
```

---

**SLURM job for multi-GPU inference (Tensor Parallelism)**:
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --gres=gpu:h100:4          # 4x H100 GPUs for large models
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=logs/inference_multi-%j.out

module load apptainer/1.1.8

# Paths
CONTAINER=$PROJECT/containers/vllm_inference.sif
MODEL_DIR=$SCRATCH/models/Meta-Llama-3-70B-Instruct  # Large 70B model
WORK_DIR=$HOME/projects/llm_inference

# Multi-GPU inference with tensor parallelism
apptainer exec \
    --nv \
    --bind $MODEL_DIR:/scratch/models,$WORK_DIR:/workspace \
    $CONTAINER \
    python /workspace/run_inference.py
```

**Key parameters for multi-GPU**:
- `tensor_parallel_size=4` in the script splits model across 4 GPUs
- vLLM automatically handles tensor sharding (no manual code needed)
- Requires GPUs to be on the same node (use `--nodes=1`)

---

#### Fine-tuning LLMs with LoRA

**Use case**: Fine-tune large language models efficiently using LoRA (Low-Rank Adaptation).

**Fine-tuning script**: `finetune_lora.py`
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def finetune_with_lora(model_name, dataset_name, output_dir):
    """
    Fine-tune LLM with LoRA for parameter-efficient training.
    """
    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # Apply LoRA to attention layers
    )

    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # Shows only ~1% of params are trainable

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # Use bfloat16 for H100
        logging_steps=10,
        save_strategy="epoch"
    )

    # Train (using Trainer from transformers)
    # ... training loop here ...

    # Save LoRA adapter (only a few MB!)
    model.save_pretrained(f"{output_dir}/lora_adapter")

if __name__ == "__main__":
    finetune_with_lora(
        model_name="/scratch/models/Meta-Llama-3-8B",
        dataset_name="tatsu-lab/alpaca",
        output_dir="/scratch/llm_finetuned"
    )
```

**SLURM job for LoRA fine-tuning**:
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

module load apptainer/1.1.8

CONTAINER=$PROJECT/containers/vllm_inference.sif
MODEL_DIR=$SCRATCH/models
OUTPUT_DIR=$SCRATCH/llm_finetuned

mkdir -p $OUTPUT_DIR

apptainer exec \
    --nv \
    --bind $MODEL_DIR:/scratch/models,$OUTPUT_DIR:/scratch/llm_finetuned \
    $CONTAINER \
    python finetune_lora.py
```

**Why LoRA is ideal for HPC**:
- Reduces trainable parameters from billions to millions (~1% of model size)
- Fits large models (70B+) on fewer GPUs
- Adapter weights are only a few MB, easy to share and version control
- Multiple LoRA adapters can share the same base model

---

#### Troubleshooting LLM Workloads

**Issue: Out of Memory (OOM) on GPU**

**Solutions**:
1. **Reduce max_model_len**: Lower context length in vLLM
   ```python
   llm = LLM(model=model_name, max_model_len=2048)  # Instead of 4096
   ```

2. **Enable quantization**: Use 8-bit or 4-bit quantization
   ```python
   from transformers import BitsAndBytesConfig

   quantization_config = BitsAndBytesConfig(load_in_8bit=True)
   model = AutoModelForCausalLM.from_pretrained(
       model_name,
       quantization_config=quantization_config
   )
   ```

3. **Increase GPU memory utilization carefully**:
   ```python
   llm = LLM(model=model_name, gpu_memory_utilization=0.95)  # Default is 0.9
   ```

---

**Issue: CUDA version mismatch between H100:20G and H100:30G**

**Symptoms**:
```
CUDA error: invalid device function
NCCL error: unhandled cuda error
```

**Solution**: This is why Apptainer is valuable! The container includes compatible CUDA libraries:
- Base image has CUDA 11.7 (compatible with both H100 variants)
- Container's CUDA libraries take precedence over system CUDA
- No need to match exact cluster CUDA version

**Verification**:
```bash
# Check CUDA version inside container
apptainer exec --nv $CONTAINER nvcc --version

# Check PyTorch CUDA version
apptainer exec --nv $CONTAINER python -c "import torch; print(torch.version.cuda)"
```

---

**Issue: `pip install` fails with `--no-index` error**

**Example error**:
```
ERROR: Could not find a version that satisfies the requirement flash-attn==2.5.0
ERROR: No matching distribution found for flash-attn (from --no-index)
```

**Solution**: Install packages in the container definition, not at runtime:
```def
%post
    # Install directly in container build (no --no-index restriction)
    pip install --no-cache-dir flash-attn==2.5.0 --no-build-isolation
```

**Why this works**: Container builds run in isolated environments without HPC `pip` restrictions. Once built, all packages are frozen in the `.sif` file.

---

**Issue: vLLM can't find model files**

**Symptoms**:
```
FileNotFoundError: Model '/scratch/models/llama3' not found
```

**Solution**: Ensure model directory is bound correctly:
```bash
# Bind the parent directory containing models
apptainer exec \
    --nv \
    --bind $SCRATCH/models:/models \
    $CONTAINER \
    python run_inference.py

# Update script to use bound path
model_name = "/models/llama3"  # Not /scratch/models/llama3
```

**Debugging binds**:
```bash
# List available paths inside container
apptainer exec --bind $SCRATCH:/scratch $CONTAINER ls /scratch
```

---

## Computational Biology Examples

### BioPython Sequence Analysis

**Definition**: `biopython.def`
```def
Bootstrap: docker
From: python:3.10-slim

%post
    apt-get update && apt-get install -y \
        build-essential \
        libz-dev \
        && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir \
        biopython==1.81 \
        numpy==1.24.0 \
        pandas==2.0.0 \
        matplotlib==3.7.0
```

**Analysis script**: `analyze_sequences.py`
```python
from Bio import SeqIO
import pandas as pd
import sys

def analyze_fasta(input_file, output_csv):
    results = []

    for record in SeqIO.parse(input_file, "fasta"):
        results.append({
            'id': record.id,
            'length': len(record.seq),
            'gc_content': (record.seq.count('G') + record.seq.count('C')) / len(record.seq) * 100
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Analyzed {len(results)} sequences")

if __name__ == '__main__':
    analyze_fasta(sys.argv[1], sys.argv[2])
```

**SLURM job**: `run_analysis.sh`
```bash
#!/bin/bash
#SBATCH --account=def-advisor
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=1:00:00

module load apptainer/1.1.8

CONTAINER=$PROJECT/containers/biopython.sif
INPUT=$PROJECT/sequences/input.fasta
OUTPUT=$SCRATCH/results/analysis.csv

apptainer exec --bind $PROJECT:/project,$SCRATCH:/scratch $CONTAINER \
    python analyze_sequences.py $INPUT $OUTPUT
```

---

## Python Research Workflows

### Data Processing Pipeline

**Definition**: `data_pipeline.def`
```def
Bootstrap: docker
From: python:3.10

%post
    pip install --no-cache-dir \
        pandas==2.0.0 \
        numpy==1.24.0 \
        scipy==1.10.0 \
        scikit-learn==1.2.2 \
        jupyter==1.0.0
```

**Pipeline script**: `process_data.py`
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_dataset(input_csv, output_csv):
    # Load data
    df = pd.read_csv(input_csv)

    # Clean data
    df = df.dropna()

    # Feature engineering
    df['feature_ratio'] = df['feature_a'] / df['feature_b']

    # Normalize
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save
    df.to_csv(output_csv, index=False)
    print(f"Processed {len(df)} rows")

if __name__ == '__main__':
    import sys
    process_dataset(sys.argv[1], sys.argv[2])
```

---

## Additional Resources

### Compute Canada Documentation
- Apptainer on Compute Canada: https://docs.alliancecan.ca/wiki/Apptainer
- GPU usage: https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm
- Storage: https://docs.alliancecan.ca/wiki/Storage_and_file_management

### Apptainer Documentation
- Official docs: https://apptainer.org/docs/
- Definition file reference: https://apptainer.org/docs/user/latest/definition_files.html
- User guide: https://apptainer.org/docs/user/latest/

### Container Registries
- Docker Hub: https://hub.docker.com/
- NVIDIA NGC: https://catalog.ngc.nvidia.com/
- BioContainers: https://biocontainers.pro/

---

## Quick Command Reference

```bash
# Pull Docker image
apptainer pull docker://repo/image:tag

# Build from definition file
apptainer build --fakeroot container.sif definition.def

# Run command
apptainer exec container.sif command

# Run with GPU
apptainer exec --nv container.sif command

# Interactive shell
apptainer shell container.sif

# Bind directories
apptainer exec --bind /source:/dest container.sif command

# Inspect container
apptainer inspect container.sif

# Get help
apptainer run-help container.sif
```

---

**This guide covers the essentials of using Apptainer on Compute Canada. For cluster-specific details, always check the official Compute Canada documentation.**
