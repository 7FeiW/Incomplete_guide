# AllianceCan / Compute Canada / DRAC

This document gives a short, practical overview of the national research-computing organizations and common workflows you will encounter on Canadian HPC resources, plus quick examples for finding software and using containers with Apptainer.

## What is Alliance (AllianceCan / DRAC)

The Digital Research Alliance of Canada (often shortened to the Alliance or AllianceCan) is the national organization that provides advanced research computing (HPC), cloud, storage, and support services for academic research across Canada. 

## Find Software/Library

On this HPC systems you typically do not install system-wide packages yourself. Instead, software is provided through a module system (Environment Modules or Lmod).

- List available modules:

```
module avail
```

- Search for a module or package (Lmod/spider):

```
module spider python
```

- Load a module into your session:

```
module load python/3.12
```

- standard enveriment, this is the one loaded by default 
```
module load StdEnv/2023
```

- cuda module - which is NOT required with pytorch as pytorch use pip installed cuda package
```
module load cuda/12.9
```

## Find Python Package

For details refer to this [[https://docs.alliancecan.ca/wiki/Python]]

## Apptainer

Apptainer (formerly Singularity) is the container runtime commonly used on HPC systems because it runs containers without requiring root privileges and integrates with shared filesystems and batch schedulers.

Basic concepts:
- Images: typically `.sif` (Singularity Image Format) files.
- Execution: run commands inside an image with `apptainer exec` or open an interactive shell with `apptainer shell`.

Examples:

- Run a command inside an image:

```
apptainer exec image.sif python script.py
```

- Start an interactive shell inside the container:

```
apptainer shell image.sif
```

- Bind host directories into the container (common on HPC):

```
apptainer exec --bind /home/youruser/data:/data image.sif python /data/script.py
```

- Pull an image from a registry (Docker Hub or Singularity Library):

```
apptainer pull docker://python:3.9
```

Notes on building images:
- `apptainer build image.sif docker://...` can create an image from a Docker source but often requires root; many centers provide remote or service-based builders.
- If you cannot build on the cluster, build locally (or use `apptainer pull`) and transfer the `.sif` file to the HPC system.

Using Apptainer in batch jobs (example Slurm script snippet):

```
#SBATCH --job-name=python-job
module load apptainer
apptainer exec --bind $HOME/data:/data image.sif python /data/script.py
```
