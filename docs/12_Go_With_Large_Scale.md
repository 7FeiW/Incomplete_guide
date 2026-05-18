## Go With Large Scale

Use this checklist when planning and running experiments at scale (HPC clusters, cloud, or distributed systems). Each item is a minimal action that reduces risk and improves reproducibility when moving from small-scale experiments to large-scale runs.

- **Project plan & requirements**: define goals, expected data size, performance targets, and budget.

1. **Data strategy**:
  - Identify data sources and sizes, this should include
  - total size of data, and number of files
  - this should include dataset size, model checkpoint dataset
  - determind which data need keep on Compute System, which should be offload to local drive
  - determind how to sapreated data to support **parallalization**
  
2. **Compute resources**:

  - Decide on target compute resouce (HPC vs cloud), node types (CPU/GPU), and instance sizes.
  - Reserve allocations / request quotas; test on small and medium nodes before full runs.
  - Always get monitoring resouce used in the test run
  - Always get estmaite for total expected resouce usage

3. **Environment & packaging**:

  - Containerize (Apptainer/Docker) images for reproducibility; store images in `containers/` or a registry.
  - Provide locked environment files (`pyproject.toml`/`poetry.lock`, `requirements.txt`, or `environment.yml`).

4. **Experiment reproducibility**:

  - Save configs for each run, fix random seeds, and log environment metadata.
  - Use experiment tracking (MLflow, Weights & Biases, or plain logs) and store run artifacts.

5. **Monitoring, logging & alerts**:

  - Centralize logs and metrics; add alerts for failed jobs, long runtimes, or resource exhaustion.
  - Store training/validation metrics and attach to experiment runs.

6. **Profiling & optimization**:

  - Profile CPU/GPU utilization, memory, and IO on medium runs before scaling up.
  - Benchmark common operations and optimize bottlenecks.

7. **Documentation & runbooks**:

  - Write onboarding docs: how to reproduce an experiment, run the pipeline, and debug failures.
  - Keep runbooks for common failures and escalation paths.
