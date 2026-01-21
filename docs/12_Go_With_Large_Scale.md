## Go With Large Scale — Quick Checklist

Use this checklist when planning and running experiments at scale (HPC clusters, cloud, or distributed systems). Each item is a minimal action that reduces risk and improves reproducibility when moving from small-scale experiments to large-scale runs.

- **Project plan & requirements**: define goals, expected data size, performance targets, SLAs, and budget.

- **Architecture & design**: choose a modular architecture, separate library (`src/`) vs experiments, and define dataflow (ingest → preprocess → train/eval → export).

- **Data strategy**:
  - Identify data sources and sizes; choose storage (object store, network file system).
  - Version datasets (DVC or dataset registry) and record provenance.
  - Define retention and access policies.

- **Compute resources**:
  - Decide on target infra (HPC vs cloud), node types (CPU/GPU), and instance sizes.
  - Reserve allocations / request quotas; test on small and medium nodes before full runs.

- **Environment & packaging**:
  - Containerize (Apptainer/Docker) images for reproducibility; store images in `containers/` or a registry.
  - Provide locked environment files (`pyproject.toml`/`poetry.lock`, `requirements.txt`, or `environment.yml`).

- **Dependency & build automation**:
  - Pin critical dependencies; separate dev deps.
  - Provide `Makefile` or task scripts to build, test, and run.

- **Experiment reproducibility**:
  - Save configs for each run, fix random seeds, and log environment metadata.
  - Use experiment tracking (MLflow, Weights & Biases, or plain logs) and store run artifacts.

- **Job orchestration & schedulers**:
  - Prepare scheduler templates (Slurm/PBS) with resource requests, time limits, and retries.
  - Use checkpointing and resume logic for long jobs.

- **Monitoring, logging & alerts**:
  - Centralize logs and metrics; add alerts for failed jobs, long runtimes, or resource exhaustion.
  - Store training/validation metrics and attach to experiment runs.

- **Profiling & optimization**:
  - Profile CPU/GPU utilization, memory, and IO on medium runs before scaling up.
  - Benchmark common operations and optimize bottlenecks.

- **Checkpointing & fault tolerance**:
  - Save intermediate checkpoints to durable storage and test recovery flows.
  - Design jobs to be idempotent where possible.

- **Data & artifact storage**:
  - Keep large artifacts (models, checkpoints) in object storage with lifecycle policies.
  - Record artifact URIs in run metadata.

- **Security & secrets**:
  - Use secret managers for credentials; avoid embedding secrets in images or repo.
  - Restrict access to data and compute with IAM or cluster ACLs.

- **Cost management**:
  - Estimate costs, use spot/preemptible instances where acceptable, and monitor spend.
  - Use autoscaling and right-sizing to reduce waste.

- **Release & deployment**:
  - Tag reproducible runs, automate packaging and releases in CI, and publish artifacts to a registry.

- **Documentation & runbooks**:
  - Write onboarding docs: how to reproduce an experiment, run the pipeline, and debug failures.
  - Keep runbooks for common failures and escalation paths.

- **Post-run housekeeping**:
  - Archive experiment data and models, record provenance, and update dashboards/reports.