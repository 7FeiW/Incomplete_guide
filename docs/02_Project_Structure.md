# Project Structure

## Table of Contents

1. [Task-Oriented Project Setup](#task-oriented-project-setup)
2. [Multi-Task Project with Shared Common Code](#multi-task-project-with-shared-common-code)
3. [Large and Complex Research Project](#large-and-complex-research-project)
4. [General Best Practices](#general-best-practices)
5. [Working with SLURM](#working-with-slurm)
6. [Working with Apptainer](#working-with-apptainer)

Projects vary in goals, lifecycles, and complexity. Choose a structure that aligns with the primary purpose of the project rather than forcing a single canonical layout. Below are three common research project setups, each optimized for different use cases.

---

## Task-Oriented Project Setup

Use this structure when the project contains a small number of related experiments or preprocessing/training tasks. It keeps data, configurations, and scripts organized per task while remaining lightweight.

### Use This Structure When

1. The project has only a few tasks.

### Limitations

1. Tasks do not share many custom-built common functions.
2. There is no intention to distribute the project as an installable package (e.g., wheel).

```text
my_project/
├── data/                 # Data directory for this project
│   ├── task_1_data
│   └── task_2_data
├── docs/                 # Documentation (user guides, API docs)
├── scripts/              # Utility scripts for experiments
├── task_1/               # Task-specific scripts
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── task_2/               # Task-specific scripts
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── notebooks/            # Jupyter notebooks for exploration
│   ├── data_notebook.ipynb
│   └── result_notebook.ipynb
├── setup_scripts/
│   ├── hpc_setup.sh
│   └── linux_requirement.txt
├── configs/              # Configuration files
│   ├── config_task_1.json
│   └── config_task_2.json
├── .gitignore
├── .gitattributes
├── requirements.txt      # Dependencies
├── README.md
└── LICENSE
```

---

## Multi-Task Project with Shared Common Code

Use this structure when the project contains multiple tasks that share a meaningful amount of custom code. Shared modules live in a `common/` directory.

### Use This Structure When

1. There are several tasks.
2. Tasks share substantial custom code.

### Limitations

1. This example has no packaging configuration. A flat layout can still be distributed as a Python package after adding appropriate metadata and build configuration.
2. Shared-code imports need a documented launch convention. If scripts move into subdirectories or notebooks use a different environment, install the shared package into that environment rather than adding ad hoc import-path changes.

```text
my_project/
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── 01_task_1.py
├── 02_task_2.py
├── common/               # Shared library code
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
├── setup_scripts/
│   ├── hpc_setup.sh
│   └── linux_requirement.txt
├── configs/
│   ├── config_task_1.json
│   └── config_task_2.json
└── tests/
    └── test_core.py
```

---

## Large and Complex Research Project

Use a `src/`-based structure to separate importable package code from the repository root. This helps prevent imports from accidentally using the source-tree copy instead of the installed package. It is useful for small reusable packages as well as large research projects; project size and publication plans are not prerequisites. See [PyPA's comparison of source and flat layouts](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) for the import behavior and tradeoffs.

### Use This Structure When

1. The codebase is large, complex, or collaborative.
2. You plan to distribute or deploy your package.
3. The project involves many scripts, datasets, or models.
4. You are developing a sophisticated computational tool or ML model.

```text
my_project/
├── src/
│   └── my_package/       # Main Python package
│       ├── __init__.py
│       ├── utils.py
│       ├── config.py
│       └── models.py
├── tests/
│   └── test_main.py
├── data/
├── docs/                 # Project knowledge and research records
│   ├── README.md         # Index of the project documentation
│   ├── architecture.md  # Components, interfaces, and data flow
│   ├── plans/           # Task and experiment plans, progress, and next checks
│   ├── findings/        # Supported observations, negative results, and conclusions
│   ├── knowledge/       # Methods, terminology, and reusable procedures
│   ├── rules/           # Agreed scientific and engineering constraints
│   └── state/
│       └── project-state.md # Current priorities, blockers, and links to active plans
├── scripts/
├── preprocess_scripts/
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── notebooks/
│   ├── data_notebook.ipynb
│   └── result_notebook.ipynb
├── setup_scripts/
│   ├── hpc_setup.sh
│   └── linux_requirement.txt
├── configs/
├── .gitignore
├── .gitattributes
├── pyproject.toml        # Package metadata, dependencies, and build configuration
├── requirements.txt      # Optional export for environments that require it
├── README.md
└── LICENSE
```

### Keeping Plans and Findings in `docs/`

Research work needs a record of what was planned, what was tried, and what the evidence supports. Use `docs/` for these records as well as user and developer documentation. The subdirectories above are optional; create them as the project needs them.

For example, `docs/plans/compare-preprocessing.md` could describe a preprocessing comparison: its question, proposed experiments, completion criteria, progress, and next checks. Record decisions and their reasons in the relevant plan. After evaluating the runs, save supported observations and limitations in `docs/findings/preprocessing-comparison.md`, linking to the plan, code revision, configurations, and run outputs. Label untested ideas as hypotheses, and retain negative results when they inform future work.

Keep these Markdown records in Git and update their status and date when the work changes. Use `docs/state/project-state.md` as a short index of active work, and link established methods or architectural decisions from `docs/knowledge/` or `docs/architecture.md`. Keep each explanation in one place and link to it. Store raw logs, datasets, and checkpoints in the data and output locations described below, with references from the findings.

See [Project Record](15_Agentic_Workflow.md#project-record) for the detailed convention and plan template. This organization also works for projects developed without coding agents.

### Installing and Importing the Package

The directory tree alone is not an installable project: `pyproject.toml` must contain valid package metadata and build configuration. Follow [Python Environments and Packaging](04_Python_Env.md#pyprojecttoml-vs-requirementstxt) to configure it and create an isolated environment.

From the example project's root, with that environment active and `python` pointing to its interpreter, run the following in Bash or PowerShell:

```bash
python -m pip install -e .
```

This installs the local project and its declared dependencies into the active environment in editable mode, so edits to Python source are available without reinstalling. Packaging metadata changes may require another installation. See [pip's editable-install documentation](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs).

Scripts can then use `import my_package`; notebooks must use a kernel backed by the same environment. Installation makes the package importable, but does not set data or output paths. Document how scripts resolve those paths, such as relative to the project root or from explicit configuration.

---

## General Best Practices

Regardless of project size, keep the following guidelines in mind:

### Code Organization

* Place **core logic** into Python modules rather than notebooks or scripts.

  * Data processing routines
  * Model definitions
  * Long-term utilities
* Use `preprocess_scripts/` for data extraction and cleaning.
* Use `scripts/` for running training jobs or automated workflows.
* Use `notebooks/` for exploration, visualization, and prototyping.
* Store experiment-specific settings in `configs/` and track them using Git.

### Research Data and Run Outputs

Use separate locations for original inputs, derived datasets, and experiment outputs when that distinction helps the project. For example, extend any of the layouts above with:

```text
my_project/
├── data/
│   ├── README.md         # Sources, dataset versions, checksums, and retrieval steps
│   ├── raw/              # Original inputs; preserve without overwriting
│   └── processed/        # Derived datasets, organized by version or processing run
└── outputs/
    └── experiment_001/   # Example unique run identifier
        ├── config.json  # Resolved settings actually used, including seeds if applicable
        ├── metadata.json # Code version, input provenance, environment, and output location
        ├── metrics.csv
        └── run.log
```

Record the Git commit and any uncommitted changes, Python and dependency versions, and input dataset versions or checksums with each run. Give each run a distinct output directory so later experiments do not overwrite earlier results. See [Experiment Reproducibility](12_Go_With_Large_Scale.md#experiment-reproducibility) for the metadata to retain.

Keep small provenance records and retrieval instructions in Git. Large datasets and generated outputs can live outside the repository, with their locations supplied through configuration. If they live inside the repository, ignore the relevant data and output directories while retaining `data/README.md`. Git exclusion is not archival storage: preserve important outputs and their run records in a documented, durable location.

### Testing

* Use a `tests/` directory for unit tests.
* Prefer `pytest` for modern, flexible test workflows.
* Highly recommended for mathematically verifiable or rule‑based tasks.

### Environment Setup

* Place environment setup scripts (e.g., HPC bootstrap scripts) in `setup_scripts/`.
* For a simple collection of research scripts, a `requirements.txt` can describe the packages needed to run them. It does not replace package metadata for an installable distribution.
* For an installable package, declare its metadata and dependencies in `pyproject.toml`. If a deployment environment needs `requirements.txt`, generate it from the chosen dependency or lock workflow instead of maintaining competing dependency lists. See [Python Environments and Packaging](04_Python_Env.md#pyprojecttoml-vs-requirementstxt) and [PyPA's explanation of package dependencies versus requirements files](https://packaging.python.org/en/latest/discussions/install-requires-vs-requirements/).
* For platform-specific environments, document which requirement or environment file to use. Tasks with incompatible dependencies may need separate environments; splitting requirement files alone does not resolve a conflict when they are installed together.

### Git Hygiene

* Use `.gitignore` to exclude large data files, temporary outputs, and caches.
* Use `.gitattributes` to:

  * Enforce line ending consistency
  * Mark binary files
  * Customize merge and diff behaviors

### Documentation

* Use `README.md` for project overviews.
* Use `docs/` for extended documentation, plans, findings, and reusable project knowledge; see [Keeping Plans and Findings in `docs/`](#keeping-plans-and-findings-in-docs).

### Naming Conventions

* Number task scripts (`01_task.py`, `02_task.py`) to make the intended execution order visible. Filenames do not execute tasks or enforce dependencies; document the commands in order, or use a workflow runner that checks prerequisites and stops on failure.
* Use leading zeros when needed (e.g., `09` before `10`).

---

## Working with SLURM

For high-performance computing (HPC) systems using the SLURM job scheduler, these are optional project conventions. Adapt storage locations to the cluster's policies.

* `slurm_scripts/` holds maintained submission scripts or templates tracked in Git.
* `slurm_working_dir/` holds temporary files in a separate directory per job, ignored by Git when stored inside the repository. On a cluster, this may instead be a configured scratch location.
* `jobs/` holds logs and submission records grouped by job, also ignored by Git in this example. Archive records needed to reproduce important results with the experiment outputs.

```text
├── slurm_scripts/
│   ├── preprocess.sh
│   └── train.sh
├── slurm_working_dir/
│   └── job_12345/        # Temporary inputs and intermediate files for an example job
└── jobs/
    └── job_12345/
        ├── submitted.sh # Exact script submitted, retained as a run record
        └── stdout.out
```

The saved `submitted.sh` is a snapshot for provenance; edit the maintained scripts in `slurm_scripts/` for future jobs. Directory names do not configure SLURM: explicitly set the working directory and log destinations in the submission workflow. SLURM normally uses the submission working directory unless overridden with `--chdir`, and does not stage your data files for you. See the [official sbatch documentation](https://slurm.schedmd.com/sbatch.html).

Choose temporary and durable storage according to local retention and access rules, and preserve required results before temporary storage is cleaned. See [Data Strategy](12_Go_With_Large_Scale.md#data-strategy) for storage planning and [Running Batch Jobs with SLURM](13_Apptainer_Compute_Canada.md#running-batch-jobs-with-slurm) for container job examples; adapt their account and resource settings to your site.

---

## Working with Apptainer

If the project uses Apptainer containers, keep build recipes and helper scripts together in an optional `apptainer/` directory:

```text
apptainer/
├── environment.def      # Container build recipe
├── build.sh             # Bash helper: build an image from the recipe
├── run.sh               # Bash helper: run a command with the required paths
└── README.md            # Build/run instructions and image storage location
```

An [Apptainer definition file](https://apptainer.org/docs/user/latest/definition_files.html) describes the base image and container setup. Track this recipe and the helper scripts in Git. Store built `.sif` container images in a documented location outside ordinary Git tracking, and record the image checksum or immutable identifier with each experiment. The recipe alone does not identify the exact image used for a run.

Document the input and output paths the run helper exposes inside the container, along with its prerequisites. See [Writing Definition Files](13_Apptainer_Compute_Canada.md#writing-definition-files) and [Binding Directories and Data](13_Apptainer_Compute_Canada.md#binding-directories-and-data) for details. Storage paths, modules, and resource settings in that chapter are site-specific examples.
