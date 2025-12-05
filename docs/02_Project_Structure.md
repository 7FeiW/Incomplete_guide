# Project Structures

## Table of Contents

1. [Task-Oriented Project Setup](#task-oriented-project-setup)
2. [Multi-Task Project with Shared Common Code](#multi-task-project-with-shared-common-code)
3. [Large and Complex Research Project](#large-and-complex-research-project)
4. [General Best Practices](#general-best-practices)
5. [Working with SLURM](#working-with-slurm)
6. [Working with Apptainer](#working-with-apptainer)

Projects vary in goals, lifecycles, and complexity. Choose a structure that aligns with the primary purpose of the project rather than forcing a single canonical layout. Below are three common research project setups, each optimized for different use cases.

---

## Task‑Oriented Project Setup

Use this structure when the project contains a small number of related experiments or preprocessing/training tasks. It keeps data, configurations, and scripts organized per task while remaining lightweight.

### Use this when:

1. The project has only a few tasks.

### Limitations:

1. Tasks do not share many custom-built common functions.
2. There is no intention to distribute the project as an installable package (e.g., wheel).

```
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

## Multi‑Task Project with Shared Common Code

Use this structure when the project contains multiple tasks that share a meaningful amount of custom code. Shared modules live in a `common/` directory.

### Use this when:

1. There are several tasks.
2. Tasks share substantial custom code.

### Limitations:

1. Not suitable for distributing as a Python package.
2. Importing shared code can be slightly cumbersome.

```
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

Use a `src/`-based structure when the project must scale, be maintained long-term, support collaboration, or be distributed (e.g., uploaded to PyPI). This layout supports modern development practices such as CI, structured testing, and modular design.

### Use this when:

1. The codebase is large, complex, or collaborative.
2. You plan to distribute or deploy your package.
3. The project involves many scripts, datasets, or models.
4. You are developing a sophisticated computational tool or ML model.

```
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
├── docs/
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
├── pyproject.toml        # Modern build/packaging configuration
├── requirements.txt
├── README.md
└── LICENSE
```

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

### Testing

* Use a `tests/` directory for unit tests.
* Prefer `pytest` for modern, flexible test workflows.
* Highly recommended for mathematically verifiable or rule‑based tasks.

### Environment Setup

* Place environment setup scripts (e.g., HPC bootstrap scripts) in `setup_scripts/`.
* Use a single `requirements.txt` for simple public distributions.
* For complex environments (e.g., HPC constraints, conflicting dependencies), use multiple requirement files or platform‑specific scripts.

### Git Hygiene

* Use `.gitignore` to exclude large data files, temporary outputs, and caches.
* Use `.gitattributes` to:

  * Enforce line ending consistency
  * Mark binary files
  * Customize merge and diff behaviors

### Documentation

* Use `README.md` for project overviews.
* Use `docs/` for extended documentation.

### Naming Conventions

* Number task scripts (`01_task.py`, `02_task.py`) to ensure predictable execution order.
* Use leading zeros when needed (e.g., `09` before `10`).

---

## Working with SLURM

Add these directories when developing on HPC systems using SLURM.

* **slurm_scripts/** — templates for SLURM submission scripts.
* **slurm_working_dir/** — working directory for active SLURM jobs (excluded via `.gitignore`).
* **jobs/** — outputs and logs from job runs (also ignored by Git).

```
├── slurm_scripts/
│   ├── script_1.sh
│   └── script_2.sh
├── slurm_working_dir/
│   ├── script_1.sh
│   └── script_2.sh
└── jobs/
    ├── job_out_1.out
    └── job_out_2.out
```

---

## Working with Apptainer

```
├── apptainer/
│   ├── script_1.sh
│   └── script_2.sh
```
