# Project Structures

Projects have different goals,lifecycles and programming complexity. Pick a structure that matches the project's primary goal rather than forcing one canonical layout. Below are three common research project setups:

## Task oriented Research Project Setup

This is setup for a task-oriented layout when the project contains a handful of related experiments or distinct preprocessing/training tasks. This layout keeps data, configs, and scripts organized per task while still being lightweight. 

Use this when you need to:
1. Create a minimal structure for a project with a few sub tasks
2. No intention to distributing your package as a wheel file (e.g., you don't need `pip install mypackage`).

```
my_project/
├── data/                 # data directory for this project
│   ├── task_1_data
│   └── task_2_data
├── docs/                 # Project documentation (e.g., user guides, API docs)
├── scripts/              # Utility scripts for experiments
├── task_1/   # Scripts for preprocessing data
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── task_2/   # Scripts for preprocessing data
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── notebooks/            # Jupyter notebooks for analysis and visualization
│   ├── data_notebook.ipynb
│   └── result_notebook.ipynb
├── setup_scripts/
│   ├── hpc_setup.sh
│   └── linux_requirement.txt
├── configs/              # configuretions
│   ├── config_task_1.json
│   └── config_task_2.json
├── .gitignore            # File to ignore in version control
├── .gitattributes        # File to Managing file encodings and Customizing merge and diff behavior 
├── requirements.txt      # Project dependencies (can be generated from pyproject.toml)
├── README.md             # Project description and instructions
└── LICENSE               # Project license
```

## Task oriented Research Project Setup

Use this when you need to:

```
my_project/
├── README.md             # Project description and instructions
├── requirements.txt      # Dependencies for the project
├── .gitignore            # Files to ignore in version control
├── .gitattributes        # Files to Managing file encodings and Customizing merge and diff behavior
├── main.py               # Main script for running the project
├── other_scripts.py      # Additional utility scripts
├── notebook.ipynb        # Jupyter notebook for analysis or visualization
├── my_project/           # Core Python package
│   ├── __init__.py       # Makes 'my_project' a Python package
│   ├── core.py           # Core functionality
│   └── utils.py          # Utility functions
└── tests/                # Unit tests
    └── test_core.py      # Tests for core functionality
```

### Large and Complicated Research Project

Use a `src/`-based, well-tested package layout when the code will be maintained long-term, shared between teams, or released as an installable package. This layout supports testing, CI, clear dependency management, and cleaner imports. This setup are intended to facility large scale code development, and also aim for package distribution. 

Use this when you:
1. Are developing a large, complex, or collaborative codebase. Need a modular structure for scalability and collaboration. A long-term project with multiple contributors.
2. Want to distribute or deploy your package (e.g., as a wheel via `pip install`).
3. Have many experimental scripts, notebooks, or datasets.
4. eg. A complicated machine leanring model, a new commputational tool.

```
my_project/
├── src/                  # Source code directory (recommended for larger projects)
│   └── my_package/       # Main package for your application
│       ├── __init__.py   # Makes 'my_package' a Python package
│       ├── utils.py      # Utility functions
│       ├── config.py     # Configuration settings
│       └── models.py     # Models (e.g., machine learning models)
├── tests/                # Unit and integration tests
│   └── test_main.py      # Main test file
├── data/                 # Data directory for datasets
├── docs/                 # Project documentation (e.g., user guides, API docs)
├── scripts/              # Utility scripts for experiments
├── preprocess_scripts/   # Scripts for preprocessing data
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── notebooks/            # Jupyter notebooks for analysis and visualization
│   ├── data_notebook.ipynb
│   └── result_notebook.ipynb
├── setup_scripts/
│   ├── hpc_setup.sh
│   └── linux_requirement.txt
├── configs/              # configuretions
├── .gitignore            # File to ignore in version control
├── .gitattributes        # File to Managing file encodings and Customizing merge and diff behavior 
├── pyproject.toml        # Modern package configuration (e.g., Poetry, Hatch)
├── requirements.txt      # Project dependencies (can be generated from pyproject.toml)
├── README.md             # Project description and instructions
└── LICENSE               # Project license
```

## Notes:
Regardless of the project size, here is a list of things you should consider:

- Put **core** logic code in Python modules (`my_package` or `src/my_package`).This is important for code reusabiliy
  - Logic for data processing
  - Logic for new machine leanring model
- Use preprocessing scripts (in `preprocessing_scripts`) for:
  - Data cleaning.
  - Number preprocessing scripts to keep the running order obvious.
- Use **scripts** (in `scripts`) for:
  - Training models at scale.
  - Automated data processing.
- Use **notebooks**(in `notebooks`) for:
  - Data exploration and visualization.
  - Running quick experiments on small sample datasets.
  - Demonstrating model predictions and results.
  - Creating human-readable, shareable reports.
- Use **configs** (in `configs`) for:
  - Configuration for each expremients, use git to keep track of them.

- Use **tests** for unit tests (in `test`). 
  - You can use pytest (<https://docs.pytest.org/en/stable/>). 
  - This is highly recommended if you have complicated projects. 
  - If your have a lot of verifiable case, eg. math problem, rule based chem or bio problems 
- Use `setup_scripts` or `requirements.txt` for environment setups:
  - **setup_scripts Directory**:
    - Place any *environment setup scripts* (usually Bash scripts like `.sh` files) in a dedicated folder, e.g., `setup_scripts/`.
    - These scripts can automate environment creation, Python version setup, and the installation of required libraries, which is especially helpful for high-performance computing (HPC) clusters, custom workstations, and environments where pip or conda alone aren’t sufficient (e.g., due to system-level dependencies or restricted access).
    - You can define logic to select between pip, conda, or module load commands, increasing portability across platforms.
  - **requirements.txt**:
    - A plain text file typically in your root directory to describe Python dependencies for pip (or conda, though conda often uses `environment.yml`).
    - Ideal for open-source/public-facing projects, where you want to make installation simple and reproducible via `pip install -r requirements.txt`.
    - Limitation: Pip (and even conda) generated requirements can include exact versions and local system quirks, which may not work identically on all platforms, especially across OSes or different hardware/compilers.
    - Using *one* requirements.txt in the root is recommended for public distribution. For restricted or customized environments (e.g., specific library paths, module loading on HPC), use customized scripts instead.
  - **Multiple requirements.txt or setup_scripts/**:
    - In complex/restricted environments (like HPCs), you may need multiple requirements files or customized setup scripts for different job types (e.g., preprocessing, training, visualization).
    - Place these additional files/scripts in `setup_scripts/`, such as `requirements_preprocessing.txt`, `requirements_training.txt`, or `setup_env_hpc.sh`.
    - This enables selective installation and environment setup, which is often critical when certain dependencies conflict or are only available on some platforms.
- Use **.gitignore** to exclude files you don't want tracked by git, e.g., data files, caches.
  - If data file size is large, keep data or generated files in `data`, and exclude them from the git repository.
- Use **.gitattributes** for
  - Define default line endings, very useful for cross-platform development, e.g., `LF` vs `CRLF`
  - Marking files as binary to avoid unwanted diffs or merges
  - Customizing merge and diff behavior
- Document your code and project structure in (`README.md` and `docs`).


## Working with SLURM

Add following directory if you are work with SLURM system on a HPC, adopt to your usage accordingly.

- **slurm_scripts** is the directory for your slurm scripts, use this to save slurm script template,
- **slurm_working_dir** is the working directory for your slurm scripts, file in this directory should not be tracked by git, thus, add this directory to `.gitignore`. Use a python to create complicated slurm bash file, or copy scipt template from `slurm_scripts` and modify to your task.
- **job** is the working directory for your slurm task output, addd this directory to `.gitignore` as well.

```
├── slurm_scripts/        # this is where your slurm script will go
│   ├── script_1.sh
│   └── script_2.sh
├── slurm_working_dir/    # this is where your slurm script will go
│   ├── script_1.sh
│   └── script_2.sh         
└── jobs                  # this is where your slurm job output will go
    ├── job_out_1.out
    └── job_out_2.out
```

## Working with Apptainer
```
├── apptainer/        # this is where your apptainer files
    ├── script_1.sh
    └── script_2.sh
```