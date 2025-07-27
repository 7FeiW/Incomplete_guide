### Simple Research Project Example
Use this when you need to:
1. Create a minimal structure for quick prototyping, a lightweight project for research or demonstration purposes.
2. Avoid distributing your package as a wheel file (e.g., you don't need `pip install mypackage`).

```
my_project/
├── README.md             # Project description and instructions
├── requirements.txt      # Dependencies for the project
├── .gitignore            # Files/directories to ignore in version control
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

### Large and Complicated Research Project Example
Use this when you:
1. Are developing a large, complex, or collaborative codebase. Need a modular structure for scalability and collaboration. A long-term project with multiple contributors.
2. Want to distribute or deploy your package (e.g., as a wheel via `pip install`).
3. Have many experimental scripts, notebooks, or datasets.

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
├── .gitignore            # Files/directories to ignore in version control
├── pyproject.toml        # Modern package configuration (e.g., Poetry, Hatch)
├── requirements.txt      # Project dependencies (can be generated from pyproject.toml)
├── README.md             # Project description and instructions
└── LICENSE               # Project license
```

### Notes:
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
- If data file size is large, keep data or generated files in `data`, and exclude them from the git repository (add them to `.gitignore`).
- Document your code and project structure in (`README.md` and `docs`).
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