### Simple Research Project
Use this when you need to:
1. Need a minimal structure for quick prototyping, a lightweight project for research or demonstration purposes.
2. Does not need to distributing your package as a wheel file (e.g., you dont need `pip install mypackage`).

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
    ├── __init__.py       # Makes 'tests' a Python package
    └── test_core.py      # Tests for core functionality
```

### Large and Complicated Research Project
Use this when you:
1. Are developing a large, complex, or collaborative codebase. Need modular structure for scalability and collaboration. A long-term project with multiple contributors.
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
├── .gitignore            # Files/directories to ignore in version control
├── pyproject.toml        # Modern package configuration (e.g., Poetry, Hatch)
├── requirements.txt      # Project dependencies (can be generated from pyproject.toml)
├── README.md             # Project description and instructions
└── LICENSE               # Project license
```


### Notes:
Regardless size of project here is list things you should consider.

- Put core logic, data loading, preprocessing, and model-training code in Python modules (`my_package` or `src/my_package`).
- Use preprocessing scripts (in `preprocessing_scripts`) for:
    - Data cleaning up
    - number preprocessing script to keep running order obvious
- Use scripts (in `scripts`) for:
    - Training models at scale
    - Automated data processing
- Use notebooks (`notebooks`) for 
    - Data exploration and visualization
    - Running quick experiments on small sample datasets
    - Demonstrating model predictions and results
    - Creating human-readable, shareable reports
- If datafile size is large, keep data or generated files in `data`, and out of the git repository (does not track it with git and add to `.gitignore`).
- Document your code and project structure in (`README.md` and `docs`).


### This is quick check list

| components                                | Develop New ML method | Run ML method | Develop Full Package |
| ----------------------------------------- | --------------------- | ------------- | -------------------- |
| `setup_scripts` or `requirement.txt` | ☑                    | ☑            | ☑                   |
| `preprocessing_scripts`                 | ☑                    | ☑            |                      |
| `scripts`                              | ☑                    | ☑            |                      |
| `postprocessing`                        |                       |               |                      |
| `data`                                  | ☑                    | ☑            |                      |
| `notebooks`                             |                       |               |                      |
| `.gitignore`                            | ☑                    | ☑            | ☑                   |
| `.gitattributes`                        | ☑                    | ☑            | ☑                   |
| `src`                                  | ☑                    |               | ☑                   |
| `setup.py`                              | ☑                    |               | ☑                   |
| `tests`                                 |                       |               | ☑                   |
| `dist`                                  |                       |               |                      |