
### Simple Research Project
Use this when you need to 
1. This is for research/demo only
2. Not for distubite your package as a wheel file eg. pip install mypackage

```
my_project/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── other_scripts.py
├── noteooke.ipynb
├── my_project/
│   ├── __init__.py
│   ├── core.py
│   └── utils.py
└── tests/
    ├── __init__.py
    └── test_core.py
```



### Large and Complicated Research Project
Use this when you:
1. Are developing a large, complex, or collaborative codebase.
2. Want to distribute or deploy your package (e.g., as a wheel via pip install).
3. Have many experimental scripts, notebooks, or datasets.
```
my_project/
├── src/                  # Source code directory (recommended for larger projects)
│   └── my_package/       # Main package for your application
│       ├── __init__.py   # Makes 'my_package' a Python package
│       ├── utils.py      # Utility functions
│       ├── config.py     # Configuration settings
│       └── models.py     # Models (e.g., models)
├── tests/                # Unit and integration tests
│   └── test_main.py
├── data/                 # Data
├── docs/                 # Project documentation (e.g., documentation)
├── scripts/              # Utility scripts for conduct experiments
├── preprocess_scripts/   # Utility scripts for preproces script, naming them with numbers to keep running order
│   ├── 01_extract_data.py
│   └── 02_create_dataset.py
├── notebooks/            # notebooks
│   ├── data_notebook.ipynb
│   └── result_notebook.pynb
├── .gitignore            # Files/directories to ignore in version control
├── pyproject.toml        # Modern package configuration (e.g., Poetry, Hatch)
├── requirements.txt      # Project dependencies (can be generated from pyproject.toml)
├── README.md             # Project description and instructions
└── LICENSE               # Project license
```

`src`: Encapsulates the core source code, preventing name collisions with other project-level files.
`src/my_package`: This subdirectory within src/ represents your primary Python package, containing modules and sub-packages.
`tests `: Contains all test files.

`setup_scripts` or `requirement.txt`: `setup_scripts` is the location for environment setups. Use bash scripts to define Python libraries you need. `requirement.txt` is environment setup for pip or conda. Be careful with pip or conda generated `requirement.txt`, as this may break in a cross-platform setup. Use one `requirement.txt` in the main directory if you are developing a public-facing package, use multiple ones in the `setup_scripts` if you need to run this code on restricted Python environments such as some HPC.

`preporcessing_scripts`:This directory is location for  preprocessing scripts goes. Number each of your task, and add documentation eg. task 1 is doing `<something>` and its expecting `<input>` will `<output>`

`scripts`: This is the place where training and evaluation scripts go.

`data`: This directory is the location for data.

`notebooks`: This directory is the location for Jupyter notebooks. Use notebooks for data analysis and plotting. Do not use this for model training. It is hard to compare.

`dist`: Where dist files will be, only use this if you want to release this as a pip install package.

`.gitignore`: Use this file to exclude files you don't want tracked by git, e.g., data files, caches.

`.gitattributes`: Use this to define default line endings, very useful for cross-platform development, e.g., `LF` vs `CRLF`.