# Project Structure

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

#### For ease of life, we will organize this project into the following directories

`setup_scripts` or `requirement.txt`: `setup_scripts` is the location for environment setups. Use bash scripts to define Python libraries you need. `requirement.txt` is environment setup for pip or conda. Be careful with pip or conda generated `requirement.txt`, as this may break in a cross-platform setup. Use one `requirement.txt` in the main directory if you are developing a public-facing package, use multiple ones in the `setup_scripts` if you need to run this code on restricted Python environments such as some HPC.

`preporcessing_scripts`:This directory is location for  preprocessing scripts goes. Number each of your task, and add documentation eg. task 1 is doing `<something>` and its expecting `<input>` will `<output>`

`scripts`: This is the place where training and evaluation scripts go.

`postprocessing`: This directory is location for postprocessing scripts goes. Number each of your task, and add documentation eg. task 1 is doing `<something>` and its expecting `<input>` will `<output>`

`data`: This directory is the location for data.

`notebooks`: This directory is the location for Jupyter notebooks. Use notebooks for data analysis and plotting. Do not use this for model training. It is hard to compare.

`dist`: Where dist files will be, only use this if you want to release this as a pip install package.

#### Useful git files

`.gitignore`: Use this file to exclude files you don't want tracked by git, e.g., data files, caches.

`.gitattributes`: Use this to define default line endings, very useful for cross-platform development, e.g., `LF` vs `CRLF`.

#### For Python package development

**This is needed if you want to develop a Python package or your project is very complicated, e.g., complicated importing**

`src`: This directory is the location for Python packages if you don't want to place them directly in the project root. I prefer this way, as placing them directly in the project root gives me a headache. If you wish to use other approaches, check out my other repo at (<https://bitbucket.org/wishartlab/cfm-toolkit/src/master/?search_id=3487d7be-19a5-481c-bc3c-bd50849f69ff>).

`setup.py`: A file that you include in the root of your project, which contains information about your package and its dependencies. The file is used by pip, the package installer for Python, to install your package and its dependencies.

`tests`: This directory is the location for unit tests. You can use pytest (<https://docs.pytest.org/en/stable/>). This is highly recommended if you have complicated projects. For more details...