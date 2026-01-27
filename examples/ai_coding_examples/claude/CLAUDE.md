## Project Overview

This is a machine learning research project for model training and inference pipelines.
- **Language**: Python 3.10+
- **Core libraries**: PyTorch, PyTorch Lightning, DGL, NumPy, scikit-learn
- **Data processing**: Pandas, Polars
- **Testing**: pytest with fixtures
- **Execution**: SLURM on HPC clusters or local GPU training
- **Key patterns**: Modular src/ package, numbered preprocessing scripts, experiment config 

### Environment Setup

**VS Code Configuration:**

This workspace is configured to use the `FRAGNNET-GPU` conda environment.

1. **Automatic Detection**: VS Code will automatically detect available conda environments
2. **Manual Selection**: Use `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → "Python: Select Interpreter" → Choose `FRAGNNET-GPU`
3. **Settings File**: The `.vscode/settings.json` file contains Python environment configuration

**Ensure the following before starting development:**
- Activate the `FRAGNNET-GPU` environment in VS Code
- Verify Python interpreter path: Use command palette → "Python: Show Python Environment Details"
- The environment includes all dependencies for GPU training with PyTorch, PyTorch Lightning, and DGL

**For HPC/SLURM workflows**, use the environment scripts in the `env/` directory for cluster-specific setups.

### Install
1. `python -m venv venv`
2. `source venv/bin/activate`  (Unix) or `venv\Scripts\activate.ps1` (Windows)
3. `pip install -e .` (editable install with dependencies from pyproject.toml)
4. `pip install -e ".[dev]"` (for testing/development with development dependencies)


### Data Processing
- `python preprocess_scripts/01_prepare_df.py --config config/preprocess.yaml`
- `python preprocess_scripts/02_prepare_proc.py --config config/dataset.yaml`
- `python preprocess_scripts/03_prepare_dag_feats.py --config config/dag_feats.yaml`
- `python preprocess_scripts/04_prepare_split.py --config config/split.yaml`

### Training
- **Local GPU**: `python scripts/run_pl_model_fit.py --config config/model_config.yaml`
- **SLURM cluster**: `sbatch slurm_scripts/fraggnn_d3_ma_mi_nist20_ft.sh` (or other experiment-specific scripts)

### Inference
- `python scripts/run_inference.py --config config/inference.yaml --checkpoint saved_ckpts/model.ckpt`

## Project Structure
src/
  └── fragnnet/               # Main Python package
      ├── __init__.py
      ├── runner.py           # Training/inference runner
      ├── model/              # Model definitions
      │   ├── __init__.py
      │   ├── base_model.py           # Base model classes
      │   ├── fragnnet_model.py       # FraGNNet architecture
      │   ├── gnn_model.py            # GNN components
      │   ├── spectrum_encoder.py     # Spectrum encoding
      │   ├── mol_encoder.py          # Molecule encoding
      │   ├── form_embedder.py        # Formula embedder
      │   ├── precursor_model.py      # Precursor models
      │   ├── siamese_gnn_model.py    # Siamese GNN
      │   ├── neims_model.py          # NEIMS models
      │   ├── nn_blocks.py            # Neural network blocks
      │   └── loss.py                 # Loss functions
      ├── dataset/            # Dataset classes
      │   ├── __init__.py
      │   ├── base_dataset.py         # Base dataset class
      │   ├── spec_mol_dataset.py     # Spectrum-molecule dataset
      │   ├── spec_mol_frag_dataset.py # Spectrum-molecule-fragment dataset
      │   └── mces_dataset.py         # MCES dataset
      ├── frag/               # Fragmentation logic
      │   ├── __init__.py
      │   ├── compute_frags.pyx        # Cython fragmentation implementation
      │   └── *.so                     # Compiled fragmentation libraries
      ├── graff/              # GRAFF module
      │   ├── __init__.py
      │   ├── data_utils.py
      │   ├── dataset.py
      │   ├── model.py
      │   └── nn_utils.py
      ├── iceberg/            # Iceberg module
      │   ├── __init__.py
      │   ├── common/
      │   │   ├── chem_utils.py
      │   │   ├── fingerprint.py
      │   │   └── misc_utils.py
      │   ├── nn_utils/
      │   │   ├── attention.py
      │   │   ├── form_embedder.py
      │   │   ├── set_transformer.py
      │   │   └── transformer_layers.py
      │   ├── dataset.py
      │   ├── fragmentation.py
      │   ├── model.py
      │   └── nn_utils.py
      ├── massformer/         # MassFormer module
      │   ├── __init__.py
      │   ├── algos.pyx                # Cython algorithms
      │   ├── data_utils.py
      │   ├── dataset.py
      │   ├── model.py
      │   └── nn_utils.py
      ├── pl_model/           # PyTorch Lightning models
      │   ├── __init__.py
      │   ├── binned_pl.py             # Binned spectrum model
      │   ├── fragnnet_pl.py           # FraGNNet Lightning wrapper
      │   ├── gnn_pl.py                # GNN Lightning wrapper
      │   ├── mces_pl.py               # MCES Lightning model
      │   ├── neims_pl.py              # NEIMS Lightning model
      │   ├── precursor_pl.py          # Precursor Lightning model
      │   ├── spectrum_mol_clip_pl.py  # Spectrum-molecule CLIP model
      │   └── spectrum_pl.py           # Spectrum model
      └── utils/              # Shared utilities
          ├── __init__.py
          ├── data_utils.py           # Data loading utilities
          ├── feat_utils.py           # Feature utilities
          ├── formula_utils.py        # Formula processing
          ├── frag_utils.py           # Fragmentation utilities
          ├── spec_utils.py           # Spectrum utilities
          ├── ms2c_utils.py           # MS2C utilities
          ├── nn_utils.py             # Neural network utilities
          ├── pl_utils.py             # PyTorch Lightning utilities
          ├── proc_utils.py           # Processing utilities
          ├── plot_utils.py           # Plotting utilities
          ├── profile_utils.py        # Profiling utilities
          ├── script_utils.py         # Script utilities
          ├── setup_utils.py          # Setup utilities
          ├── dgl_compat_utils.py     # DGL compatibility
          └── misc_utils.py           # Miscellaneous utilities

tests/
  ├── test_mces_weight_loading.py     # MCES weight loading tests
  └── test_dmpnn_edge_format.py       # DMPNN edge format tests

preproc_scripts/
  ├── 01_prepare_df.py              # Prepare dataframe
  ├── 02_prepare_proc.py            # Data processing
  ├── 03_prepare_dag_feats.py       # DAG feature preparation
  ├── 04_prepare_split.py           # Dataset splitting
  ├── 05_prepare_magma_feats.py     # MAGMA feature preparation
  ├── 06_predict_magma_dags.py      # MAGMA DAG prediction
  ├── 07_prepare_dag_hdf5.py        # HDF5 conversion
  ├── casmi_ms2c/                   # CASMI MS2C processing
  ├── dataset_migration/            # Dataset migration scripts
  ├── inference/                    # Inference preprocessing
  ├── mces_pretraining/             # MCES pretraining data
  ├── mix_training_exp/             # Mixed training experiments
  ├── no_dag_exp/                   # Experiments without DAGs
  └── pubchem_ms2c/                 # PubChem MS2C processing

scripts/
  ├── run_pl_model_fit.py           # PyTorch Lightning model training
  ├── run_inference.py              # Inference script
  ├── run_compute_frags.py          # Fragment computation
  ├── run_wandb_sweep.py            # W&B hyperparameter sweep
  ├── setup_job_cc.py               # Compute Canada job setup
  ├── bash/                         # Bash utility scripts
  │   ├── cc_wandb_download.sh
  │   └── cc_wandb_sync.sh
  ├── eval/                         # Evaluation scripts
  │   ├── run_save_entropy.py
  │   ├── run_save_inference.py
  │   ├── run_save_inference_ablations.py
  │   └── run_save_inference_eval.py
  ├── debug/                        # Debugging utilities
  ├── manual_slurm/                 # Manual SLURM job submission
  ├── misc/                         # Miscellaneous scripts
  ├── plotting/                     # Plotting and visualization
  └── stats_scripts/                # Statistical analysis scripts

notebooks/
  ├── ablation_exp.ipynb
  ├── c2ms_sim_exp.ipynb
  ├── formula_ann_exp.ipynb
  ├── frag_ann_exp.ipynb
  ├── frag_stats.ipynb
  ├── ms2c_retrieval_exp.ipynb
  ├── param_counts.ipynb
  ├── condtion_notebooks/           # Conditional analysis notebooks
  ├── data_notebooks/               # Data exploration and analysis
  ├── debug_notebooks/              # Debugging and troubleshooting
  ├── msg_notebooks/                # Message/spectrum related notebooks
  ├── smarts_notebooks/             # SMARTS pattern notebooks
  └── test_notebooks/               # Testing and validation notebooks

config/
  ├── template.yml                 # Config template
  ├── ablations/
  ├── charge_migration_fragmentation/
  ├── compare_interstage/
  ├── compare_mol_pooling/
  ├── debug/
  ├── entropy/
  ├── exp_fraggnn_ma_mi/
  ├── fragnnet_finetuning/
  ├── loss_funcs/
  ├── massspecgym_merged/
  ├── mces_pretraining/
  ├── mix_percision/
  ├── nist20_merged_inchikey/
  ├── nist20_merged_scaffold/
  ├── nist23_eims_inchikey/
  ├── pairwise_loss/
  ├── spectraverse/
  └── sweeps/

slurm_scripts/
  ├── deepmet_predicted.sh          # DeepMet experiment
  ├── fraggnn_d3_ma_mi_nist20_ft.sh # FraGNNet fine-tuning on NIST20
  ├── nist20_d3_mixed.sh            # NIST20 mixed precision training
  ├── nist23_d3_mixed.sh            # NIST23 mixed precision training
  ├── nist23_mces_finetuning_m15.sh # NIST23 MCES fine-tuning
  ├── test.sh                       # Testing script
  └── vulcan_test.sh                # Vulcan cluster testing

.github/
  └── copilot-instructions.md

