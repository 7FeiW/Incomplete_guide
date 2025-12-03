# Logging 

Logs help identify issues by providing a record of events, errors, and system behaviors. Logs can also be used as a record of each experiments. Combined with configuration, this provided a way to effectively record and keep track of your reserach results.

It’s a good idea to explicitly mention and differentiate between “results logs” and “running logs” in your research logging structure. Results logs provide concise summaries of experiment outcomes, making it easy to compare or aggregate results across experiments. In contrast, running logs capture detailed events and diagnostics, which are invaluable for investigating issues or understanding the execution process in depth. Since running logs tend to be large and verbose, separating them from results logs not only streamlines storage and analysis but also makes it easier to efficiently query and review experiment outcomes.

## Results Logs

- **Purpose:** Capture final outputs, evaluation metrics, summaries, or any high-level results produced by an experiment.
- **Typical Content:**
    - Final accuracy, loss, or custom metrics.
    - Hyperparameters and configuration used for the run.
    - Dataset version and key identifiers.
    - Time/date stamps for experiment completion.
- **Format Recommendation:** Use a structured, parsable format like JSON,CSV,PKL,Dataframes for easy aggregation and later analysis. Make sure this format is machine readable for your analysis script.
- sample reuslts log:
    ```json
    {
    "experiment_id": "exp042",
    "timestamp": "2025-07-30T14:00:00Z",
    "status": "COMPLETED",
    "model": "ResNet34",
    "dataset": "CIFAR-10_v3.2",
    "config": {
        "learning_rate": 0.001,
        "batch_size": 128,
        "epochs": 20,
        "seed": 12345
    },
    "metrics": {
        "train_accuracy": 0.991,
        "valid_accuracy": 0.932,
        "test_accuracy": 0.927,
        "final_loss": 0.183
    },
    "notes": "Baseline experiment for reproducibility."
    }

    ```
## Running Logs

- **Purpose:** Record events, errors, warnings, and detailed steps that occur during experiment execution. Useful for troubleshooting, debugging, or understanding experiment internals.
- **Typical Content:**
    - Training/validation progress (loss, accuracy per epoch/batch).
    - Warnings about data, resource limits, or anomalies.
    - Error traces if the process fails.
    - Step-by-step debug details if enabled.
- **Format Recommendation:** JSON lines or a structured plain text format works well. Allow filtering by log level (INFO, WARNING, ERROR, DEBUG).
- A sample running log
    ```text
    2025-07-30 13:00:00 [INFO] ExperimentStarted: experiment_id=exp042
    2025-07-30 13:00:05 [INFO] ConfigLoaded: learning_rate=0.001, batch_size=128, epochs=20
    2025-07-30 13:05:11 [INFO] EpochEnd: epoch=1, train_loss=1.12, train_accuracy=63.0%
    2025-07-30 13:10:56 [WARNING] DataWarning: Detected 25 missing labels, using placeholder values
    2025-07-30 13:21:40 [DEBUG] BatchProcessed: epoch=5, batch=42, loss=0.94, accuracy=77.0%
    2025-07-30 13:44:59 [ERROR] CheckpointError: Failed to save checkpoint at epoch 10 due to IO error
    2025-07-30 14:00:00 [INFO] ExperimentCompleted: experiment_id=exp042, status=COMPLETED
    ```
## Loging Levels
**Logs should be in different tiers:**

1. **Info/Experiment Tier**
    - Purpose: Record high-level events, such as the start and end of experiments, key parameter values, dataset versions, and final results (e.g., accuracy, loss).
    - Example:
        - `"Experiment started at 2025-07-30 13:00"`
        - `"Config: learning_rate=0.001, batch_size=128, dataset_version='v3.2'"`
        - `"Final test accuracy: 91.5%"`
2. **Warning Tier**
    - Purpose: Note unusual or suboptimal conditions that do not stop execution but might affect results (e.g., missing data, unexpected value distributions, minor hardware slowdowns).
    - Example:
        - `"Warning: Detected 250 missing values in feature 'age'"`
        - `"GPU memory almost full, switching to CPU for final epoch"`
3. **Error Tier**
    - Purpose: Record serious problems that result in experiment failures, crashes, or loss of results. This is essential for post-mortem analysis and correcting issues.
    - Example:
        - `"Error: Unable to load model checkpoint at epoch 10"`
        - `"Exception: OutOfMemoryError during training phase"`
4. **Debug/Detail Tier**
    - Purpose: Capture granular, step-by-step outputs for diagnosis—ideal for making sense of complex behaviors but very verbose. Typically enabled only during active debugging.
    - Example:
        - `"Epoch 5: Batch 34 - loss=1.128, accuracy=71.2%"`
        - `"Data augmentation: rotating image 42 by 15 degrees"`