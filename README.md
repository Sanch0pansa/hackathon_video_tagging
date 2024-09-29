# Hack-ai oMLet team's repository

CLI Usage Instructions for Video Feature Extraction, Inference, and Training
============================================================================

This CLI tool is designed to help you extract features from videos, perform inference using a pre-trained model, and train a model using extracted features. Below you'll find instructions on how to use each command with example usage.

Table of Contents
-----------------

*   [Installation](#installation)
*   [Commands Overview](#commands-overview)
*   [Extracting Features from a Single Video](#extracting-features-from-a-single-video)
*   [Running Inference on a Single Feature File](#running-inference-on-a-single-feature-file)
*   [Full Inference Workflow](#full-inference-workflow)
*   [Extracting Features from a Directory](#extracting-features-from-a-directory)
*   [Running Inference on a Directory of Features](#running-inference-on-a-directory-of-features)
*   [Training a Model](#training-a-model)

Installation
------------

1.  Clone the repository or copy the script that contains the CLI.
2.  Install the required dependencies using `pip`:
    
        pip install -r requirements.txt
    
3.  Run the CLI using:
    
        python cli.py [command] [options]
    

Commands Overview
-----------------

`extract-features`

Extract features from a single video file.

`run-inference`

Run inference using a single feature file.

`full-inference`

Extract features and run inference in one step.

`extract-features-dir`

Extract features from all video files in a directory.

`run-inference-dir`

Run inference on all feature files in a directory.

`train`

Train a model using a directory of feature files.

Extracting Features from a Single Video
---------------------------------------

To extract features from a single video, use the `extract-features` command.

### Syntax

    python cli.py extract-features VIDEO_PATH TITLE DESCRIPTION [OPTIONS]

### Arguments:

*   **VIDEO\_PATH**: Path to the video file (e.g., `videos/sample.mp4`).
*   **TITLE**: The title of the video.
*   **DESCRIPTION**: A description of the video.

### Options:

*   `--output-tensor-path`, `-o`: Optional path to save the extracted features (tensor). Defaults to `output_tensor.pt`.

### Example:

    python cli.py extract-features videos/sample.mp4 "Sample Title" "This is a description of the sample video" --output-tensor-path output/sample_features.pt

Running Inference on a Single Feature File
------------------------------------------

To run inference on a feature file and predict the tags, use the `run-inference` command.

### Syntax

    python cli.py run-inference FEATURES_PATH [OPTIONS]

### Arguments:

*   **FEATURES\_PATH**: Path to the `.pt` file containing extracted features.

### Options:

*   `--save-to-file`, `-s`: Optional path to save the predicted tags. If not provided, tags are printed to the console.

### Example:

    python cli.py run-inference output/sample_features.pt --save-to-file output/tags.txt

Full Inference Workflow
-----------------------

The `full-inference` command combines feature extraction and inference into a single step.

### Syntax

    python cli.py full-inference VIDEO_PATH TITLE DESCRIPTION [OPTIONS]

### Arguments:

*   **VIDEO\_PATH**: Path to the video file.
*   **TITLE**: The title of the video.
*   **DESCRIPTION**: A description of the video.

### Options:

*   `--output-tensor-path`, `-o`: Optional path to save the extracted features (tensor). Defaults to `output_tensor.pt`.
*   `--save-to-file`, `-s`: Optional path to save the predicted tags.

### Example:

    python cli.py full-inference videos/sample.mp4 "Sample Title" "This is a description" --output-tensor-path output/sample_features.pt --save-to-file output/tags.txt

Extracting Features from a Directory
------------------------------------

To extract features from all video files in a directory, use the `extract-features-dir` command.

### Syntax

    python cli.py extract-features-dir DIRECTORY [OPTIONS]

### Arguments:

*   **DIRECTORY**: Path to the directory containing video files.

### Options:

*   `--output-dir`, `-o`: Directory to save the extracted tensors. Defaults to the current directory.

### Example:

    python cli.py extract-features-dir videos/ --output-dir output/features/

Running Inference on a Directory of Features
--------------------------------------------

To run inference on all feature files in a directory, use the `run-inference-dir` command.

### Syntax

    python cli.py run-inference-dir FEATURES_DIR [OPTIONS]

### Arguments:

*   **FEATURES\_DIR**: Path to the directory containing feature files.

### Options:

*   `--save-dir`, `-s`: Directory to save the predicted tags.

### Example:

    python cli.py run-inference-dir output/features/ --save-dir

Training a Model
----------------

To train a model using a directory of feature files and a corresponding table file with labels, use the `train` command.

### Syntax

    python cli.py train FEATURES_DIR TABLE_PATH MODEL_SAVE_PATH [OPTIONS]

### Arguments:

*   **FEATURES\_DIR**: Path to the directory containing feature files (e.g., `features/`).
*   **TABLE\_PATH**: Path to the CSV file containing labels and metadata for training (e.g., `labels.csv`).
*   **MODEL\_SAVE\_PATH**: Path to save the trained model (e.g., `models/model.pth`).

### Options:

*   `--batch-size`, `-b`: Batch size for training. Defaults to **16**.
*   `--learning-rate`, `-lr`: Learning rate for the optimizer. Defaults to **0.001**.
*   `--epochs`, `-e`: Number of training epochs. Defaults to **20**.
*   `--wandb-run-name`, `-w`: Name of the WandB run (for tracking experiments). Defaults to **"mlp-classifier-1"**.
*   `--hidden-size`, `-hs`: Size of the hidden layer in the neural network. Defaults to **256**.

### Example:

    python cli.py train "./model/embeddings_1536/" "./model/train_dataset_tag_video/baseline/train_data_categories.csv" "./model/checkpoints/final_model_1.ckpt" -r "mlp-classifier-256"