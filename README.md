# NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation

This repository contains the official PyTorch implementation for our paper, **NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation**.

The core of this repository is a two-stage pipeline designed to address the challenges of storing and processing large-scale multimodal medical data. Our framework first uses a fine-tuned vision-language model to guide the structural pruning of chest X-rays, keeping only the most diagnostically relevant regions identified via cross-attention scores. This pruned visual data is then fused with a knowledge graph from the corresponding radiological report to create a unified, lightweight graph representation for efficient downstream diagnostic tasks.

The code is organized into a modular structure for clarity and ease of use:

```
src/
├── main.py                 # Main entry point to run the pipeline
├── stage1_train.py         # Training script for Stage 1 (VLM fine-tuning)
├── stage2_train.py         # Training script for Stage 2 (GNN training)
├── config.py               # Central configuration for paths, and hyperparameters
├── models.py               # Definitions for VisionLanguageModel and MPNN
├── datasets.py             # PyTorch and PyG Dataset classes and data loaders
└── utils.py                # Helper functions for text processing and graph creation
```

# Dependencies

The project was developed using Python 3.9 and PyTorch 2.0. We recommend setting up a dedicated virtual environment. The primary dependencies are listed below:

```
# PyTorch & TorchGeometric (for CUDA 12.8)
torch==2.0.0+cu128
torchvision==0.15.1+cu128
torch-scatter==2.1.1+pt20cu128
torch-sparse==0.6.17+pt20cu128
torch-geometric==2.3.0

# Hugging Face & NLP
transformers==4.30.2
spacy==3.5.0
en_core_sci_sm

# Other Core Libraries
numpy==1.24.1
pandas==1.5.3
scikit-learn==1.2.2
networkx==3.1
Pillow==9.5.0
tqdm==4.65.0
```

**Note:** We used the **CUDA 12.8** toolkit with an NVIDIA A30 GPU. You should use the CUDA toolkit version that corresponds to your NVIDIA GPU driver. You can check compatibility [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver).

# Setup

These instructions are written for a Linux (Ubuntu 24.04 LTS) environment.

1.  **Create a Conda Environment** (Recommended)
    ```bash
    conda create -n neural python=3.9
    conda activate neural
    ```
2.  **Install PyTorch and its ecosystem**
    The following commands install PyTorch, torchvision, and the required PyTorch Geometric libraries for CUDA 12.8.
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu128.html
    ```
3.  **Install Remaining Python Packages**
    ```bash
    pip install transformers==4.30.2
    pip install spacy==3.5.0
    pip install numpy==1.24.1
    pip install pandas==1.5.3
    pip install scikit-learn==1.2.2
    pip install networkx==3.1
    pip install Pillow==9.5.0
    pip install tqdm==4.65.0
    ```
4.  **Download SpaCy Model**
    Our pipeline uses the `en_core_sci_sm` model for clinical entity extraction. Download it using the following command:
    ```bash
    python -m spacy download en_core_sci_sm
    ```

# Running the NEURAL Pipeline

## Preparing Data and Models

Before running the code, you must download the required datasets and pre-trained models.

1.  **Datasets:** The framework is designed for the **MIMIC-CXR**  and **CheXpert**  datasets. Download them and organize the files as you see fit.
2.  **Pre-trained Models:** You will need the **Clinical-T5-Base** language model. Download the model files from a trusted source like Hugging Face.
3.  **Configuration:** Open the `config.py` file and update the following path variables to point to your local data and model directories:
      * `LLM_MODEL_NAME`: Path to the downloaded Clinical-T5 model directory.
      * `CSV_PATH`: Path to the CSV file containing labels and study IDs (e.g., `pneumonia_subset_1500.csv`).
      * `IMAGE_BASE_DIR`: Path to the root directory containing the chest X-ray images.
      * `REPORT_BASE_DIR`: Path to the root directory containing the corresponding radiology reports.

## Executing the Two-Stage Pipeline

The pipeline is executed via `main.py` using a `--stage` argument. You must run Stage 1 before Stage 2.

**Stage 1: Fine-Tuning the Vision-Language Model**

This stage fine-tunes the `VisionLanguageModel` for joint report generation and classification. This process generates the cross-attention scores needed for pruning and saves the best-performing model checkpoint.

```bash
python main.py --stage 1
```

This command will train the model based on the hyperparameters in `config.py` (e.g., `STAGE1_EPOCHS`, `STAGE1_LR_LM`) and save the best model to `stage1_best_model.pt`.

**Stage 2: NEURAL Pruning and GNN Training**

This stage uses the checkpoint from Stage 1 to perform attention-guided pruning. It creates the unified multimodal graphs and trains the `MPNN` for the final pneumonia classification task.

```bash
python main.py --stage 2
```

This command loads `stage1_best_model.pt`, processes the data to create pruned graphs (controlled by `STAGE2_PRUNING_PERCENTILE`), trains the GNN classifier, and saves the best model to `stage2_best_mpnn_model.pt`.

## Further Information

  * To learn more about the NEURAL framework, check out our paper:
    [**[https://github.com/basiralab/NEURAL](https://github.com/basiralab/NEURAL)**] 

  * You can find a video presentation of our work here: [Link to YouTube video]

## Please cite our paper if you use this code

```latex
@inproceedings{joshi2025neural,
  title={NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation},
  author={Joshi, Devvrat and Rekik, Islem},
  booktitle={},
  year={2025},
  organization={Springer}
}
```