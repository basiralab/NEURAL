# NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation

## Overview

This repository contains the official PyTorch implementation for our paper, **NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation**.

The core of this repository is a two-stage pipeline designed to address the challenges of storing and processing large-scale multimodal medical data. Our framework first uses a fine-tuned vision-language model to guide the structural pruning of chest X-rays, keeping only the most diagnostically relevant regions identified via cross-attention scores. This pruned visual data is then fused with a knowledge graph from the corresponding radiological report to create a unified, lightweight graph representation for efficient downstream diagnostic tasks.

Accepted at the 14th CLIP Workshop, MICCAI Conference 2025

## Framework

![https://raw.githubusercontent.com/basiralab/NEURAL/blob/main/NEURAL.png](https://github.com/basiralab/NEURAL/blob/main/NEURAL.png)

## File Structure

The code is organized into a modular structure for clarity and ease of use:

```
baselines/                  # Baseline Models
src/                        # Source Code Folder
├── main.py                 # Main entry point to run the pipeline
├── stage1_train.py         # Training script for Stage 1 (VLM fine-tuning)
├── stage2_train.py         # Training script for Stage 2 (GNN training)
├── config.py               # Central configuration for paths, and hyperparameters
├── models.py               # Definitions for VisionLanguageModel and MPNN
├── datasets.py             # PyTorch and PyG Dataset classes and data loaders
└── utils.py                # Helper functions for text processing and graph creation
```

## Key Features of the NEURAL Framework

NEURAL is a novel framework designed to reduce the storage and transmission burdens of large multimodal medical datasets without sacrificing diagnostic accuracy. Its key features include:

* **Semantics-Guided Image Pruning**: Instead of conventional compression that is agnostic to content, NEURAL uses a vision-language model to intelligently prune chest X-rays. It leverages the cross-attention scores between an image and its corresponding radiological report to identify and preserve only the most diagnostically critical regions, achieving a data size reduction of **93-97%**.

* **Unified Graph Representation**: The framework transforms multimodal data (images and text) into a single, unified graph structure. The pruned image regions form a visual graph, which is then fused with a knowledge graph derived from the clinical report. This creates a lightweight, extensible data asset that simplifies downstream modeling.

* **Task-Agnostic and Persistent Compression**: The pruning is performed once, guided by the comprehensive information in a radiological report, rather than a specific downstream task. This creates a persistent, compressed data asset that can be stored and reused for various clinical applications, such as disease classification or report generation. 

* **High Diagnostic Performance on Compressed Data**: Despite the extreme compression, NEURAL maintains high performance on diagnostic tasks. For pneumonia detection, it achieved an AUC of **0.88-0.95**, outperforming baseline models that use uncompressed data.

* **Preservation of Full Resolution**: Unlike methods that downsample images and risk losing important details, NEURAL's patch-based approach operates on full-resolution images, ensuring that fine-grained visual features are considered during the pruning process.

---

## Models

The NEURAL framework integrates several models for its two-stage pipeline and is benchmarked against other state-of-the-art models.

### Core Framework Models

* **Swin Transformer (`Swin-base`)**: This serves as the vision encoder. It processes the chest X-ray by dividing it into patches and generating feature representations for each patch. The pre-trained model is automatically downloaded from Hugging Face.

* **Clinical-T5 (`Clinical-T5-Base`)**: A T5-based language model pre-trained on clinical text, used here as the decoder. It is fine-tuned to generate radiological reports from the visual embeddings provided by the Swin encoder. The cross-attention scores from this model are repurposed to guide the image pruning.

* **Message Passing Neural Network (MPNN)**: A graph neural network used for the final downstream diagnostic task. It operates on the unified graph (fused visual and text graphs) to perform classification.

* **BiomedVLP-CXR-BERT**: This model is used to extract medical entities and their relationships from the radiological reports to construct the textual knowledge graph ($G_2$).

## Dependencies

All required Python packages and their specific versions are listed in the `requirements.txt` file. The project was developed using Python 3.9, PyTorch 2.0, and CUDA 12.8.

## Setup

These instructions are written for a Linux (Ubuntu 24.04 LTS) environment.

1.  **Create a Conda Environment** (Recommended)
    ```bash
    conda create -n neural python=3.9
    conda activate neural
    ```
2.  **Install Dependencies**
    Install all the required packages using the `requirements.txt` file. This file includes PyTorch, PyTorch Geometric, and all other necessary libraries.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download SpaCy Model**
    Our pipeline uses the `en_core_sci_sm` model for clinical entity extraction. Download it using the following command:
    ```bash
    python -m spacy download en_core_sci_sm
    ```

## Running the NEURAL Pipeline

### Preparing Data and Models

Before running the code, you must download the required datasets and pre-trained models.

#### Required Datasets and Models

1.  **Datasets**

      * **MIMIC-CXR**

          * **Description**: A large publicly available chest X-ray (CXR) dataset in DICOM format, with free-text radiology reports.
          * **Access**: Requires PhysioNet credentialed access and Data Use Agreement.
          * **Link**: [MIMIC-CXR Database (v2.0.0) on PhysioNet](https://physionet.org/content/mimic-cxr/)

      * **CheXpert**

          * **Description**: A dataset of 224,316 chest radiographs from 65,240 patients, automatically labeled for 14 observations.
          * **Access**: Publicly available via Stanford ML Group; user must agree to terms.
          * **Link**: [CheXpert Dataset (Stanford ML Group)](http://stanfordmlgroup.github.io/competitions/chexpert/)

2.  **Pre-trained Model**

      * **Clinical-T5-Base**
          * **Description**: A T5-Base language model further pretrained on clinical notes from MIMIC-III and MIMIC-IV.
          * **Published**: January 25, 2023, version 1.0.0.
          * **Link**: [Clinical-T5 (v1.0.0) on PhysioNet](https://www.physionet.org/content/clinical-t5/1.0.0/)

3.  **Configuration**
    Open the `config.py` file and update the following path variables to point to your local data and model directories:

      * `LLM_MODEL_NAME`: Path to the downloaded Clinical-T5 model directory.
      * `CSV_PATH`: Path to the CSV file containing labels and study IDs (e.g., `pneumonia_subset_10000.csv`).
      * `IMAGE_BASE_DIR`: Path to the root directory containing the chest X-ray images.
      * `REPORT_BASE_DIR`: Path to the root directory containing the corresponding radiology reports.

### Executing the Two-Stage Pipeline

The pipeline is executed via `main.py` using a `--stage` argument. You must run Stage 1 before Stage 2.

#### Stage 1: Fine-Tuning the Vision-Language Model

This stage fine-tunes the `VisionLanguageModel` for joint report generation and classification. This process generates the cross-attention scores needed for pruning and saves the best-performing model checkpoint.

```bash
python main.py --stage 1
```

This command will train the model based on the hyperparameters in `config.py` (e.g., `STAGE1_EPOCHS`, `STAGE1_LR_LM`) and save the best model to `stage1_best_model.pt`.

#### Stage 2: NEURAL Pruning and GNN Training

This stage uses the checkpoint from Stage 1 to perform attention-guided pruning. It creates the unified multimodal graphs and trains the `MPNN` for the final pneumonia classification task.

```bash
python main.py --stage 2
```

This command loads `stage1_best_model.pt`, processes the data to create pruned graphs (controlled by `STAGE2_PRUNING_PERCENTILE`), trains the GNN classifier, and saves the best model to `stage2_best_mpnn_model.pt`.

## Further Information

  * To learn more about the NEURAL framework, check out our paper:
    [**https://arxiv.org/abs/2508.09715v1**](https://arxiv.org/abs/2508.09715v1)

  * You can find a video presentation of our work here: [**https://www.youtube.com/watch?v=6GZ_Gpk1KZM**](https://www.youtube.com/watch?v=6GZ_Gpk1KZM)

## Please cite our paper if you use this code

```latex
@misc{joshi2025neuralattentionguidedpruningunified,
      title={NEURAL: Attention-Guided Pruning for Unified Multimodal Resource-Constrained Clinical Evaluation}, 
      author={Devvrat Joshi and Islem Rekik},
      year={2025},
      eprint={2508.09715},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.09715}, 
}
```
