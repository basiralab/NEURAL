import os
import logging
import warnings
import torch

# --- Universal Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model and Data Path Configuration ---
LLM_MODEL_NAME = '/vol/bitbucket/dj623/SciER/cross-doc-relation-extraction/physionet.org/files/clinical-t5/1.0.0/Clinical-T5-Base'
VISION_MODEL_NAME = 'microsoft/swin-base-patch4-window12-384-in22k'
CSV_PATH = 'pneumonia_subset_1500.csv'
IMAGE_BASE_DIR = 'pneumonia_dataset_images'
REPORT_BASE_DIR = 'pneumonia_dataset/reports'
SPACY_MODEL_NAME = "en_core_sci_sm"

# --- Stage 1 (Fine-Tuning) Configuration ---
STAGE1_CHECKPOINT_PATH = 'stage1_best_model.pt'
STAGE1_NUM_SAMPLES = 1500
STAGE1_EPOCHS = 5
STAGE1_BATCH_SIZE = 4
STAGE1_ACCUMULATION_STEPS = 16
STAGE1_LR_LM = 2e-5
STAGE1_LR_VISION = 2e-6
STAGE1_JPEG_QUALITY = 10

# --- Stage 2 (Downstream) Configuration ---
STAGE2_CHECKPOINT_PATH = 'stage2_best_mpnn_model.pt'
STAGE2_NUM_SAMPLES = 500
STAGE2_PRUNING_PERCENTILE = 2.3
STAGE2_EPOCHS_MPNN = 15
STAGE2_BATCH_SIZE_MPNN = 16
STAGE2_LR_MPNN = 1e-4

# --- Shared Configuration ---
IMAGE_SIZE = 384
MAX_TEXT_LENGTH = 256
DROPOUT = 0.2