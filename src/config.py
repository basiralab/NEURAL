"""
Configuration file for the NEURAL Framework.
"""

import os
import logging
import warnings
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Model and Data Path Configuration ---
# Change these paths as needed for your environment
LLM_MODEL_NAME = ''                                                     # Pre-trained language model (e.g., T5, BERT)       
VISION_MODEL_NAME = 'microsoft/swin-base-patch4-window12-384-in22k'     # Pre-trained vision model
CSV_PATH = 'pneumonia_subset_10000.csv'                                 # Path to the CSV file containing metadata
IMAGE_BASE_DIR = 'pneumonia_dataset_images'                             # Base directory for images              
REPORT_BASE_DIR = 'pneumonia_dataset/reports'                           # Base directory for reports
SPACY_MODEL_NAME = "en_core_sci_sm"                                     # SpaCy model for text processing                


# --- Stage 1 (Fine-Tuning) Configuration ---
STAGE1_CHECKPOINT_PATH = 'stage1_best_model.pt'                         # Path to save the best model after fine-tuning
STAGE1_NUM_SAMPLES = 10000                                              # Number of samples to use for fine-tuning             
STAGE1_EPOCHS = 5                                                       # Number of epochs for fine-tuning                 
STAGE1_BATCH_SIZE = 4                                                   # Batch size for fine-tuning               
STAGE1_ACCUMULATION_STEPS = 16                                          # Gradient accumulation steps for larger effective batch size
STAGE1_LR_LM = 2e-5                                                     # Learning rate for the language model               
STAGE1_LR_VISION = 2e-6                                                 # Learning rate for the vision model
STAGE1_JPEG_QUALITY = 10                                                # JPEG quality for image compression during Stage 1                    

# --- Stage 2 (Downstream) Configuration ---
STAGE2_CHECKPOINT_PATH = 'stage2_best_mpnn_model.pt'                    # Path to save the best MPNN model after training
STAGE2_NUM_SAMPLES = 10000                                              # Number of samples to use for Stage 2 training              
STAGE2_PRUNING_PERCENTILE = 2.3                                         # Adjust depending on the requirement       
STAGE2_EPOCHS_MPNN = 15                                                 # Number of epochs for MPNN training             
STAGE2_BATCH_SIZE_MPNN = 16                                             # Batch size for MPNN training              
STAGE2_LR_MPNN = 1e-4                                                   # Learning rate for MPNN training                   

# --- Shared Configuration ---
# Change depending on the resolution of the downloaded dataset
IMAGE_SIZE = 384                                                        # Size of the input images for the vision model                 
MAX_TEXT_LENGTH = 256                                                   # Maximum length of text input for the language model          
DROPOUT = 0.2                                                           # Dropout rate for the model layers               