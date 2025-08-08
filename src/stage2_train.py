"""
This script defines and executes Stage 2 of the NEURAL pipeline: attention-guided
pruning and Graph Neural Network (GNN) training.

The process involves these key steps:
1.  Loading the fine-tuned vision-language model from Stage 1.
2.  Using this model to generate attention scores for high-fidelity images.
3.  Creating a `FusedGraphDataset`, which performs the core NEURAL logic:
    - Pruning the image patches based on attention scores.
    - Creating a visual graph from the pruned patches.
    - Creating a knowledge graph from the corresponding report text.
    - Fusing these two graphs into a single, unified multimodal representation.
4.  Training a Message Passing Neural Network (MPNN) on these fused graphs for
    the final diagnostic task (e.g., pneumonia classification).
5.  Evaluating the GNN using ROC AUC and saving the best-performing model.
"""

import os
import logging
import spacy
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
from torch_geometric.loader import DataLoader as PyGDataLoader

# Import project-specific configurations, models, and dataset utilities
import config
from models import VisionLanguageModel, MPNN
from datasets import FusedGraphDataset, load_data_stage2


def run_stage2():
    """
    Orchestrates the end-to-end process for Stage 2.

    This function handles loading all necessary models and data, creating the
    fused graph datasets, and running the training and validation loop for the
    GNN classifier.
    """
    logging.info("===== STARTING STAGE 2: NEURAL PRUNING AND GNN TRAINING =====")

    # Stage 2 depends on the model fine-tuned in Stage 1.
    # Check if the checkpoint exists before proceeding.
    if not os.path.exists(config.STAGE1_CHECKPOINT_PATH):
        logging.error(
            f"FATAL: Stage 1 checkpoint '{config.STAGE1_CHECKPOINT_PATH}' not found. "
            "Please run stage 1 first."
        )
        return

    # Load the fine-tuned vision-language model from Stage 1.
    # This model is not trained further; it's used in evaluation mode to
    # generate the attention scores needed for pruning.
    logging.info(f"Loading fine-tuned model from {config.STAGE1_CHECKPOINT_PATH}")
    attention_model = VisionLanguageModel().to(config.DEVICE)
    attention_model.load_state_dict(torch.load(config.STAGE1_CHECKPOINT_PATH))
    attention_model.eval()  # Set to evaluation mode.

    # Initialize the tokenizer and the spaCy NLP model for text processing.
    tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir='./cache', legacy=False)
    try:
        nlp = spacy.load(config.SPACY_MODEL_NAME)
    except OSError:
        logging.error(
            f"SpaCy model '{config.SPACY_MODEL_NAME}' not found. "
            f"Please run: python -m spacy download {config.SPACY_MODEL_NAME}"
        )
        return

    # Load the high-fidelity, uncompressed data for Stage 2.
    master_df = load_data_stage2()
    if master_df.empty:
        logging.error("[Stage 2] No data loaded. Exiting.")
        return

    # Split the data into training and validation sets.
    train_df, val_df = train_test_split(
        master_df,
        test_size=0.3,
        random_state=42,
        stratify=master_df['Pneumonia']
    )

    # PyG requires a specific directory structure for processed data.
    # Ensure these directories exist before creating the dataset.
    os.makedirs('data/train/processed', exist_ok=True)
    os.makedirs('data/val/processed', exist_ok=True)

    # Instantiate the FusedGraphDataset. This is a critical step where the
    # actual pruning and graph creation logic (in `datasets.py`) is triggered.
    # The dataset will process the raw data and save the fused graphs to disk.
    train_graph_ds = FusedGraphDataset('data/train', train_df, attention_model, tokenizer, nlp)
    val_graph_ds = FusedGraphDataset('data/val', val_df, attention_model, tokenizer, nlp)

    # Create PyTorch Geometric DataLoaders to handle batches of graph data.
    train_loader = PyGDataLoader(train_graph_ds, batch_size=config.STAGE2_BATCH_SIZE_MPNN, shuffle=True)
    val_loader = PyGDataLoader(val_graph_ds, batch_size=config.STAGE2_BATCH_SIZE_MPNN)

    # Initialize the GNN model (MPNN). The input dimension must match the
    # feature dimension of the nodes in the fused graph, which comes from the
    # vision model's hidden size.
    mpnn_model = MPNN(input_dim=attention_model.vision_model.config.hidden_size).to(config.DEVICE)
    optimizer = optim.Adam(mpnn_model.parameters(), lr=config.STAGE2_LR_MPNN)
    # Use BCEWithLogitsLoss as it's numerically stable for binary classification.
    loss_fn = nn.BCEWithLogitsLoss()

    logging.info(f"Starting MPNN training for {config.STAGE2_EPOCHS_MPNN} epochs...")
    best_val_auc = 0

    # Main training loop for the GNN.
    for epoch in range(1, config.STAGE2_EPOCHS_MPNN + 1):
        mpnn_model.train()  # Set the model to training mode.
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Stage 2] Epoch {epoch} Training"):
            batch = batch.to(config.DEVICE)  # Move graph batch to the target device.
            optimizer.zero_grad()
            out = mpnn_model(batch)  # Forward pass.
            # Calculate loss between model output and ground truth labels.
            loss = loss_fn(out.squeeze(), batch.y.squeeze())
            loss.backward()  # Compute gradients.
            optimizer.step()  # Update weights.
            total_loss += loss.item()

        # Validation phase.
        mpnn_model.eval()  # Set the model to evaluation mode.
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.DEVICE)
                out = mpnn_model(batch)
                # Apply sigmoid to get probabilities and store predictions and labels.
                all_preds.extend(torch.sigmoid(out).cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        # Calculate ROC AUC score for the validation set.
        # Handle the case where a batch might contain only one class.
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        logging.info(
            f"Epoch {epoch:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Val AUC: {val_auc:.4f}"
        )

        # Checkpointing: Save the model if it has the best validation AUC so far.
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(mpnn_model.state_dict(), config.STAGE2_CHECKPOINT_PATH)
            logging.info(f"New best GNN model saved with AUC: {best_val_auc:.4f}")

    logging.info("===== STAGE 2 COMPLETE =====")