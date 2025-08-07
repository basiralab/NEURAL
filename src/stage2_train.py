# stage2_train.py

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

import config
from models import VisionLanguageModel, MPNN
from datasets import FusedGraphDataset, load_data_stage2

def run_stage2():
    logging.info("===== STARTING STAGE 2: NEURAL PRUNING AND GNN TRAINING =====")
    if not os.path.exists(config.STAGE1_CHECKPOINT_PATH):
        logging.error(f"FATAL: Stage 1 checkpoint '{config.STAGE1_CHECKPOINT_PATH}' not found. Please run stage 1 first.")
        return

    logging.info(f"Loading fine-tuned model from {config.STAGE1_CHECKPOINT_PATH}")
    attention_model = VisionLanguageModel().to(config.DEVICE)
    attention_model.load_state_dict(torch.load(config.STAGE1_CHECKPOINT_PATH))
    attention_model.eval()

    tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir='./cache', legacy=False)
    
    try:
        nlp = spacy.load(config.SPACY_MODEL_NAME)
    except OSError:
        logging.error(f"SpaCy model '{config.SPACY_MODEL_NAME}' not found. Please run: python -m spacy download {config.SPACY_MODEL_NAME}")
        return

    master_df = load_data_stage2()
    if master_df.empty:
        logging.error("[Stage 2] No data loaded. Exiting.")
        return

    train_df, val_df = train_test_split(master_df, test_size=0.3, random_state=42, stratify=master_df['Pneumonia'])
    
    # Create directories for processed data if they don't exist
    os.makedirs('data/train/processed', exist_ok=True)
    os.makedirs('data/val/processed', exist_ok=True)

    train_graph_ds = FusedGraphDataset('data/train', train_df, attention_model, tokenizer, nlp)
    val_graph_ds = FusedGraphDataset('data/val', val_df, attention_model, tokenizer, nlp)
    
    train_loader = PyGDataLoader(train_graph_ds, batch_size=config.STAGE2_BATCH_SIZE_MPNN, shuffle=True)
    val_loader = PyGDataLoader(val_graph_ds, batch_size=config.STAGE2_BATCH_SIZE_MPNN)
    
    mpnn_model = MPNN(input_dim=attention_model.vision_model.config.hidden_size).to(config.DEVICE)
    optimizer = optim.Adam(mpnn_model.parameters(), lr=config.STAGE2_LR_MPNN)
    loss_fn = nn.BCEWithLogitsLoss()
    
    logging.info(f"Starting MPNN training for {config.STAGE2_EPOCHS_MPNN} epochs...")
    best_val_auc = 0

    for epoch in range(1, config.STAGE2_EPOCHS_MPNN + 1):
        mpnn_model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"[Stage 2] Epoch {epoch} Training"):
            batch = batch.to(config.DEVICE)
            optimizer.zero_grad()
            out = mpnn_model(batch)
            loss = loss_fn(out.squeeze(), batch.y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mpnn_model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.DEVICE)
                out = mpnn_model(batch)
                all_preds.extend(torch.sigmoid(out).cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                
        val_auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.5
        logging.info(f"Epoch {epoch:02d} | Train Loss: {total_loss / len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(mpnn_model.state_dict(), config.STAGE2_CHECKPOINT_PATH)
            logging.info(f"New best GNN model saved with AUC: {best_val_auc:.4f}")
            
    logging.info("===== STAGE 2 COMPLETE =====")