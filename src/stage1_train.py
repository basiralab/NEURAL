import logging
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer

import config
from models import VisionLanguageModel
from datasets import Stage1Dataset, Stage1Collator, load_data_stage1

def run_stage1():
    logging.info("===== STARTING STAGE 1: FINE-TUNING REPORT GENERATOR =====")
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    master_df = load_data_stage1()
    if master_df.empty:
        logging.error("[Stage 1] No data loaded. Exiting.")
        return

    train_df, val_df = train_test_split(master_df, test_size=0.2, random_state=42, stratify=master_df['Pneumonia'])
    tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir='./cache', legacy=False)
    
    train_ds = Stage1Dataset(train_df, transform)
    val_ds = Stage1Dataset(val_df, transform)
    
    collator = Stage1Collator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config.STAGE1_BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=config.STAGE1_BATCH_SIZE, collate_fn=collator)
    
    model = VisionLanguageModel().to(config.DEVICE)
    optimizer = optim.AdamW([
        {'params': model.text_model.parameters(), 'lr': config.STAGE1_LR_LM},
        {'params': model.vision_model.parameters(), 'lr': config.STAGE1_LR_VISION},
        {'params': model.vision_to_text_projection.parameters(), 'lr': config.STAGE1_LR_LM},
        {'params': model.classifier.parameters(), 'lr': config.STAGE1_LR_VISION}
    ])
    
    best_val_loss = float('inf')
    logging.info(f"Starting fine-tuning for {config.STAGE1_EPOCHS} epochs...")

    for epoch in range(1, config.STAGE1_EPOCHS + 1):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"[Stage 1] Epoch {epoch} Training")):
            images = batch['images'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            p_labels = batch['pneumonia_labels'].to(config.DEVICE)
            
            gen_loss, cls_loss, _ = model(images, labels, p_labels)
            total_loss = (gen_loss + 0.5 * cls_loss) / config.STAGE1_ACCUMULATION_STEPS
            total_loss.backward()
            total_train_loss += total_loss.item() * config.STAGE1_ACCUMULATION_STEPS
            
            if (i + 1) % config.STAGE1_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Stage 1] Epoch {epoch} Validation"):
                images = batch['images'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                p_labels = batch['pneumonia_labels'].to(config.DEVICE)
                
                gen_loss, cls_loss, _ = model(images, labels, p_labels)
                total_val_loss += (gen_loss + 0.5 * cls_loss).item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.STAGE1_CHECKPOINT_PATH)
            logging.info(f"New best model saved to {config.STAGE1_CHECKPOINT_PATH}")
            
    logging.info("===== STAGE 1 COMPLETE =====")