# datasets.py

import os
import io
import glob
import logging
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as PyGDataset
from torchvision import transforms

import config
import utils

# --- Stage 1 Dataset & Collator ---

class Stage1Dataset(Dataset):
    def __init__(self, df, transform):
        self.df, self.transform = df, transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(io.BytesIO(row['image_bytes'])).convert('RGB')
        return {"image": self.transform(image), "text": row["text"], "pneumonia_label": row["Pneumonia"]}

class Stage1Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, batch):
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text'] for item in batch]
        labels = self.tokenizer(texts, padding='longest', truncation=True, max_length=config.MAX_TEXT_LENGTH, return_tensors='pt').input_ids
        pneumonia_labels = torch.tensor([item['pneumonia_label'] for item in batch], dtype=torch.long)
        return {'images': images, 'labels': labels, 'pneumonia_labels': pneumonia_labels}

def load_data_stage1():
    logging.info(f"[Stage 1] Loading data with JPEG quality={config.STAGE1_JPEG_QUALITY}...")
    try:
        df = pd.read_csv(config.CSV_PATH).head(config.STAGE1_NUM_SAMPLES)
    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at {config.CSV_PATH}."); return pd.DataFrame()

    all_reports = {os.path.basename(p): p for p in glob.glob(os.path.join(config.REPORT_BASE_DIR, '**', '*.txt'), recursive=True)}
    all_images = glob.glob(os.path.join(config.IMAGE_BASE_DIR, '**', '*.jpg'), recursive=True)
    
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Reading & Compressing"):
        try:
            study_id = str(int(row['study_id']))
            report_path = all_reports.get(f"s{study_id}.txt")
            image_path = next((p for p in all_images if f'/s{study_id}/' in p), None)
            if report_path and image_path:
                with open(image_path, 'rb') as f: original_bytes = f.read()
                img = Image.open(io.BytesIO(original_bytes))
                buffer = io.BytesIO()
                img.convert('RGB').save(buffer, format="JPEG", quality=config.STAGE1_JPEG_QUALITY)
                with open(report_path, 'r', encoding='utf-8') as f: text_content = f.read()
                impression = utils.extract_impression(text_content)
                if impression:
                    processed_data.append({'image_bytes': buffer.getvalue(), 'text': impression, 'Pneumonia': int(row['Pneumonia'])})
        except Exception as e:
            logging.warning(f"Skipping study {row.get('study_id', 'N/A')} due to error: {e}")
    return pd.DataFrame(processed_data)


# --- Stage 2 Dataset & Data Loading ---

class FusedGraphDataset(PyGDataset):
    def __init__(self, root, df, attention_model, tokenizer, nlp_model):
        self.df = df
        self.attention_model = attention_model
        self.tokenizer = tokenizer
        self.nlp_model = nlp_model
        super().__init__(root)
        self.processed_data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f'fused_graphs_prune_{config.STAGE2_PRUNING_PERCENTILE}.pt']

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

    def process(self):
        logging.info("[Stage 2] Processing data to create fused graphs...")
        data_list, projection_layer = [], None
        num_patches = 144 # Swin-base-384 gives 12x12=144 patches
        num_patches_side = int(np.sqrt(num_patches))
        patch_grid = np.arange(num_patches).reshape(num_patches_side, num_patches_side)

        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="[Stage 2] Generating Fused Graphs"):
            try:
                image = Image.open(io.BytesIO(row['image_bytes'])).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
                report_text, label = row['text'], torch.tensor([row['Pneumonia']], dtype=torch.float)
                text_ids = self.tokenizer(report_text, return_tensors='pt', max_length=config.MAX_TEXT_LENGTH, truncation=True).input_ids.to(config.DEVICE)
                
                with torch.no_grad():
                    patch_features_full, cross_attentions = self.attention_model(image_tensor, labels=text_ids, pneumonia_labels=torch.tensor([0]).to(config.DEVICE), output_attentions=True)
                
                cross_attention = cross_attentions[-1] # Use last layer's cross-attention
                agg_attention = cross_attention.mean(dim=(0, 1)).squeeze().cpu().numpy()
                
                if agg_attention.size == 0 or agg_attention.ndim == 0: continue
                
                threshold = np.percentile(agg_attention, 100 - config.STAGE2_PRUNING_PERCENTILE)
                pruned_indices = np.where(agg_attention > threshold)[0]
                if len(pruned_indices) == 0: pruned_indices = [np.argmax(agg_attention)] # Keep at least one patch
                    
                pruned_patch_features = patch_features_full.squeeze(0)[pruned_indices].cpu()
                idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(pruned_indices)}
                
                pruned_edges = set()
                for idx in pruned_indices:
                    r, c = np.where(patch_grid == idx)
                    if r.size == 0: continue
                    r, c = r[0], c[0]
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if 0 <= r + dr < num_patches_side and 0 <= c + dc < num_patches_side:
                            neighbor_idx = patch_grid[r + dr, c + dc]
                            if neighbor_idx in idx_map:
                                pruned_edges.add(tuple(sorted((idx_map[idx], idx_map[neighbor_idx]))))
                
                g1 = utils.create_pruned_visual_graph(pruned_patch_features, list(pruned_edges))
                g2 = utils.create_text_knowledge_graph(report_text, self.nlp_model)

                if projection_layer is None and g2.num_nodes > 0:
                     projection_layer = torch.nn.Linear(g2.x.shape[1], g1.x.shape[1]).to(config.DEVICE)
                
                if g2.num_nodes > 0:
                    g2.x = projection_layer(g2.x.to(config.DEVICE)).cpu().detach()

                fused_graph = utils.fuse_graphs(g1, g2)
                fused_graph.y = label
                data_list.append(fused_graph)
            except Exception as e:
                logging.warning(f"Skipping data point at index {index} due to error: {e}")
        
        if not data_list: raise RuntimeError("No data could be processed.")
        torch.save(data_list, self.processed_paths[0])

def load_data_stage2():
    logging.info("[Stage 2] Loading raw data (no compression)...")
    try:
        df = pd.read_csv(config.CSV_PATH).head(config.STAGE2_NUM_SAMPLES)
    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at {config.CSV_PATH}."); return pd.DataFrame()

    all_reports = {os.path.basename(p): p for p in glob.glob(os.path.join(config.REPORT_BASE_DIR, '**', '*.txt'), recursive=True)}
    all_images = glob.glob(os.path.join(config.IMAGE_BASE_DIR, '**', '*.jpg'), recursive=True)
    
    processed_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 2] Reading Raw Data"):
        try:
            study_id = str(int(row['study_id']))
            report_path = all_reports.get(f"s{study_id}.txt")
            image_path = next((p for p in all_images if f'/s{study_id}/' in p), None)
            if report_path and image_path:
                with open(image_path, 'rb') as f: image_bytes = f.read()
                with open(report_path, 'r', encoding='utf-8') as f: text_content = f.read()
                impression = utils.extract_impression(text_content)
                if impression:
                    processed_data.append({'image_bytes': image_bytes, 'text': impression, 'Pneumonia': int(row['Pneumonia'])})
        except Exception as e:
            logging.warning(f"Skipping study {row.get('study_id', 'N/A')} due to error: {e}")
    return pd.DataFrame(processed_data)