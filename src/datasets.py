"""
This file contains all data handling logic for the NEURAL pipeline. It defines the PyTorch `Dataset` and `DataLoader` structures needed for both training stages. Key responsibilities include:

Loading raw data: Reading DICOM images and text reports from disk.
Pre-processing: Converting DICOM files into a usable format and preparing them for the models.
Graph Creation: Handling the complex process of converting image-text pairs into pruned, fused multimodal graphs for Stage 2.
"""

import os
import io
import glob
import logging
import torch
import numpy as np
import pandas as pd
import pydicom  # Added for DICOM file support
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as PyGDataset
from torchvision import transforms

# Import project-specific configurations and utility functions
import config
import utils


def dicom_to_pil_image(path: str) -> Image.Image:
    """
    Reads a DICOM file and converts it into a PIL Image object.

    This function handles the conversion by reading the pixel data, normalizing
    it to a standard 8-bit range (0-255), and then creating an image object.
    This is necessary because medical images often have a higher bit depth.

    Args:
        path (str): The file path to the DICOM (.dcm) file.

    Returns:
        Image.Image: A PIL Image object representing the DICOM image.
    """
    # Read the DICOM file using pydicom
    dicom_file = pydicom.dcmread(path)

    # Access the pixel data and ensure it's in a floating-point format for normalization
    pixels = dicom_file.pixel_array.astype(float)

    # Normalize the pixel array to a 0-255 scale
    pixels = (pixels - np.min(pixels)) / (np.max(pixels) - np.min(pixels)) * 255.0

    # Convert the normalized float array to an 8-bit unsigned integer array
    pixels = pixels.astype(np.uint8)

    # Create and return a PIL Image from the processed numpy array
    return Image.fromarray(pixels)


class Stage1Dataset(Dataset):
    """
    A PyTorch Dataset for loading image-report pairs for Stage 1 fine-tuning.

    This dataset takes a pandas DataFrame containing pre-processed data. Each item
    consists of an image (read from bytes), its corresponding report text, and a
    pneumonia label.
    """
    def __init__(self, df: pd.DataFrame, transform: transforms.Compose):
        """
        Initializes the dataset.

        Args:
            df (pd.DataFrame): A DataFrame with 'image_bytes', 'text', 'Pneumonia'.
            transform (transforms.Compose): A torchvision transform for images.
        """
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary with the transformed image tensor, report
                  text, and the pneumonia label.
        """
        # Retrieve the data for the given index
        row = self.df.iloc[idx]

        # The image bytes were pre-converted to a standard format (JPEG/PNG)
        # during the data loading step, so PIL can open them directly.
        image = Image.open(io.BytesIO(row['image_bytes'])).convert('RGB')

        return {
            "image": self.transform(image),
            "text": row["text"],
            "pneumonia_label": row["Pneumonia"]
        }


class Stage1Collator:
    """
    A collator for the Stage 1 DataLoader.

    This class takes a batch of samples from Stage1Dataset and formats them
    into tensors suitable for model input. It handles image stacking and text
    tokenization with padding.
    """
    def __init__(self, tokenizer):
        """
        Initializes the collator.

        Args:
            tokenizer: A Hugging Face tokenizer for processing text.
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: list) -> dict:
        """
        Processes a batch of data.

        Args:
            batch (list): A list of dictionary samples from Stage1Dataset.

        Returns:
            dict: A dictionary of batched tensors for 'images', 'labels',
                  and 'pneumonia_labels'.
        """
        # Stack all image tensors into a single batch tensor
        images = torch.stack([item['image'] for item in batch])
        # Collect all text reports in the batch
        texts = [item['text'] for item in batch]
        # Tokenize the texts, padding to the length of the longest sequence
        labels = self.tokenizer(
            texts,
            padding='longest',
            truncation=True,
            max_length=config.MAX_TEXT_LENGTH,
            return_tensors='pt'
        ).input_ids
        # Create a tensor for the pneumonia labels
        pneumonia_labels = torch.tensor([item['pneumonia_label'] for item in batch], dtype=torch.long)

        return {'images': images, 'labels': labels, 'pneumonia_labels': pneumonia_labels}


class FusedGraphDataset(PyGDataset):
    """
    A PyTorch Geometric Dataset for creating and loading fused multimodal graphs.

    This class handles the complex processing of converting image-text pairs into
    a unified graph structure. It uses a pre-trained attention model to prune the
    visual data and fuses the result with a knowledge graph derived from the text.
    The processed graphs are saved to disk to avoid regeneration.
    """
    def __init__(self, root: str, df: pd.DataFrame, attention_model, tokenizer, nlp_model):
        self.df = df
        self.attention_model = attention_model
        self.tokenizer = tokenizer
        self.nlp_model = nlp_model
        super().__init__(root)
        # Load the processed data from disk after initialization
        self.processed_data = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self) -> list:
        """Returns the name of the file where the processed data is stored."""
        return [f'fused_graphs_prune_{config.STAGE2_PRUNING_PERCENTILE}.pt']

    def len(self) -> int:
        """Returns the total number of processed graphs."""
        return len(self.processed_data)

    def get(self, idx: int):
        """Retrieves a single processed graph by index."""
        return self.processed_data[idx]

    def process(self):
        """
        Core logic for converting raw data into fused graphs. This is only run
        once if the processed file doesn't already exist.
        """
        logging.info("[Stage 2] Processing data to create fused graphs...")
        data_list, projection_layer = [], None
        # Swin-base-384 gives a 12x12=144 patch grid
        num_patches, num_patches_side = 144, int(np.sqrt(144))
        patch_grid = np.arange(num_patches).reshape(num_patches_side, num_patches_side)

        for index, row in tqdm(self.df.iterrows(), total=len(self.df), desc="[Stage 2] Generating Fused Graphs"):
            try:
                # Pre-process Image and Text
                image = Image.open(io.BytesIO(row['image_bytes'])).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
                report_text, label = row['text'], torch.tensor([row['Pneumonia']], dtype=torch.float)
                text_ids = self.tokenizer(report_text, return_tensors='pt', max_length=config.MAX_TEXT_LENGTH, truncation=True).input_ids.to(config.DEVICE)

                # Extract Attention Scores and Prune Patches
                with torch.no_grad():
                    patch_features_full, cross_attentions = self.attention_model(
                        image_tensor, labels=text_ids,
                        pneumonia_labels=torch.tensor([0]).to(config.DEVICE), output_attentions=True
                    )
                # Aggregate last layer's attention scores
                agg_attention = cross_attentions[-1].mean(dim=(0, 1)).squeeze().cpu().numpy()
                if agg_attention.size == 0 or agg_attention.ndim == 0:
                    continue

                # Prune based on attention percentile
                threshold = np.percentile(agg_attention, 100 - config.STAGE2_PRUNING_PERCENTILE)
                pruned_indices = np.where(agg_attention > threshold)[0]
                if len(pruned_indices) == 0:  # Keep at least one patch
                    pruned_indices = [np.argmax(agg_attention)]

                # Create Pruned Visual Graph (G1)
                pruned_patch_features = patch_features_full.squeeze(0)[pruned_indices].cpu()
                idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(pruned_indices)}
                pruned_edges = set()
                for idx in pruned_indices:
                    r, c = np.where(patch_grid == idx)
                    if r.size == 0: continue
                    r, c = r[0], c[0]
                    # Check for adjacent neighbors (up, down, left, right)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        if 0 <= r + dr < num_patches_side and 0 <= c + dc < num_patches_side:
                            neighbor_idx = patch_grid[r + dr, c + dc]
                            if neighbor_idx in idx_map:
                                pruned_edges.add(tuple(sorted((idx_map[idx], idx_map[neighbor_idx]))))
                g1 = utils.create_pruned_visual_graph(pruned_patch_features, list(pruned_edges))

                # Create Text Knowledge Graph (G2) and Fuse
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

        if not data_list:
            raise RuntimeError("No data could be processed. Check data paths and content.")
        torch.save(data_list, self.processed_paths[0])


def load_data_stage1() -> pd.DataFrame:
    """
    Loads and preprocesses data for Stage 1 from DICOM files.

    This function reads a master CSV, finds corresponding DICOM (.dcm) and
    report (.txt) files, converts the DICOM to a PIL Image, compresses it to
    a specified JPEG quality, extracts the 'IMPRESSION' from the report,
    and returns a consolidated pandas DataFrame.

    Returns:
        pd.DataFrame: Contains 'image_bytes', 'text', and 'Pneumonia' for
                      each sample. Returns an empty DataFrame on failure.
    """
    logging.info(f"[Stage 1] Loading DICOM data and compressing with JPEG quality={config.STAGE1_JPEG_QUALITY}...")
    try:
        df = pd.read_csv(config.CSV_PATH).head(config.STAGE1_NUM_SAMPLES)
    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at {config.CSV_PATH}.")
        return pd.DataFrame()

    all_reports = {os.path.basename(p): p for p in glob.glob(os.path.join(config.REPORT_BASE_DIR, '**', '*.txt'), recursive=True)}
    all_images = glob.glob(os.path.join(config.IMAGE_BASE_DIR, '**', '*.dcm'), recursive=True)
    processed_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 1] Reading DICOMs & Compressing"):
        try:
            study_id = str(int(row['study_id']))
            report_path = all_reports.get(f"s{study_id}.txt")
            image_path = next((p for p in all_images if f'/s{study_id}/' in p), None)

            if report_path and image_path:
                # Convert DICOM file to a PIL Image object
                pil_img = dicom_to_pil_image(image_path)
                # Re-compress the image to simulate resource-constrained settings
                buffer = io.BytesIO()
                pil_img.convert('RGB').save(buffer, format="JPEG", quality=config.STAGE1_JPEG_QUALITY)
                compressed_bytes = buffer.getvalue()

                with open(report_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                impression = utils.extract_impression(text_content)
                if impression:
                    processed_data.append({
                        'image_bytes': compressed_bytes,
                        'text': impression,
                        'Pneumonia': int(row['Pneumonia'])
                    })
        except Exception as e:
            logging.warning(f"Skipping study {row.get('study_id', 'N/A')} due to error: {e}")
    return pd.DataFrame(processed_data)


def load_data_stage2() -> pd.DataFrame:
    """
    Loads data for Stage 2 from DICOM files.

    This function reads DICOM images, converts them to high-quality PNG format
    in memory, and prepares them for the graph generation process. This ensures
    the original image fidelity is maintained for this stage.

    Returns:
        pd.DataFrame: A DataFrame with raw image bytes (as PNG), text, and
                      labels. Returns an empty DataFrame on failure.
    """
    logging.info("[Stage 2] Loading DICOM data (as high-quality)...")
    try:
        df = pd.read_csv(config.CSV_PATH).head(config.STAGE2_NUM_SAMPLES)
    except FileNotFoundError:
        logging.error(f"FATAL: CSV file not found at {config.CSV_PATH}.")
        return pd.DataFrame()

    all_reports = {os.path.basename(p): p for p in glob.glob(os.path.join(config.REPORT_BASE_DIR, '**', '*.txt'), recursive=True)}
    all_images = glob.glob(os.path.join(config.IMAGE_BASE_DIR, '**', '*.dcm'), recursive=True)
    processed_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="[Stage 2] Reading DICOMs"):
        try:
            study_id = str(int(row['study_id']))
            report_path = all_reports.get(f"s{study_id}.txt")
            image_path = next((p for p in all_images if f'/s{study_id}/' in p), None)

            if report_path and image_path:
                # Convert DICOM to a PIL Image
                pil_img = dicom_to_pil_image(image_path)
                # Save the image to a buffer in a lossless format (PNG)
                buffer = io.BytesIO()
                pil_img.convert('RGB').save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

                with open(report_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                impression = utils.extract_impression(text_content)
                if impression:
                    processed_data.append({
                        'image_bytes': image_bytes,
                        'text': impression,
                        'Pneumonia': int(row['Pneumonia'])
                    })
        except Exception as e:
            logging.warning(f"Skipping study {row.get('study_id', 'N/A')} due to error: {e}")
    return pd.DataFrame(processed_data)