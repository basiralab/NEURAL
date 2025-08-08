"""
This script defines and executes Stage 1 of the NEURAL pipeline: fine-tuning
the multimodal vision-language model.

The primary goal of this stage is to train the model to generate radiology
reports from chest X-ray images while also learning to classify the images for a
specific condition (e.g., pneumonia). The cross-attention scores learned during
this process are essential for the pruning step in Stage 2.
"""

import logging
import torch
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer

# Import project-specific configurations, models, and dataset utilities
import config
from models import VisionLanguageModel
from datasets import Stage1Dataset, Stage1Collator, load_data_stage1


def run_stage1():
    """
    End-to-end training process for Stage 1.

    This function performs the following steps:
    1.  Sets up data augmentation and transformation pipelines.
    2.  Loads the pre-processed data from disk.
    3.  Splits the data into training and validation sets.
    4.  Initializes the vision-language model, tokenizer, and data loaders.
    5.  Sets up the optimizer with different learning rates for vision and text components.
    6.  Executes the training and validation loop for a specified number of epochs.
    7.  Saves the best model checkpoint based on validation loss.
    """
    logging.info("===== STARTING STAGE 1: FINE-TUNING REPORT GENERATOR =====")

    # Define the image transformations for data augmentation and normalization.
    # These operations help the model generalize better by seeing varied images.
    transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the master dataframe containing image paths and metadata.
    master_df = load_data_stage1()
    if master_df.empty:
        logging.error("[Stage 1] No data loaded. Exiting.")
        return

    # Split the dataset into training and validation sets.
    # `stratify` ensures that the proportion of pneumonia cases is the same
    # in both the training and validation sets, which is crucial for imbalanced data.
    train_df, val_df = train_test_split(
        master_df,
        test_size=0.2,
        random_state=42,
        stratify=master_df['Pneumonia']
    )

    # Initialize the tokenizer for the text model.
    tokenizer = T5Tokenizer.from_pretrained(config.LLM_MODEL_NAME, cache_dir='./cache', legacy=False)

    # Create PyTorch Dataset and DataLoader instances for training and validation.
    train_ds = Stage1Dataset(train_df, transform)
    val_ds = Stage1Dataset(val_df, transform)
    collator = Stage1Collator(tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.STAGE1_BATCH_SIZE,
        shuffle=True,
        collate_fn=collator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.STAGE1_BATCH_SIZE,
        collate_fn=collator
    )

    # Initialize the model and move it to the configured device (GPU or CPU).
    model = VisionLanguageModel().to(config.DEVICE)

    # Set up the AdamW optimizer. Different learning rates are used for the vision
    # and text components, a common technique for fine-tuning multimodal models.
    # The vision backbone often requires a smaller learning rate than the new layers.
    optimizer = optim.AdamW([
        {'params': model.text_model.parameters(), 'lr': config.STAGE1_LR_LM},
        {'params': model.vision_model.parameters(), 'lr': config.STAGE1_LR_VISION},
        {'params': model.vision_to_text_projection.parameters(), 'lr': config.STAGE1_LR_LM},
        {'params': model.classifier.parameters(), 'lr': config.STAGE1_LR_VISION}
    ])

    best_val_loss = float('inf')
    logging.info(f"Starting fine-tuning for {config.STAGE1_EPOCHS} epochs...")

    # Main training loop.
    for epoch in range(1, config.STAGE1_EPOCHS + 1):
        # Set the model to training mode.
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()  # Clear gradients at the start of the epoch.

        # Training phase.
        for i, batch in enumerate(tqdm(train_loader, desc=f"[Stage 1] Epoch {epoch} Training")):
            # Move batch data to the target device.
            images = batch['images'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            p_labels = batch['pneumonia_labels'].to(config.DEVICE)

            # Forward pass to get losses.
            gen_loss, cls_loss, _ = model(images, labels, p_labels)

            # Combine the generation loss and classification loss.
            # The classification loss is weighted to balance the two tasks.
            # The total loss is scaled down for gradient accumulation.
            total_loss = (gen_loss + 0.5 * cls_loss) / config.STAGE1_ACCUMULATION_STEPS

            # Backward pass to compute gradients.
            total_loss.backward()
            total_train_loss += total_loss.item() * config.STAGE1_ACCUMULATION_STEPS

            # Gradient accumulation: update model weights only after a set number of steps.
            # This effectively simulates a larger batch size, which can stabilize training,
            # without requiring more GPU memory.
            if (i + 1) % config.STAGE1_ACCUMULATION_STEPS == 0:
                optimizer.step()  # Update weights.
                optimizer.zero_grad()  # Reset gradients for the next accumulation cycle.

        # Set the model to evaluation mode.
        model.eval()
        total_val_loss = 0

        # Validation phase.
        # `torch.no_grad()` disables gradient calculation, which speeds up
        # inference and reduces memory usage.
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Stage 1] Epoch {epoch} Validation"):
                images = batch['images'].to(config.DEVICE)
                labels = batch['labels'].to(config.DEVICE)
                p_labels = batch['pneumonia_labels'].to(config.DEVICE)

                # Forward pass to get losses.
                gen_loss, cls_loss, _ = model(images, labels, p_labels)
                total_val_loss += (gen_loss + 0.5 * cls_loss).item()

        # Calculate and log average losses for the epoch.
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpointing: Save the model if the validation loss has improved.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config.STAGE1_CHECKPOINT_PATH)
            logging.info(f"New best model saved to {config.STAGE1_CHECKPOINT_PATH}")

    logging.info("===== STAGE 1 COMPLETE =====")