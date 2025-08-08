"""
This script defines the neural network architectures used in the NEURAL pipeline.

It includes:
1.  VisionLanguageModel: A multimodal model combining a Swin Transformer for vision
    and a T5 model for language. It's used in Stage 1 for fine-tuning and in
    Stage 2 for generating attention scores.
2.  MPNN: A Message Passing Neural Network (GNN) used in Stage 2 for
    classification on the fused multimodal graphs.
"""

import torch
import torch.nn as nn
from transformers import SwinModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch_geometric.nn import GCNConv, global_mean_pool

# Import project-specific configurations
import config


class VisionLanguageModel(nn.Module):
    """
    A unified vision-language model for joint report generation and classification.

    This model integrates a pre-trained vision encoder (Swin Transformer) with a
    pre-trained language model (T5). The vision features are projected into the
    language model's embedding space and used to condition the text generation
    process. It is trained on two tasks simultaneously: generating a radiological
    report and classifying the image for a specific condition (e.g., pneumonia).

    The model can also output cross-attention scores, which are repurposed in
    Stage 2 to guide the pruning of visual information.
    """
    def __init__(self, dropout: float = config.DROPOUT):
        """
        Initializes the model components.

        Args:
            dropout (float): The dropout rate to use in the projection layer.
        """
        super().__init__()
        # Load the pre-trained Swin Transformer as the vision encoder.
        # `add_pooling_layer=False` ensures we get patch-level features.
        self.vision_model = SwinModel.from_pretrained(
            config.VISION_MODEL_NAME, add_pooling_layer=False, cache_dir='./cache'
        )

        # Load the pre-trained T5 model for conditional text generation.
        self.text_model = T5ForConditionalGeneration.from_pretrained(
            config.LLM_MODEL_NAME, cache_dir='./cache'
        )

        # Get embedding dimensions from the model configs.
        vision_embedding_dim = self.vision_model.config.hidden_size
        text_embedding_dim = self.text_model.config.hidden_size

        # A projection layer to map the vision embedding space to the text
        # embedding space, making them compatible for the T5 decoder.
        self.vision_to_text_projection = nn.Sequential(
            nn.Linear(vision_embedding_dim, text_embedding_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(text_embedding_dim, text_embedding_dim)
        )

        # A simple linear classifier for the parallel image classification task.
        self.classifier = nn.Linear(vision_embedding_dim, 2)

    def forward(self, images, labels=None, pneumonia_labels=None, output_attentions=False):
        """
        Defines the forward pass for the model.

        This method has two main modes:
        1.  Training/Validation (`labels` is not None): Computes both the text
            generation loss and the image classification loss. It can also return
            attention scores if `output_attentions` is True.
        2.  Inference (`labels` is None): Generates text (a report) based on the
            input image.

        Args:
            images (torch.Tensor): A batch of input images.
            labels (torch.Tensor, optional): Ground-truth token sequences for
                                             teacher-forced text generation.
            pneumonia_labels (torch.Tensor, optional): Ground-truth labels for the
                                                       image classification task.
            output_attentions (bool, optional): If True, the model returns patch
                                                features and cross-attention scores.
                                                Defaults to False.

        Returns:
            - In training mode (with attentions): (patch_features, cross_attentions)
            - In training mode (no attentions): (generation_loss, classification_loss, None)
            - In inference mode: Generated token sequence.
        """
        # 1. Get visual embeddings from the vision model.
        visual_outputs = self.vision_model(pixel_values=images)
        visual_embeds = visual_outputs.last_hidden_state  # Shape: (batch_size, num_patches, vision_dim)

        # Keep the original (pre-projection) patch features if needed for Stage 2.
        patch_features = visual_embeds if output_attentions else None

        # 2. Perform image classification.
        # Pool the patch features to get a single vector per image.
        pooled_output = visual_embeds.mean(dim=1)
        logits_cls = self.classifier(pooled_output)

        # 3. Prepare visual embeddings for the language model.
        projected_embeds = self.vision_to_text_projection(visual_embeds)
        encoder_attention_mask = torch.ones(projected_embeds.shape[:2], device=images.device)

        # Training/Validation Mode
        if labels is not None:
            # Pass the projected visual embeddings directly to the T5 decoder's
            # cross-attention mechanism by setting them as `encoder_outputs`.
            outputs = self.text_model(
                encoder_outputs=(projected_embeds,),
                attention_mask=encoder_attention_mask,
                labels=labels,
                output_attentions=output_attentions,
                return_dict=True
            )

            # Calculate generation loss (CrossEntropy for text).
            loss_fct_gen = nn.CrossEntropyLoss(ignore_index=-100)
            gen_loss = loss_fct_gen(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))

            # Calculate classification loss.
            loss_fct_cls = nn.CrossEntropyLoss()
            cls_loss = loss_fct_cls(logits_cls, pneumonia_labels)

            cross_attentions = outputs.cross_attentions if output_attentions else None

            # For Stage 2, we need the features and attentions, not the losses.
            if output_attentions:
                return patch_features, cross_attentions
            # For standard Stage 1 training, we need the losses.
            else:
                return gen_loss, cls_loss, None

        # Inference/Generation Mode
        else:
            # Package the visual embeddings in the format expected by the `generate` method.
            encoder_outputs_struct = BaseModelOutput(last_hidden_state=projected_embeds)
            # Generate a report using beam search.
            return self.text_model.generate(
                encoder_outputs=encoder_outputs_struct,
                attention_mask=encoder_attention_mask,
                max_length=config.MAX_TEXT_LENGTH,
                num_beams=5,
                early_stopping=True
            )


class MPNN(nn.Module):
    """
    A Message Passing Neural Network (MPNN) for graph classification.

    This model uses two Graph Convolutional Network (GCN) layers to learn node
    representations by aggregating information from their neighbors. A global
-    pooling layer then aggregates all node features into a single graph-level
    representation, which is passed to a linear classifier to make a final
    prediction. This architecture is used in Stage 2 for classification of the
    [cite_start]fused multimodal graphs. [cite: 145]
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Initializes the GNN layers.

        Args:
            input_dim (int): The dimensionality of the input node features.
            hidden_dim (int): The dimensionality of the hidden layers.
        """
        super().__init__()
        # First GCN layer: transforms input features to hidden dimension.
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Second GCN layer: further refines the node representations.
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Final classifier to make a prediction from the graph-level embedding.
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        """
        Defines the forward pass for a batch of graphs.

        Args:
            data (torch_geometric.data.Batch): A batch of graph data from a
                                               PyG DataLoader.

        Returns:
            torch.Tensor: The final classification logits for the batch of graphs.
        """
        # Unpack the graph batch object.
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply the first GCN layer followed by a ReLU activation.
        x = torch.relu(self.conv1(x, edge_index))

        # Apply the second GCN layer followed by a ReLU activation.
        x = torch.relu(self.conv2(x, edge_index))

        # Use global mean pooling to aggregate node features into a single
        # feature vector for each graph in the batch.
        pooled_graph_embedding = global_mean_pool(x, batch)

        # Pass the graph-level embedding through the final classifier.
        return self.classifier(pooled_graph_embedding)