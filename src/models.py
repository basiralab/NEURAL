# models.py

import torch
import torch.nn as nn
from transformers import SwinModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch_geometric.nn import GCNConv, global_mean_pool

import config

class VisionLanguageModel(nn.Module):
    def __init__(self, dropout=config.DROPOUT):
        super().__init__()
        self.vision_model = SwinModel.from_pretrained(config.VISION_MODEL_NAME, add_pooling_layer=False, cache_dir='./cache')
        self.text_model = T5ForConditionalGeneration.from_pretrained(config.LLM_MODEL_NAME, cache_dir='./cache')
        vision_embedding_dim = self.vision_model.config.hidden_size
        text_embedding_dim = self.text_model.config.hidden_size
        self.vision_to_text_projection = nn.Sequential(
            nn.Linear(vision_embedding_dim, text_embedding_dim), nn.Dropout(dropout),
            nn.ReLU(), nn.Linear(text_embedding_dim, text_embedding_dim)
        )
        self.classifier = nn.Linear(vision_embedding_dim, 2)

    def forward(self, images, labels=None, pneumonia_labels=None, output_attentions=False):
        visual_outputs = self.vision_model(pixel_values=images)
        visual_embeds = visual_outputs.last_hidden_state
        
        # Extract patch features before projection for use in Stage 2
        patch_features = visual_embeds if output_attentions else None

        pooled_output = visual_embeds.mean(dim=1)
        logits_cls = self.classifier(pooled_output)
        projected_embeds = self.vision_to_text_projection(visual_embeds)
        encoder_attention_mask = torch.ones(projected_embeds.shape[:2], device=images.device)
        
        if labels is not None:
            outputs = self.text_model(
                encoder_outputs=(projected_embeds,), attention_mask=encoder_attention_mask,
                labels=labels, output_attentions=output_attentions, return_dict=True
            )
            loss_fct_gen = nn.CrossEntropyLoss(ignore_index=-100)
            gen_loss = loss_fct_gen(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            loss_fct_cls = nn.CrossEntropyLoss()
            cls_loss = loss_fct_cls(logits_cls, pneumonia_labels)
            cross_attentions = outputs.cross_attentions if output_attentions else None

            if output_attentions:
                return patch_features, cross_attentions
            else:
                 return gen_loss, cls_loss, None
        else:
            encoder_outputs_struct = BaseModelOutput(last_hidden_state=projected_embeds)
            return self.text_model.generate(
                encoder_outputs=encoder_outputs_struct, attention_mask=encoder_attention_mask,
                max_length=config.MAX_TEXT_LENGTH, num_beams=5, early_stopping=True
            )

class MPNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        return self.classifier(global_mean_pool(x, batch))