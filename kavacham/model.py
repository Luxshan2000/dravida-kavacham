import torch
import torch.nn as nn
from transformers import XLMRobertaModel


class _AbusiveCommentClassifier(nn.Module):
    """
    Abusive Comment Classifier using XLM-RoBERTa with multi-head
    attention and deep classification layers.

    Args:
        model_name (str): Pre-trained transformer model name.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
    """

    def __init__(self, model_name="xlm-roberta-base", num_classes=2, dropout_rate=0.3):
        super(_AbusiveCommentClassifier, self).__init__()

        self.transformer = XLMRobertaModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.attention_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                    nn.Softmax(dim=1),
                )
                for _ in range(4)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        print("Step 1: Passing through the transformer model")
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        print(f"Transformer output shape: {sequence_output.shape}")

        print("Step 2: Applying attention heads")
        attended_outputs = []
        for idx, attention in enumerate(self.attention_heads):
            print(f"  Attention head {idx + 1}: Computing attention weights")
            attention_weights = attention(sequence_output)
            print(f"    Attention weights shape: {attention_weights.shape}")
            attended_output = torch.sum(attention_weights * sequence_output, dim=1)
            print(f"    Attended output shape: {attended_output.shape}")
            attended_outputs.append(attended_output)

        print("Step 3: Concatenating attended outputs")
        combined_output = torch.cat(attended_outputs, dim=1)
        print(f"Combined output shape: {combined_output.shape}")

        print("Step 4: Passing through classifier layers")
        logits = self.classifier(combined_output)
        print(f"Logits shape: {logits.shape}")

        return logits
