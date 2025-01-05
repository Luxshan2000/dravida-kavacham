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

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        num_classes: int = 2,
        dropout_rate: float = 0.3,
    ) -> None:
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

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        attended_outputs = []
        for _, attention in enumerate(self.attention_heads):
            attention_weights = attention(sequence_output)
            attended_output = torch.sum(attention_weights * sequence_output, dim=1)
            attended_outputs.append(attended_output)

        combined_output = torch.cat(attended_outputs, dim=1)

        logits = self.classifier(combined_output)

        return logits
