import openai
import torch
import torch.nn as nn

class OpenaiAbusiveCommentClassifier(nn.Module):
    """
    Abusive Comment Classifier using GPT-4 embeddings and deep classification layers.
    
    Args:
        embedding_dim (int): Dimension of the GPT-4 embeddings.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        use_attention (bool): Whether to use an attention mechanism to combine embeddings.
    """

    def __init__(self, embedding_dim=1536, num_classes=2, dropout_rate=0.3, use_attention=False):
        super(OpenaiAbusiveCommentClassifier, self).__init__()
        
        self.use_attention = use_attention
        if use_attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(embedding_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
                nn.Softmax(dim=1)
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
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
            
            nn.Linear(256, num_classes)
        )

    def forward(self, embeddings):
        """
        Forward pass through the classifier.

        Args:
            embeddings (torch.Tensor): Input embeddings from GPT-4 API.
        
        Returns:
            torch.Tensor: Logits for each class.
        """
        print(f"Input embeddings shape: {embeddings.shape}")
        
        if self.use_attention:
            # Apply attention mechanism to combine embeddings
            print("Using attention mechanism")
            attention_weights = self.attention_layer(embeddings)
            print(f"Attention weights shape: {attention_weights.shape}")
            embeddings = torch.sum(attention_weights * embeddings, dim=1)
            print(f"Attended embeddings shape: {embeddings.shape}")
        else:
            # If no attention, ensure embeddings are the correct shape
            embeddings = embeddings.mean(dim=1) if embeddings.dim() > 2 else embeddings
        
        # Passing through classifier layers
        print("Passing through classifier layers")
        logits = self.classifier(embeddings)
        print(f"Logits shape: {logits.shape}")
        
        return logits