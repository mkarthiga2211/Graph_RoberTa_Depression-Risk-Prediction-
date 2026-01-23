"""
Phase 2: Hybrid Contextual Encoding
Responsibility: Model Architecture combining RoBERTa (Text) and GAT (Social/Semantic Graph).

NOTE: Uses PyTorch Geometric (PyG) as per project standard, replacing DGL due to env constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from torch_geometric.nn import GATConv

class TextEncoder(nn.Module):
    """
    Semantic Leg: RoBERTa-base backbone.
    Extracts 768-dim [CLS] token embeddings from clinical text.
    """
    def __init__(self, model_name='roberta-base', freeze_backbone=False):
        super(TextEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask):
        """
        Returns:
            pooler_output: (batch_size, 768) representation of [CLS] token.
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # We use the [CLS] token (last_hidden_state[:, 0, :]) or pooler_output
        return outputs.pooler_output

class GraphEncoder(nn.Module):
    """
    Structural Leg: Graph Attention Network (GAT).
    Aggregates information from semantic neighbors using PyG.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.1):
        super(GraphEncoder, self).__init__()
        
        # GAT Layer 1
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, concat=True)
        # GAT Layer 2 (Output)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, data_flow='target_to_source', concat=False)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features (from RoBERTa) [Num_Nodes, 768]
            edge_index: Graph connectivity [2, Num_Edges]
            edge_attr: Edge weights (optional)
        """
        # Layer 1
        x = self.dropout(x)
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)
        
        # Layer 2
        x = self.dropout(x)
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        
        return x

class GraphRobertaCL(nn.Module):
    """
    Fusion Model: Graph-RoBERTa-CL
    Integrates TextEncoder and GraphEncoder for Contrastive Learning & Classification.
    """
    def __init__(self, num_classes=2, freeze_roberta=False):
        super(GraphRobertaCL, self).__init__()
        
        # 1. Text Encoder (RoBERTa)
        self.text_encoder = TextEncoder(freeze_backbone=freeze_roberta)
        roberta_dim = 768
        
        # 2. Graph Encoder (GAT)
        # We project RoBERTa embeddings into graphical latent space
        gat_hidden = 256
        gat_out = 128
        self.graph_encoder = GraphEncoder(in_channels=roberta_dim, 
                                          hidden_channels=gat_hidden, 
                                          out_channels=gat_out)
        
        # 3. Projection Head (for Contrastive Learning)
        # Maps graph features to a normalized space for SupConLoss
        self.projection_head = nn.Sequential(
            nn.Linear(gat_out, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # 4. Classifier Head (for Final Prediction)
        self.classifier = nn.Linear(gat_out, num_classes)
        
    def forward(self, input_ids, attention_mask, edge_index, edge_attr=None):
        """
        Forward pass for both Training (Contrastive) and Inference.
        
        Args:
            input_ids, attention_mask: Tokenized text batch.
            edge_index: Connectivity of the batch (or full graph).
        
        Returns:
            logits: Classification scores [Batch, Num_Classes]
            features: Projected latent vectors [Batch, 128] for Contrastive Loss
            embeddings: Raw GAT embeddings [Batch, 128]
        """
        # Step 1: Semantic Embedding (RoBERTa)
        # h_text shape: [Batch_Size, 768]
        h_text = self.text_encoder(input_ids, attention_mask)
        
        # Step 2: Structural Aggregation (GAT)
        # Uses h_text as initial node features
        # h_graph shape: [Batch_Size, 128]
        h_graph = self.graph_encoder(h_text, edge_index, edge_attr)
        
        # Step 3: Outputs
        # Features for Contrastive Loss
        features = self.projection_head(h_graph)
        features = F.normalize(features, dim=1)
        
        # Logits for Classification
        logits = self.classifier(h_graph)
        
        return logits, features, h_graph
