"""
Phase 2: Hybrid Model Architecture
Responsibility: Semantic and Graph Encoders

This module defines independent encoders:
1. RoBERTa Encoder: Fine-tunes pre-trained RoBERTa-base for text semantics.
2. GAT Encoder: Graph Attention Network for capturing social context/interaction structure.
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
# import dgl.nn as dglnn # Removed DGL

class RobertaEncoder(nn.Module):
    """
    Wraps Hugging Face RoBERTa model to extract [CLS] token embeddings.
    """
    def __init__(self, model_name='roberta-base'):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(model_name)
    
    def forward(self, input_ids, attention_mask):
        """
        Returns the pooled output (CLS token representation).
        """
        pass

class GraphEncoder(nn.Module):
    """
    Implements a Graph Attention Network (GAT) to process the User-Interaction Graph.
    Uses PyTorch Geometric.
    """
    def __init__(self, in_dim, hidden_dim, num_heads):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, concat=True)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        self.activation = nn.ELU()
        
    def forward(self, x, edge_index):
        """
        Forward pass through GAT layers.
        Args:
            x: Node feature matrix (N, in_dim)
            edge_index: Graph connectivity (2, E)
        """
        x = self.activation(self.gat1(x, edge_index))
        x = self.gat2(x, edge_index)
        return x
