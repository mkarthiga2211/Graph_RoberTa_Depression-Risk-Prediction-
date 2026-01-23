"""
Phase 3: Optimization & Loss Functions
Responsibility: Implement Supervised Contrastive Loss (SupCon) and the Hybrid Training Loop.

Theory:
1. SupConLoss pulls 'At-Risk' samples closer to other 'At-Risk' samples in the embedding space.
2. CrossEntropyLoss optimizes the decision boundary for classification.
3. The HybridTrainer balances these two objectives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss: https://arxiv.org/abs/2004.11362
    It handles multiple positive pairs per anchor (unlike SimCLR).
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, feature_dim] Normalized embeddings.
            labels: [batch_size] Class labels.
        """
        device = features.device
        batch_size = features.shape[0]
        
        # 1. Compute Similarity Matrix (Cosine Similarity since features are normalized)
        # Result: [batch, batch]
        similarity_matrix = torch.matmul(features, features.T)
        
        # 2. Configure Mask for Positive Pairs (Same Label)
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num labels must match num features')
            
        # mask[i, j] = 1 if label[i] == label[j]
        mask = torch.eq(labels, labels.T).float().to(device)
        
        # 3. Mask out self-contrast (the diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # 4. Compute Logits
        # Scale by temperature
        logits = similarity_matrix / self.temperature
        
        # Numerical stability: shift max to 0
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        
        # 5. Compute Log-Probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
        
        # 6. Compute Mean Log-Likelihood over Positive Pairs
        # mean_log_prob_pos: Sum of log probs for positive pairs / Number of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
        # Loss is negative of that likelihood
        loss = -mean_log_prob_pos
        return loss.mean()

class HybridTrainer:
    """
    Manages the dual-objective training: Classification + Contrastive Learning.
    """
    def __init__(self, model, optimizer, device='cuda', lambda_weight=0.5):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_weight = lambda_weight # Weight for CrossEntropy (1 - lambda for SupCon)
        
        # Loss Functions
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_scl = SupConLoss(temperature=0.07)
        
        # Hooks for Visualization (t-SNE)
        self.stored_embeddings = []
        self.stored_labels = []

    def train_step(self, batch_data):
        """
        Performs one forward/backward pass.
        Args:
            batch_data: PyG Data object or tuple containing (input_ids, mask, edge_index, y)
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Unpack PyG batch or Custom batch
        # Assuming batch_data is a dictionary for simplicity/safety
        input_ids = batch_data['input_ids'].to(self.device)
        attention_mask = batch_data['attention_mask'].to(self.device)
        edge_index = batch_data['edge_index'].to(self.device)
        labels = batch_data['y'].to(self.device)
        
        # Forward Pass
        logits, features, _ = self.model(input_ids, attention_mask, edge_index)
        
        # Calculate Losses
        loss_ce = self.criterion_ce(logits, labels)
        loss_scl = self.criterion_scl(features, labels)
        
        # Combined Loss
        # lambda * CE + (1-lambda) * SCL
        total_loss = (self.lambda_weight * loss_ce) + ((1 - self.lambda_weight) * loss_scl)
        
        # Backprop
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'loss_ce': loss_ce.item(),
            'loss_scl': loss_scl.item()
        }

    def validate(self, dataloader, capture_embeddings=False):
        """
        Evaluates the model and optionally captures embeddings for t-SNE.
        """
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        if capture_embeddings:
            self.stored_embeddings = []
            self.stored_labels = []
            
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                labels = batch['y'].to(self.device)
                
                logits, features, raw_embeds = self.model(input_ids, attention_mask, edge_index)
                
                # Metrics
                loss_ce = self.criterion_ce(logits, labels)
                total_loss += loss_ce.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Store for t-SNE (Phase 4)
                if capture_embeddings:
                    self.stored_embeddings.append(features.cpu().numpy())
                    self.stored_labels.append(labels.cpu().numpy())
                    
        # Calculate F1
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        
        return {
            'val_loss': total_loss / len(dataloader),
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
