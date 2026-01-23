"""
Phase 4: Explainability
Responsibility: Unpack the 'Black Box' using SHAP and Attention Weights.
"""

import shap
import torch
import numpy as np

class ModelExplainer:
    """
    Wrapper for SHAP analysis and GAT Attention extraction.
    """
    def __init__(self):
        pass

    def explain_text_predictions(self, model, tokenizer, text_batch):
        """
        Runs SHAP Explainer on the TextEncoder component.
        Note: SHAP for Transformers is computationally expensive. Use small batches.
        """
        # Define prediction function for SHAP (takes strings -> outputs probabilities)
        def predict_wrapper(texts):
            # Tokenize
            inputs = tokenizer(texts.tolist(), return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # We assume model has a text_encoder + classifier flow
                # This is a simplification; usually we'd explain the full pipeline.
                # Here we strictly test the TextEncoder + Classifier Head (ignoring Graph for pure text SHAP)
                # Or pass dummy graph args if needed.
                
                # For Phase 4, simpler to check pure text model behavior if graph is complex
                embeddings = model.text_encoder(inputs['input_ids'], inputs['attention_mask'])
                # Project if needed, or pass to classifier
                # Assuming model has a way to classify from text embeddings solely for this check
                # or we define a proxy classifier.
                
                logits = model.classifier(model.projection_head(embeddings)) # Mock flow, adjust to architecture
                probs = torch.nn.functional.softmax(logits, dim=1)
                return probs.cpu().numpy()

        # Initialize Explainer
        # mask_token = tokenizer.mask_token
        # explainer = shap.Explainer(predict_wrapper, tokenizer)
        # shap_values = explainer(text_batch)
        
        print("[!] SHAP Explainability placeholder. Requires active model pipeline to init.")
        return None 

    def extract_graph_attention(self, model, edge_index, attention_weights_raw):
        """
        Extracts and normalizes GAT attention weights.
        Args:
            attention_weights_raw: The tuple returned by GATConv (edge_index, alpha)
        """
        # alpha shape: [num_edges, heads, 1]
        edge_index, alpha = attention_weights_raw
        
        # Average over heads
        alpha_mean = alpha.mean(dim=1).squeeze()
        
        return edge_index, alpha_mean
