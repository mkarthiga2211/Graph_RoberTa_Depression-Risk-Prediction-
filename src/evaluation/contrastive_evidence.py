"""
Phase 4 (Sub-module): Contrastive Evidence
Responsibility: Demonstrate the effectiveness of Supervised Contrastive Learning (SCL)
via Latent Space (t-SNE) and Loss Convergence plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE

class ContrastiveVisualizer:
    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Premium style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    def plot_latent_space(self, features, labels, baseline_features=None, baseline_labels=None):
        """
        Visual 1: t-SNE Projection of the Latent Space.
        
        Args:
            features: Embeddings from the Proposed Graph-RoBERTa-CL model.
            labels: Ground truth labels.
            baseline_features: (Optional) Embeddings from a Standard CE model.
        """
        print("[-] Generating t-SNE Latent Space Projection...")
        
        # Determine if we stick to 1 plot or 2
        cols = 2 if baseline_features is not None else 1
        fig, axes = plt.subplots(1, cols, figsize=(7 if cols==1 else 14, 6), dpi=300)
        
        if cols == 1:
            axes = [axes] # standardize to list
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        
        # --- Plot A: Baseline (if exists) ---
        if baseline_features is not None:
            print("    [.] Processing Baseline t-SNE...")
            reduced_baseline = tsne.fit_transform(baseline_features)
            
            sns.scatterplot(
                x=reduced_baseline[:, 0], y=reduced_baseline[:, 1],
                hue=baseline_labels, palette={0: "#2ecc71", 1: "#e74c3c"}, # Green/Red
                alpha=0.6, s=50, ax=axes[0], edgecolor="w", linewidth=0.5
            )
            axes[0].set_title("Baseline (Cross-Entropy Only)\nOverlapping Clusters", fontweight='bold')
            axes[0].set_xlabel("t-SNE Dim 1")
            axes[0].set_ylabel("t-SNE Dim 2")
            axes[0].legend(title='Class', labels=['Control', 'At-Risk'])

        # --- Plot B: Proposed (Graph-RoBERTa-CL) ---
        # If we have baseline, Proposed goes to axes[1], else axes[0]
        ax_idx = 1 if baseline_features is not None else 0
        
        print("    [.] Processing Proposed Model t-SNE...")
        reduced_proposed = tsne.fit_transform(features)
        
        sns.scatterplot(
            x=reduced_proposed[:, 0], y=reduced_proposed[:, 1],
            hue=labels, palette={0: "#2ecc71", 1: "#e74c3c"},
            alpha=0.7, s=50, ax=axes[ax_idx], edgecolor="w", linewidth=0.5
        )
        axes[ax_idx].set_title("Graph-RoBERTa-CL (Proposed)\nDistinct, Tight Clusters", fontweight='bold')
        axes[ax_idx].set_xlabel("t-SNE Dim 1")
        if cols == 2: axes[ax_idx].set_ylabel("") # clean look
        axes[ax_idx].legend(title='Class', labels=['Control', 'At-Risk'])
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "phase_C_latent_space.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] t-SNE Plot saved to {save_path}")

    def plot_loss_convergence(self, loss_history):
        """
        Visual 2: Loss Convergence Curve (SupConLoss).
        """
        print("[-] Generating Loss Convergence Curve...")
        
        epochs = np.arange(1, len(loss_history) + 1)
        
        plt.figure(figsize=(10, 6), dpi=300)
        
        # Raw Loss
        sns.lineplot(x=epochs, y=loss_history, color='orange', alpha=0.3, label='Raw Loss')
        
        # Smoothed Trend (Moving Average)
        window = max(2, len(loss_history) // 10)
        smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        # Adjust x-axis for valid convolution
        smooth_epochs = np.arange(window, len(loss_history) + 1)
        
        sns.lineplot(x=smooth_epochs, y=smoothed, color='#e74c3c', linewidth=2.5, label='Trend (Smoothed)')
        
        plt.title("Supervised Contrastive Loss Convergence", fontweight='bold')
        plt.xlabel("Training Epochs")
        plt.ylabel("SupCon Loss Value")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        save_path = os.path.join(self.output_dir, "phase_C_loss_convergence.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Loss Curve saved to {save_path}")
