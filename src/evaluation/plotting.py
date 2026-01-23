"""
Phase 4: Visualization
Responsibility: Generate high-quality plots for Latent Space (t-SNE) and Performance Curves.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve

class ResultVisualizer:
    def __init__(self, output_dir='outputs/plots'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)

    def plot_tsne(self, features, labels, title="Latent Space Separation (Contrastive Learning)"):
        """
        Visualizes the embeddings using t-SNE.
        """
        print("[-] Generating t-SNE plot...")
        tsne = TSNE(n_components=2, random_state=42)
        reduced_features = tsne.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x=reduced_features[:, 0], 
            y=reduced_features[:, 1], 
            hue=labels, 
            palette={0: "green", 1: "red"},
            alpha=0.6
        )
        plt.title(title)
        plt.legend(title='Class', labels=['Control', 'At-Risk'])
        
        save_path = f"{self.output_dir}/tsne_plot.png"
        plt.savefig(save_path)
        plt.close()
        print(f"[+] t-SNE saved to {save_path}")

    def plot_pr_curve_comparison(self, models_dict, y_true):
        """
        Overlays PR Curves for multiple models.
        models_dict: { 'Graph-RoBERTa': y_probs_1, 'BERT': y_probs_2 }
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, y_probs in models_dict.items():
            precision, recall, _ = precision_recall_curve(y_true, y_probs)
            plt.plot(recall, precision, label=model_name, lw=2)
            
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve Comparison")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        save_path = f"{self.output_dir}/pr_curve_comparison.png"
        plt.savefig(save_path)
        plt.close()
        print(f"[+] PR Comparison saved to {save_path}")
