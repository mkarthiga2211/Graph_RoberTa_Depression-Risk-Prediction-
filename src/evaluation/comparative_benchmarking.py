"""
Phase 4 (Sub-module): Comparative Benchmarking
Responsibility: Generate rigorous performance comparisons against baselines (Phase D).

Visual 1: Multi-Model ROC & PR Curves
Visual 2: Ablation Study (Incremental Value Add)
Visual 3: Training Dynamics (Loss/F1)
Table 1: Final Metrics Table (LaTeX ready)
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, accuracy_score, precision_recall_fscore_support, roc_auc_score

class BenchmarkingVisualizer:
    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Publication Style
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        self.colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'] # Green (Prop), Blue, Red, Purple

    def plot_performance_curves(self, model_results):
        """
        Visual 1: ROC and PR Curves overlaid for all models.
        args: model_results = {'ModelName': {'y_true': [], 'y_scores': []}}
        """
        print("[-] Generating Comparative Performance Curves...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        # --- Plot A: ROC Curve ---
        for i, (name, data) in enumerate(model_results.items()):
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_scores'])
            roc_auc = auc(fpr, tpr)
            
            # Highlight Proposed Model
            lw = 3 if 'Graph-RoBERTa' in name else 2
            ls = '-' if 'Graph-RoBERTa' in name else '--'
            
            axes[0].plot(fpr, tpr, lw=lw, linestyle=ls, label=f'{name} (AUC = {roc_auc:.2f})')
            
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve Comparison', fontweight='bold')
        axes[0].legend(loc="lower right")

        # --- Plot B: Precision-Recall Curve ---
        for i, (name, data) in enumerate(model_results.items()):
            precision, recall, _ = precision_recall_curve(data['y_true'], data['y_scores'])
            ap = average_precision_score(data['y_true'], data['y_scores'])
            
            lw = 3 if 'Graph-RoBERTa' in name else 2
            ls = '-' if 'Graph-RoBERTa' in name else '--'
            
            axes[1].plot(recall, precision, lw=lw, linestyle=ls, label=f'{name} (AP = {ap:.2f})')
        
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve Comparison', fontweight='bold')
        axes[1].legend(loc="lower left")
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "phase_D_performance_curves.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Performance Curves saved to {save_path}")

    def plot_ablation_study(self, ablation_scores):
        """
        Visual 2: Bar chart showing incremental F1 gains.
        args: ablation_scores = {'RoBERTa': 0.72, '+ Graph': 0.79, ...}
        """
        print("[-] Generating Ablation Study Chart...")
        
        plt.figure(figsize=(8, 6), dpi=300)
        
        # Create Bar Plot
        names = list(ablation_scores.keys())
        values = list(ablation_scores.values())
        
        bars = sns.barplot(x=names, y=values, palette="Blues_d")
        
        # Add labels
        for i, p in enumerate(bars.patches):
            height = p.get_height()
            # Calculate gain from previous
            gain_text = ""
            if i > 0:
                prev = values[i-1]
                gain = ((height - prev) / prev) * 100
                gain_text = f" (+{gain:.1f}%)"
            
            plt.text(p.get_x() + p.get_width() / 2., 
                     height + 0.01, 
                     f"{height:.2f}{gain_text}", 
                     ha="center", fontweight='bold')

        plt.title('Ablation Study: Impact of Components (F1-Score)', fontweight='bold')
        plt.ylabel('Macro F1-Score')
        plt.ylim(0, 1.0)
        
        save_path = os.path.join(self.output_dir, "phase_D_ablation_study.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Ablation Study saved to {save_path}")

    def plot_training_dynamics(self, history_dict):
        """
        Visual 3: Loss and Accuracy curves over epochs.
        history_dict = {'Proposed': {'loss': [], 'acc': []}, 'Baseline': ...}
        """
        print("[-] Generating Training Dynamics...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        
        epochs = None
        
        for name, metrics in history_dict.items():
            if epochs is None: epochs = range(1, len(metrics['loss']) + 1)
            
            # Loss
            axes[0].plot(epochs, metrics['loss'], label=name, lw=2)
            # Accuracy
            axes[1].plot(epochs, metrics['acc'], label=name, lw=2)
            
        axes[0].set_title('Training Loss Convergence', fontweight='bold')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].legend()
        axes[0].grid(True, linestyle=':')
        
        axes[1].set_title('Validation Accuracy Evolution', fontweight='bold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].legend()
        axes[1].grid(True, linestyle=':')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "phase_D_training_dynamics.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"[+] Training Dynamics saved to {save_path}")

    def generate_metrics_table(self, model_results, threshold=0.5):
        """
        Generates and Saves Metrics Table suitable for LaTeX.
        """
        print("[-] Generating Final Metrics Table...")
        records = []
        
        for name, data in model_results.items():
            y_true = np.array(data['y_true'])
            y_scores = np.array(data['y_scores'])
            y_pred = (y_scores >= threshold).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            roc_auc = roc_auc_score(y_true, y_scores)
            
            records.append({
                'Model': name,
                'Accuracy': f"{acc:.3f}",
                'Precision': f"{p:.3f}",
                'Recall': f"{r:.3f}",
                'F1-Score': f"{f1:.3f}",
                'AUC': f"{roc_auc:.3f}"
            })
            
        df = pd.DataFrame(records)
        print("\n", df)
        
        # Save CSV
        csv_path = os.path.join(self.output_dir, "phase_D_metrics_table.csv")
        df.to_csv(csv_path, index=False)
        
        # Save LaTeX
        latex_path = os.path.join(self.output_dir, "phase_D_metrics_table.tex")
        with open(latex_path, 'w') as f:
            f.write(df.to_latex(index=False, caption="Comparative Performance Metrics", label="tab:metrics"))
        
        # Save Image (PNG)
        print("    [.] Rendering Table as Image...")
        plt.figure(figsize=(10, 3), dpi=300)
        ax = plt.gca()
        ax.axis('off')
        
        # Create table: cellText, colLabels, cellLoc, loc
        table = plt.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.5) # Scale columns and rows (height)
        
        # Color headers
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#40466e')
            elif i > 0:
                if i % 2 == 0:
                    cell.set_facecolor('#f2f2f2') # Zebra striping
        
        image_path = os.path.join(self.output_dir, "phase_D_metrics_table.png")
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
            
        print(f"[+] Metrics Table saved to CSV, LaTeX, and PNG in {self.output_dir}")
        return df
