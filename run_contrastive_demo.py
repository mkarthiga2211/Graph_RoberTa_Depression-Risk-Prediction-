"""
Demo Script: Generate Contrastive Evidence Artifacts
Responsibility: Run the ContrastiveVisualizer to produce Phase C visualizations.
Simulates high-dimensional embedding data to demonstrate the "clustering" effect.
"""

import sys
import os
import numpy as np
from sklearn.datasets import make_blobs

# Ensure src is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.contrastive_evidence import ContrastiveVisualizer

def main():
    print("[-] Initializing Contrastive Evidence Visualizer Demo...")
    visualizer = ContrastiveVisualizer(output_dir='outputs/figures')
    
    # 1. Simulate Latent Space Data
    # Baseline: Overlapping clusters (Cross-Entropy weakness)
    base_X, base_y = make_blobs(n_samples=300, centers=2, n_features=50, cluster_std=3.0, random_state=42)
    
    # Proposed: Tight, distinct clusters (Contrastive Learning strength)
    prop_X, prop_y = make_blobs(n_samples=300, centers=2, n_features=50, cluster_std=1.0, random_state=42)
    
    # Generate t-SNE Plot
    visualizer.plot_latent_space(
        features=prop_X, 
        labels=prop_y, 
        baseline_features=base_X, 
        baseline_labels=base_y
    )
    
    # 2. Simulate Loss Curve
    # Exponential decay with noise
    epochs = 50
    x = np.linspace(0, 5, epochs)
    loss_curve = 2.5 * np.exp(-x) + np.random.normal(0, 0.1, epochs)
    
    # Generate Loss Plot
    visualizer.plot_loss_convergence(loss_history=loss_curve)
    
    print("\n[+] Demo Complete. Check 'outputs/figures/' for the generated images.")

if __name__ == "__main__":
    main()
