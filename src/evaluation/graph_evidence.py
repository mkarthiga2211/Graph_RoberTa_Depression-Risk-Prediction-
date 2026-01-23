"""
Phase 4 (Sub-module): Graph & Temporal Evidence
Responsibility: Generate visualizations for the Structural (GAT) and Temporal aspects of the model.

Visual 1: Graph Attention Network (Edge weights = Attention)
Visual 2: Temporal User Risk Trajectory
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import os
import pandas as pd

class GraphEvidenceVisualizer:
    def __init__(self, output_dir='outputs/figures'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_graph_attention_network(self):
        """
        Visual 1: Network diagram where edge thickness represents GAT Attention Weights.
        Since we are in a demo mode without a trained GAT model, we simulate the structure.
        """
        print("[-] Generating Graph Attention Visualization (Mock)...")
        
        # 1. Create a simulated graph (Cluster of depression posts vs control posts)
        G = nx.Graph()
        
        # Central Node (Simulating a user's new post being classified)
        target_node = "Target_Post"
        G.add_node(target_node, color='red', size=800)
        
        # Neighbors (Semantically similar posts found by k-NN)
        neighbors = [
            ("Similar_Post_1", 0.95, 'red'),   # High sim, Red (Risk)
            ("Similar_Post_2", 0.85, 'red'),   # High sim, Red (Risk)
            ("Neutral_Ctx_1", 0.20, 'green'),  # Low sim, Green (Control)
            ("Neutral_Ctx_2", 0.10, 'green'),  # Low sim, Green (Control)
            ("Similar_Post_3", 0.92, 'red')    # High sim, Red (Risk)
        ]
        
        edge_colors = []
        edge_weights = []
        node_colors = ['red'] # Start with target
        labels = {target_node: "TARGET"}
        
        for name, attn_weight, color in neighbors:
            G.add_edge(target_node, name, weight=attn_weight)
            edge_weights.append(attn_weight * 5) # Scale for visual thickness
            node_colors.append(color)
            # Color edge based on weight intensity (Darker = Higher Attention)
            edge_colors.append((0, 0, 0, attn_weight)) 
            labels[name] = "" # Don't label neighbors, just show nodes
            
        # 2. Plot
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw Nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=[800 if n==target_node else 300 for n in G.nodes()], alpha=0.9)
        
        # Draw Edges (Thickness = Attention)
        nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color='gray')
        
        # Labels
        nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_weight='bold', font_size=10)
        
        plt.title("Graph Attention Weights (Model Focus)", fontsize=15)
        plt.axis('off')
        
        # Legend (Manual)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='At-Risk Content', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Control Content', markersize=10),
            Line2D([0], [0], color='gray', lw=4, label='High Attention (Strong Semantic Link)'),
            Line2D([0], [0], color='gray', lw=1, label='Low Attention (Weak Link)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        save_path = os.path.join(self.output_dir, "phase_B_graph_attention.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Graph Attention plot saved to {save_path}")

    def plot_temporal_risk_trajectory(self):
        """
        Visual 2: Longitudinal line graph of a user's risk score over time.
        """
        print("[-] Generating Temporal Risk Trajectory (Mock)...")
        
        # 1. Simulate Data (12 Months)
        dates = pd.date_range(start="2024-01-01", periods=12, freq='M')
        # Risk scores (0-1). Simulating a worsening trajectory then a spyke vs a recovery
        risk_scores = [0.2, 0.25, 0.3, 0.45, 0.6, 0.85, 0.9, 0.88, 0.7, 0.6, 0.5, 0.4]
        
        data = pd.DataFrame({'Date': dates, 'Risk_Score': risk_scores})
        
        # 2. Plot
        plt.figure(figsize=(12, 6))
        
        # Gradient Fill or Line
        sns.lineplot(data=data, x='Date', y='Risk_Score', marker='o', linewidth=3, color='#e74c3c')
        
        # Threshold Line
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Clinical Threshold (0.5)')
        
        # Highlight Spikes (Interaction Events)
        spike_date = dates[5] # Index 5 is 0.85
        plt.annotate('Critical Interaction Event\n(Risk Peak)', 
                     xy=(spike_date, 0.85), 
                     xytext=(dates[2], 0.95),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.title("Temporal Risk Trajectory (User ID: <Anonymized>)", fontsize=14)
        plt.ylabel("Depression Risk Probability")
        plt.xlabel("Timeline (12 Months)")
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend()
        
        save_path = os.path.join(self.output_dir, "phase_B_temporal_trajectory.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Temporal Trajectory saved to {save_path}")

