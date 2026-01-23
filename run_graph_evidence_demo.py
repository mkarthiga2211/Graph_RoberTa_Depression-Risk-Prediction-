"""
Demo Script: Generate Graph & Temporal Evidence Artifacts
Responsibility: Run the GraphEvidenceVisualizer to produce Phase B visualizations.
"""

import sys
import os

# Ensure src is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.graph_evidence import GraphEvidenceVisualizer

def main():
    print("[-] Initializing Graph Evidence Visualizer Demo...")
    visualizer = GraphEvidenceVisualizer(output_dir='outputs/figures')
    
    # 1. Graph Attention Network
    visualizer.plot_graph_attention_network()
    
    # 2. Temporal Trajectory
    visualizer.plot_temporal_risk_trajectory()
    
    print("\n[+] Demo Complete. Check 'outputs/figures/' for the generated images.")

if __name__ == "__main__":
    main()
