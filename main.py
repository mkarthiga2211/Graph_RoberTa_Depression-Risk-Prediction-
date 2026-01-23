"""
Graph-RoBERTa-CL: Main Entry Point
Responsibility: Orchestrate the full Training and Evaluation Pipeline.

Note: For generating the Research Paper Artifacts (Visualizations), 
please use the dedicated demo scripts:
- run_evidence_demo.py (Phase A)
- run_graph_evidence_demo.py (Phase B)
- run_contrastive_demo.py (Phase C)
- run_benchmarking_demo.py (Phase D)
"""

import argparse
import sys
import torch
import os

# Ensure correct imports based on actual file structure
from src.data.preprocessing import ClinicalTextPreprocessor
from src.data.graph_builder import InteractionGraphBuilder
from src.models.architecture import GraphRobertaCL
from src.training.optimization import HybridTrainer

def main():
    parser = argparse.ArgumentParser(description="Graph-RoBERTa-CL Training & Evaluation")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'demo'], default='demo')
    parser.add_argument('--features', type=str, default='all', help='Feature set to use')
    args = parser.parse_args()

    print(f"[-] Initializing Graph-RoBERTa-CL Framework in [{args.mode}] mode...")

    # 1. Initialize Components
    preprocessor = ClinicalTextPreprocessor()
    graph_builder = InteractionGraphBuilder()
    
    # 2. Model Setup
    print("[-] Building Hybrid Graph-RoBERTa Model...")
    model = GraphRobertaCL(num_classes=2)
    
    if args.mode == 'demo':
        print("\n[i] INFO: To generate the research artifacts, please run the dedicated scripts:")
        print("    1. python run_evidence_demo.py")
        print("    2. python run_graph_evidence_demo.py")
        print("    3. python run_contrastive_demo.py")
        print("    4. python run_benchmarking_demo.py")
        
    elif args.mode == 'train':
        print("[-] Starting Full Training Pipeline...")
        # Placeholder for full training loop integration
        # trainer = HybridTrainer(model, optimizer=torch.optim.Adam(model.parameters(), lr=1e-5))
        print("[!] Full training requires ~4-8 hours on GPU. Please execute on HPC cluster.")
        
    elif args.mode == 'evaluate':
        print("[-] Evaluating Model Checkpoints...")
        
if __name__ == "__main__":
    main()
