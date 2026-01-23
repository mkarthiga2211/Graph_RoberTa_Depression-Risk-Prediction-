"""
Demo Script: Generate Linguistic Evidence Artifacts
Responsibility: Run the LinguisticAnalyzer to produce Phase A visualizations (Mock/Prototype mode).

Note: In a real production run, this would be called AFTER training with the real model.
For now, we use the embedded mock data to demonstrate the visualization capabilities.
"""

import sys
import os

# Ensure src is discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.linguistic_evidence import LinguisticAnalyzer

def main():
    print("[-] Initializing Linguistic Analyzer Demo...")
    analyzer = LinguisticAnalyzer(output_dir='outputs/figures')
    
    # 1. Word Cloud
    # We pass None for shap_values/tokenizer because the method has a fallback "Mock Data" 
    # block specifically for this testing phase.
    print("[-] Generating Risk Word Cloud (Prototype)...")
    analyzer.generate_risk_wordcloud(shap_values=None, tokenizer=None)
    
    # 2. SHAP Heatmap
    # Real SHAP plots require a live Javascript environment or saved HTML.
    # The current implementation generates an HTML file.
    # We will simulate a call.
    print("[-] Generating SHAP Heatmap (Prototype - placeholders only)...")
    analyzer.plot_shap_text_heatmap(None, 0) 

    print("\n[+] Demo Complete. Check 'outputs/figures/' for the generated image.")

if __name__ == "__main__":
    main()
