# Graph-RoBERTa-CL Framework

## Project Overview
This project implements a depression risk prediction framework combining **RoBERTa** (text semantics), **Graph Attention Networks** (social interaction), and **Supervised Contrastive Learning** (optimization).

## Directory Structure
```
Graph-RoBERTa-CL/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── visual_proofs_shap.ipynb
│   └── visual_proofs_tsne.ipynb
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # Text cleaning, normalization
│   │   ├── graph_builder.py   # DGL Graph construction
│   │   └── temporal.py        # Temporal windowing
│   ├── models/
│   │   ├── encoders.py        # RoBERTa and GAT modules
│   │   └── hybrid.py          # Fusion architecture
│   ├── training/
│   │   ├── loss.py            # Supervised Contrastive Loss
│   │   └── trainer.py         # Training loop integration
│   └── evaluation/
│       ├── metrics.py         # F1 @ 18% FPR, etc.
│       └── explainability.py  # SHAP, t-SNE, Attention
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training:
   ```bash
   python main.py --mode train --use_cl
   ```
