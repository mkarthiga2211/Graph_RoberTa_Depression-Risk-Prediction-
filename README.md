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
## Data Preparation

### Two-Stage Balancing Protocol

This framework uses a rigorous two-stage balancing protocol:

**Stage 1: Pre-Split Post-Level Random Undersampling**
```bash
# Assumes you have a balanced dataset (232,074 posts)
python create_user_level_splits.py --data_path data/balanced_dataset.csv --seed 42
```

**Stage 2: Post-Split Within-Split Post-Level Rebalancing**
```bash
python apply_stage2_balancing.py --data_path data/balanced_dataset.csv --splits_dir data/splits
```

This will create:
- `train_indices_stage2.npy` - Balanced training indices (use this for training)
- `val_indices_stage2.npy` - Balanced validation indices
- `test_indices_stage2.npy` - Balanced test indices
- `stage2_metadata.json` - Statistics and metadata
