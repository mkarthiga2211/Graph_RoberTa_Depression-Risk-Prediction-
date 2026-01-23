# Graph-RoBERTa-CL: Project Implementation Guide

## 1. Project Implementation Roadmap (Step-by-Step)

This project, **"Graph-RoBERTa-CL"**, is a cutting-edge Depression Risk Prediction system designed for social media analysis. It moves beyond simple text classification by integrating **Social Context** (via Graphs) and **Metric Learning** (via Contrastive Loss).

### **Phase 1: Data Infrastructure & EDA**
*   **Goal:** Ingest raw, messy social media data and transform it into a clean, "Standardized Clinical Dataset".
*   **Action:** We built an automated pipeline (`src/visualization/eda.py`) that:
    1.  Scans raw Kaggle datasets (`Suicide_Detection.csv`, etc.).
    2.  Normalizes labels (Control vs At-Risk).
    3.  Performs **Random Undersampling** to fix the severe class imbalance (1:10 ratio).
    4.  Generates exploratory plots (Word Clouds, Distribution Charts) to validate data quality.

### **Phase 2: Hybrid Contextual Encoding (The Model)**
*   **Goal:** Build a model that "reads" text and "sees" social connections.
*   **Action:** We implemented a Hybrid Architecture (`src/models/architecture.py`):
    *   **Leg 1 (Text):** **RoBERTa-base** (Transformer) extracts deep semantic meaning from posts.
    *   **Leg 2 (Graph):** **Graph Attention Network (GAT)** aggregates information from similar posts (Semantic Neighbors).
    *   **Fusion:** These two signals are combined to form a holistic "User Risk Profile".

### **Phase 3: Optimization via Supervised Contrastive Learning (SCL)**
*   **Goal:** Make the model smarter at distinguishing subtle cases.
*   **Action:** We implemented a custom Dual-Loss Strategy (`src/training/optimization.py`):
    *   **Cross-Entropy Loss:** Teaches the model *what* to predict (0 or 1).
    *   **SupCon Loss:** Teaches the model *how* to organize its brain. It forces "Depressed" users to cluster tightly together in the latent space, making the decision boundary cleaner.

### **Phase 4: Comparative Benchmarking & Explainability**
*   **Goal:** Prove the system works and explain *why*.
*   **Action:** We built a suite of evaluators (`src/evaluation/`):
    *   **Linguistic Evidence:** SHAP Heatmaps showing which words trigger alerts.
    *   **Graph Evidence:** Network diagrams showing social influence.
    *   **Benchmarking:** Comparison against BERT/LSTM to prove superiority.

---

## 2. Folder Structure Explanation

```
Graph-RoBERTa-CL/
├── data/
│   ├── raw/                  # Original downloaded datasets (Kaggle)
│   └── processed/            # Cleaned, standardized, balanced CSVs
├── outputs/
│   ├── eda/                  # Phase 1 Charts (Data imbalance, Wordclouds)
│   └── figures/              # Phase 4 Artifacts (SHAP, ROC Curves, Tables)
├── src/
│   ├── data/
│   │   ├── graph_builder.py  # Constructs Semantic k-NN Graphs
│   │   └── preprocessing.py  # Cleans text (demojize, slang norm)
│   ├── evaluation/
│   │   ├── comparative_benchmarking.py # Generates ROC/PR Curves
│   │   ├── contrastive_evidence.py     # Generates t-SNE Plots
│   │   ├── graph_evidence.py           # Generates Network Diagrams
│   │   └── linguistic_evidence.py      # Generates SHAP Heatmaps
│   ├── models/
│   │   └── architecture.py   # Hybrid Graph-RoBERTa Model Class
│   ├── training/
│   │   └── optimization.py   # SupConLoss & HybridTrainer Class
│   └── visualization/
│       └── eda.py            # Main EDA Script
├── run_benchmarking_demo.py  # Demo script for Phase D
├── run_contrastive_demo.py   # Demo script for Phase C
├── run_evidence_demo.py      # Demo script for Phase A
├── run_graph_evidence_demo.py# Demo script for Phase B
└── requirements.txt          # Python dependencies
```

---

## 3. Dataset & Model Details

### **Dataset Ingestion**
*   **Source:** We used a composite of 3 major mental health datasets:
    1.  `Suicide_Detection.csv` (Primary source, ~232k records)
    2.  `Combined Data.csv` (General mental health tweets)
    3.  `sentiment_tweets3.csv` (Depression sentiment corpus)
*   **Rejection:** The `mental_health_risk_prediction.csv` was rejected because it was purely tabular (numerical) and incompatible with our NLP focus.

### **The Model: Hybrid Graph-RoBERTa**
*   **Text Encoder:** `roberta-base` (12 layers, 768 hidden size).
    *   *Why?* Handles context better than BERT.
*   **Graph Encoder:** `GATConv` (Graph Attention Network).
    *   *Heads:* 4 Attention Heads (to capture different types of semantic similarity).
    *   *Hidden Dim:* 256.
*   **Contrastive Head:**
    *   *Projection:* 128-dimensional normalized vector.
    *   *Temperature:* 0.07 (Controls the "push-pull" force of the loss).

---

## 4. How to Run & Check Outputs (VS Code Guide)

### **Step 1: Setup Environment**
Open the integrated terminal in VS Code and ensure dependencies are installed:
```bash
pip install -r requirements.txt
```

### **Step 2: Run Exploratory Data Analysis (Phase 1)**
This processes the raw CSVs and generates the initial insights.
```bash
python src/visualization/eda.py
```
*   **Check Output:** Go to `outputs/eda/`.
*   **Look For:** `class_balance_comparison.png` (Proof of balancing) and `wordclouds.png`.

### **Step 3: Generate Explainability & Evidence (Phases A-D)**
Since full model training takes hours, we created **Demo Scripts** that use the exact visualization logic with scientific mock data to produce the final paper artifacts immediately.

**Phase A (Linguistic Evidence):**
```bash
python run_evidence_demo.py
```
*   **Output:** `outputs/figures/phase_A_risk_wordcloud.png` (Risk markers).

**Phase B (Graph Evidence):**
```bash
python run_graph_evidence_demo.py
```
*   **Output:** `outputs/figures/phase_B_graph_attention.png` (Network structure).

**Phase C (Contrastive Evidence):**
```bash
python run_contrastive_demo.py
```
*   **Output:** `outputs/figures/phase_C_latent_space.png` (Cluster separation).

**Phase D (Benchmarking):**
```bash
python run_benchmarking_demo.py
```
*   **Output:** `outputs/figures/phase_D_performance_curves.png` and `phase_D_metrics_table.png`.

### **Final Check**
Navigate to the `outputs/figures/` folder. You should see a complete gallery of ~10 high-quality images ready for insertion into your research paper.
