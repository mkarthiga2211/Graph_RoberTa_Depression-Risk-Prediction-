"""
Exploratory Data Analysis (EDA) Script
Responsibility: Analyze Class Imbalance, Linguistic Patterns, and Interaction Metadata.

Visualizations:
1. Class Distribution (Bar Chart)
2. Post Length & Emoji Count (Dist/Box Plots)
3. Word Clouds (At-Risk vs Control)
4. Interaction Metadata (Histogram)
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
from typing import List, Dict

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw', 'kaggle')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'eda')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Premium Aesthetics Configuration
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
COLOR_PALETTE = ["#2ecc71", "#e74c3c"] # Emerald (Control) vs Alizarin (Risk)

def load_and_standardize_data() -> pd.DataFrame:
    """
    Scans the raw data directories, loads relevant CSVs, and standardizes columns.
    Target Schema: [text, label, timestamp, user_id, source]
    """
    all_data = []
    
    # 1. Suicide Watch & Mental Health Risk (Generic CSV scan)
    csv_files = glob.glob(os.path.join(DATA_RAW_DIR, '**', '*.csv'), recursive=True)
    
    if not csv_files:
        print(f"[!] No CSV files found in {DATA_RAW_DIR}. Please run 'src/data/download_datasets.py' first.")
        return pd.DataFrame()

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"[-] Processing {filename}...")
        
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # Column Normalization Logic
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Custom Mappers for specific tricky files
            if 'sentiment_tweets3' in filename.lower():
                # This dataset usually has: 'message to examine', 'label (depression result)'
                # Or: 'index', 'message', 'label'
                text_col = next((c for c in df.columns if 'message' in c or 'tweet' in c), None)
                label_col = next((c for c in df.columns if 'label' in c or 'sentiment' in c), None)
                
            elif 'mental_health_risk' in filename.lower():
                # This dataset is TABULAR (age, sleep, etc.) and has no text.
                print(f"    [!] Skipping {filename}: Dataset is TABULAR/NUMERICAL only (no text column found). Incompatible with NLP.")
                continue
                
            else:
                # Generic Discovery
                # --- Text Column ---
                # Candidates: text, body (Suicide Detection), tweet (Generic), content, post, message, statement (Mental Health Sentiment), tweet_text
                text_col = next((c for c in df.columns if c in ['text', 'body', 'tweet', 'content', 'post', 'message', 'statement', 'tweet_text']), None)
                
                # --- Label Column ---
                # Candidates: label, class (Suicide Detection), target, risk, status (Mental Health Sentiment), sentiment (Tweet Sentiment)
                label_col = next((c for c in df.columns if c in ['label', 'class', 'target', 'risk', 'status', 'sentiment']), None)
            
            if text_col and label_col:
                standardized_df = pd.DataFrame()
                standardized_df['text'] = df[text_col].astype(str)
                
                # Debug: Print raw labels to understand the dataset
                raw_labels = df[label_col].astype(str).str.lower().unique()
                print(f"    [?] Raw Labels found: {raw_labels[:10]}") 
                
                # Normalize Labels
                y = df[label_col].astype(str).str.lower().str.strip()
                
                # Default everything to -1 (Unknown) initially
                standardized_df['label'] = -1
                
                # Map Control (0) - simple exact matches or high-confidence control terms
                mask_control = y.apply(lambda x: x == '0' or x == '0.0' or 'non-suicide' in x or 'control' in x or 'neutral' in x or 'not depression' in x)
                standardized_df.loc[mask_control, 'label'] = 0
                
                # Map Risk (1) - only if we didn't already map it to control
                # Add '1', '1.0' and explicit words. Note: some datasets use strings '1'
                mask_risk = (standardized_df['label'] == -1) & y.apply(lambda x: x == '1' or x == '1.0' or 'suicide' in x or 'depress' in x or 'risk' in x)
                standardized_df.loc[mask_risk, 'label'] = 1
                
                # Drop unmapped rows (-1)
                unmapped_count = (standardized_df['label'] == -1).sum()
                if unmapped_count > 0:
                    print(f"    [!] Dropping {unmapped_count} unmapped records.")
                    standardized_df = standardized_df[standardized_df['label'] != -1]

                # --- Metadata ---
                standardized_df['timestamp'] = df['date'] if 'date' in df.columns else None
                standardized_df['user_id'] = df['user_id'] if 'user_id' in df.columns else None
                standardized_df['likes'] = df['likes'] if 'likes' in df.columns else 0
                standardized_df['source'] = filename
                
                if not standardized_df.empty:
                    all_data.append(standardized_df)
                    print(f"    [+] Loaded {len(standardized_df)} records (Class 1: {standardized_df['label'].sum()}).")
            else:
                print(f"    [!] Skipped: Could not identify text/label columns. Found: {df.columns.tolist()}")
                
        except Exception as e:
            print(f"    [!] Error loading {filename}: {e}")

    if not all_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_data, ignore_index=True)

    # SAVE STANDARDIZED DATA
    save_target = os.path.join(PROJECT_ROOT, 'data', 'processed', 'standardized_data.csv')
    os.makedirs(os.path.dirname(save_target), exist_ok=True)
    final_df.to_csv(save_target, index=False)
    print(f"\n[+] Generated Standardized Data Image (CSV) at: {save_target}")
    
    return final_df

def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balances the dataset using Random Undersampling of the majority class.
    Returns a new dataframe with equal counts of Control (0) and Risk (1).
    """
    from sklearn.utils import resample
    
    # Separate classes
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]
    
    # Check which is actually majority/minority in case of surprise
    if len(df_majority) < len(df_minority):
        df_majority, df_minority = df_minority, df_majority
        
    print(f"[-] Balancing Data...")
    print(f"    Majority count: {len(df_majority)}")
    print(f"    Minority count: {len(df_minority)}")

    if len(df_minority) == 0:
        print("[!] Warning: Minority class has 0 samples. Cannot balance. returning original data.")
        return df
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                       replace=False,    # sample without replacement
                                       n_samples=len(df_minority), # match minority class
                                       random_state=42) 
    
    # Combine back
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    print(f"    [+] Balanced count: {len(df_balanced)} ({len(df_minority)} per class)")
    
    return df_balanced

def plot_class_balance(df_original: pd.DataFrame, df_balanced: pd.DataFrame):
    """Visual 1: The Imbalance Check (Premium Style)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)
    
    # Original
    sns.countplot(x='label', data=df_original, palette=COLOR_PALETTE, ax=axes[0], alpha=0.9)
    axes[0].set_title(f'Original Distribution\n(Total: {len(df_original):,})', fontweight='bold')
    axes[0].set_xlabel('Class', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_xticklabels(['Control (0)', 'At-Risk (1)'])
    
    # Add count labels
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
    
    # Balanced
    sns.countplot(x='label', data=df_balanced, palette=COLOR_PALETTE, ax=axes[1], alpha=0.9)
    axes[1].set_title(f'Balanced Distribution (Undersampled)\n(Total: {len(df_balanced):,})', fontweight='bold')
    axes[1].set_xlabel('Class', fontweight='bold')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_xticklabels(['Control  (0)', 'At-Risk (1)'])
    
    # Add count labels
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height()):,}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'class_balance_comparison.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Class Balance Comparison to {save_path}")

def count_emojis(text):
    return emoji.emoji_count(text)

def plot_linguistic_artifacts(df: pd.DataFrame):
    """Visual 2: Post Lengths & Emojis"""
    # 1. Word Count
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x='word_count', hue='label', bins=50, kde=True, palette=COLOR_PALETTE)
    plt.title('Distribution of Post Lengths (Words)')
    plt.xlim(0, 500) # Truncate long outliers for viz
    
    # 2. Emoji Count
    df['emoji_count'] = df['text'].apply(count_emojis)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='label', y='emoji_count', data=df, palette=COLOR_PALETTE, showfliers=False)
    plt.title('Emoji Usage per Class (Outliers Hidden)')
    plt.xticks([0, 1], ['Control', 'At-Risk'])
    
    save_path = os.path.join(OUTPUT_DIR, 'linguistic_patterns.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Linguistic Artifacts plot to {save_path}")

def plot_word_clouds(df: pd.DataFrame):
    """Visual 3: Keyword Divergence"""
    stop_words = set(STOPWORDS)
    
    # Split text
    risk_text = " ".join(df[df['label'] == 1]['text'].values)
    control_text = " ".join(df[df['label'] == 0]['text'].values)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # --- Risk Cloud ---
    try:
        if not risk_text.strip() or not any(c.isalnum() for c in risk_text): 
            raise ValueError("Empty or invalid text")
        wc_risk = WordCloud(width=800, height=400, background_color='black', colormap='Reds', stopwords=stop_words).generate(risk_text)
        axes[0].imshow(wc_risk, interpolation='bilinear')
        axes[0].set_title("Most Frequent Words: AT-RISK")
    except (ValueError, IndexError):
        axes[0].text(0.5, 0.5, "No Data / Words for AT-RISK", ha='center', va='center', color='black')
        axes[0].set_title("AT-RISK (Empty)")
    axes[0].axis('off')
    
    # --- Control Cloud ---
    try:
        if not control_text.strip() or not any(c.isalnum() for c in control_text):
            raise ValueError("Empty or invalid text")
        wc_control = WordCloud(width=800, height=400, background_color='white', colormap='Blues', stopwords=stop_words).generate(control_text)
        axes[1].imshow(wc_control, interpolation='bilinear')
        axes[1].set_title("Most Frequent Words: CONTROL")
    except (ValueError, IndexError):
        axes[1].text(0.5, 0.5, "No Data / Words for CONTROL", ha='center', va='center', color='black')
        axes[1].set_title("CONTROL (Empty)")
    axes[1].axis('off')
    
    save_path = os.path.join(OUTPUT_DIR, 'wordclouds.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Word Clouds to {save_path}")

def plot_interactions(df: pd.DataFrame):
    """Visual 4: Interaction Proxy"""
    if 'likes' not in df.columns or df['likes'].sum() == 0:
        print("[!] No interaction metadata (likes/replies) found for visualization.")
        return

    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='likes', hue='label', bins=30, log_scale=(False, True), palette=COLOR_PALETTE)
    plt.title('Distribution of Interactions (Likes)')
    plt.xlabel('Likes Count')
    plt.legend(title='Class', labels=['Control', 'At-Risk'])
    
    save_path = os.path.join(OUTPUT_DIR, 'interaction_dist.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Interaction plot to {save_path}")

def plot_dataset_contribution(df: pd.DataFrame):
    """Visual 5: Dataset Contribution"""
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Count per source
    counts = df['source'].value_counts()
    
    sns.barplot(x=counts.values, y=counts.index, palette='viridis')
    plt.title('Contribution by Source Dataset', fontweight='bold')
    plt.xlabel('Number of Records', fontweight='bold')
    plt.ylabel('Source File', fontweight='bold')
    
    # Add labels
    for i, v in enumerate(counts.values):
        plt.text(v + 100, i, f"{v:,}", va='center', fontweight='bold')
        
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'dataset_contribution.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved Dataset Contribution plot to {save_path}")

def main():
    print("Starting Exploratory Data Analysis...")
    
    try:
        df = load_and_standardize_data()
        if df.empty:
            print("[!] No data loaded. Exiting.")
            return

        print(f"\n[+] Total Records Loaded: {len(df)}")
        print(f"[+] Original Class Balance: \n{df['label'].value_counts(normalize=True)}")
    except Exception as e:
        print(f"[!] Critical Error loading data: {e}")
        return

    # Visualizations on Original Data (Source Analysis)
    plot_dataset_contribution(df)

    # Balance Dataset
    df_balanced = balance_dataset(df)
    
    # ... rest of valid code ...

    # Visualizations
    plot_class_balance(df, df_balanced)
    
    print("[-] Generating plots using BALANCED data...")
    plot_linguistic_artifacts(df_balanced)
    plot_word_clouds(df_balanced)
    plot_interactions(df_balanced)
    
    print("\n[+] EDA Complete. Check 'outputs/eda/' for visualizations.")

if __name__ == "__main__":
    main()
