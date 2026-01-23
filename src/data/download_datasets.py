"""
Data Ingestion Script for Graph-RoBERTa-CL
Responsibility: Automate downloading of Kaggle datasets and structure eRisk placeholders.

Prerequisites:
1.  'kaggle' library installed (pip install kaggle).
2.  Kaggle API Token (kaggle.json) placed in C:/Users/<User>/.kaggle/kaggle.json or ~/.kaggle/kaggle.json.
"""

import os
import sys
import shutil
import zipfile

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
KAGGLE_DIR = os.path.join(DATA_RAW_DIR, 'kaggle')
ERISK_DIR = os.path.join(DATA_RAW_DIR, 'erisk')

# Dataset Mappings (Kaggle ID -> Local Folder Name)
KAGGLE_DATASETS = {
    "sahideseker/mental-health-risk-prediction-dataset": "mental_health_risk",
    "gargmanas/sentimental-analysis-for-tweets": "tweet_sentiment",
    "suchintikasarkar/sentiment-analysis-for-mental-health": "mental_health_sentiment",
    "nikhileswarkomati/suicide-watch": "suicide_watch"
}

def check_kaggle_setup():
    """Checks if kaggle library is installed and credentials exist."""
    print("[-] Checking Kaggle environment...")
    
    try:
        import kaggle
    except ImportError:
        print("[!] Error: 'kaggle' library not found. Please install it via 'pip install kaggle'.")
        sys.exit(1)

    # Check for credentials in default locations
    # On Windows: C:\Users\<Current User>\.kaggle\kaggle.json
    kaggle_config_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json_path = os.path.join(kaggle_config_dir, 'kaggle.json')

    if not os.path.exists(kaggle_json_path):
        print(f"[!] Warning: 'kaggle.json' not found at {kaggle_json_path}.")
        print("    Please download your API key from Kaggle -> Settings -> Create New API Token.")
        print("    Place the file in the directory above to enable automated downloads.")
        return False
    
    print("[+] Kaggle credentials found.")
    return True

def download_kaggle_datasets():
    """Authenticates and downloads specific datasets using Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    print("\n[-] Authenticating with Kaggle...")
    api = KaggleApi()
    try:
        api.authenticate()
        print("[+] Authentication successful.")
    except Exception as e:
        print(f"[!] Authentication failed: {e}")
        return

    if not os.path.exists(KAGGLE_DIR):
        os.makedirs(KAGGLE_DIR)

    for dataset_id, folder_name in KAGGLE_DATASETS.items():
        target_path = os.path.join(KAGGLE_DIR, folder_name)
        
        if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:
            print(f"[*] Skipping {dataset_id} (already exists at {folder_name}).")
            continue
            
        print(f"[-] Downloading {dataset_id} into '{folder_name}'...")
        try:
            # Download and unzip
            api.dataset_download_files(dataset_id, path=target_path, unzip=True)
            print(f"[+] Successfully downloaded {dataset_id}.")
        except Exception as e:
            print(f"[!] Failed to download {dataset_id}: {e}")

def setup_erisk_placeholder():
    """Creates directory structure for manual eRisk data ingestion."""
    print("\n[-] Setting up eRisk dataset structure...")
    
    if not os.path.exists(ERISK_DIR):
        os.makedirs(ERISK_DIR)
        
    subdirs = ['depression', 'control']
    for sd in subdirs:
        path = os.path.join(ERISK_DIR, sd)
        if not os.path.exists(path):
            os.makedirs(path)
            
    print(f"[!] ACTION REQUIRED: eRisk data cannot be downloaded automatically.")
    print(f"    Please manually copy your XML/Text files into: {ERISK_DIR}")
    print(f"    Structure expected:")
    print(f"      - {ERISK_DIR}/depression/ (Positive samples)")
    print(f"      - {ERISK_DIR}/control/    (Negative samples)")

def verify_downloads():
    """Lists downloaded files to confirm success."""
    print("\n[-] Verifying Data Directory State...")
    
    for root, dirs, files in os.walk(DATA_RAW_DIR):
        level = root.replace(DATA_RAW_DIR, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def main():
    if check_kaggle_setup():
        download_kaggle_datasets()
    else:
        print("[!] Skipping automated Kaggle downloads due to missing credentials.")
    
    setup_erisk_placeholder()
    verify_downloads()

if __name__ == "__main__":
    main()
