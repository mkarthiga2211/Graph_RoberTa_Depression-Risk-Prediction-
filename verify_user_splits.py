"""
Verify User-Level Data Splits

"""

import numpy as np
import json
import argparse
import os


def load_splits(split_dir):
    """Load all split files."""
    print(f"Loading splits from {split_dir}/...")
    
    files = {
        'train_users': f'{split_dir}/user_train_ids.npy',
        'val_users': f'{split_dir}/user_val_ids.npy',
        'test_users': f'{split_dir}/user_test_ids.npy',
        'metadata': f'{split_dir}/split_metadata.json'
    }
    
    # Check all files exist
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
    
    splits = {
        'train_users': np.load(files['train_users'], allow_pickle=True),
        'val_users': np.load(files['val_users'], allow_pickle=True),
        'test_users': np.load(files['test_users'], allow_pickle=True),
    }
    
    with open(files['metadata'], 'r') as f:
        splits['metadata'] = json.load(f)
    
    print("✓ All files loaded successfully\n")
    return splits


def verify_user_overlap(splits):
    """Verify zero user overlap between splits."""
    print("="*70)
    print("VERIFICATION: USER OVERLAP CHECK")
    print("="*70)
    
    train_set = set(splits['train_users'])
    val_set = set(splits['val_users'])
    test_set = set(splits['test_users'])
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    print(f"Training users: {len(train_set):,}")
    print(f"Validation users: {len(val_set):,}")
    print(f"Test users: {len(test_set):,}")
    print()
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("❌ FAILED: User overlap detected!")
        if overlap_train_val:
            print(f"  ❌ Train-Val overlap: {len(overlap_train_val)} users")
        if overlap_train_test:
            print(f"  ❌ Train-Test overlap: {len(overlap_train_test)} users")
        if overlap_val_test:
            print(f"  ❌ Val-Test overlap: {len(overlap_val_test)} users")
        print("="*70)
        return False
    else:
        print("✅ PASSED: Zero user overlap between all splits")
        print("  ✓ Complete user isolation guaranteed")
        print("="*70)
        return True


def main():
    parser = argparse.ArgumentParser(description='Verify user-level data splits')
    parser.add_argument('--split_dir', type=str, default='data/splits',
                        help='Directory containing split files')
    
    args = parser.parse_args()
    
    print("\nUSER-LEVEL DATA SPLIT VERIFICATION")
    print("="*70)
    
    splits = load_splits(args.split_dir)
    success = verify_user_overlap(splits)
    
    return success


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
