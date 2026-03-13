"""
Stage 2: Within-Split Post-Level Rebalancing

This script performs the second stage of the two-stage balancing protocol:
- Loads splits created by create_user_level_splits.py
- Applies random undersampling WITHIN each split to achieve exact 50/50 balance
- Preserves user-level isolation (no posts move between splits)
- Uses independent random seeds for each split

Author: Modified for Graph-RoBERTa-CL Framework
"""

import numpy as np
import pandas as pd
import argparse
import json
import os
from collections import Counter


def load_splits_and_data(data_path, splits_dir='data/splits'):
    """
    Load the original data and split indices.
    
    Args:
        data_path: Path to the full balanced dataset CSV
        splits_dir: Directory containing split files from Stage 1
        
    Returns:
        Tuple of (full_df, train_indices, val_indices, test_indices)
    """
    print("=" * 70)
    print("LOADING STAGE 1 SPLITS")
    print("=" * 70)
    
    # Load full dataset
    print(f"Loading full dataset from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} total posts")
    
    # Load split indices
    train_indices = np.load(f'{splits_dir}/train_indices.npy')
    val_indices = np.load(f'{splits_dir}/val_indices.npy')
    test_indices = np.load(f'{splits_dir}/test_indices.npy')
    
    print(f"\nStage 1 split sizes (BEFORE Stage 2 rebalancing):")
    print(f"  Train: {len(train_indices):,} posts")
    print(f"  Val:   {len(val_indices):,} posts")
    print(f"  Test:  {len(test_indices):,} posts")
    
    return df, train_indices, val_indices, test_indices


def analyze_split_distribution(df, indices, split_name):
    """
    Analyze class distribution in a split.
    
    Args:
        df: Full dataframe
        indices: Indices for this split
        split_name: Name for display (e.g., 'Train')
        
    Returns:
        Dictionary with distribution statistics
    """
    split_df = df.iloc[indices]
    label_counts = split_df['label'].value_counts().sort_index()
    
    control_count = label_counts.get(0, 0)
    at_risk_count = label_counts.get(1, 0)
    total = len(split_df)
    
    stats = {
        'split_name': split_name,
        'total_posts': total,
        'control_posts': control_count,
        'at_risk_posts': at_risk_count,
        'control_pct': control_count / total * 100 if total > 0 else 0,
        'at_risk_pct': at_risk_count / total * 100 if total > 0 else 0,
        'imbalance': abs(control_count - at_risk_count)
    }
    
    return stats


def apply_within_split_balancing(df, indices, random_seed):
    """
    Apply random undersampling within a single split to achieve exact 50/50 balance.
    
    Args:
        df: Full dataframe
        indices: Indices for this split
        random_seed: Random seed for reproducibility
        
    Returns:
        Numpy array of balanced indices
    """
    # Get split dataframe
    split_df = df.iloc[indices].copy()
    split_df['original_index'] = indices  # Track original indices
    
    # Separate by class
    control_df = split_df[split_df['label'] == 0]
    at_risk_df = split_df[split_df['label'] == 1]
    
    # Find minority class size
    min_count = min(len(control_df), len(at_risk_df))
    
    # Undersample both classes to minority size
    np.random.seed(random_seed)
    
    control_sampled = control_df.sample(n=min_count, random_state=random_seed)
    at_risk_sampled = at_risk_df.sample(n=min_count, random_state=random_seed)
    
    # Combine and extract original indices
    balanced_df = pd.concat([control_sampled, at_risk_sampled])
    balanced_indices = balanced_df['original_index'].values
    
    # Verify balance
    final_labels = df.iloc[balanced_indices]['label'].value_counts()
    assert final_labels[0] == final_labels[1], "Balancing failed!"
    
    return balanced_indices


def save_stage2_splits(train_indices, val_indices, test_indices, 
                       train_stats_before, val_stats_before, test_stats_before,
                       train_stats_after, val_stats_after, test_stats_after,
                       output_dir='data/splits'):
    """
    Save Stage 2 balanced splits and metadata.
    
    Args:
        *_indices: Balanced indices for each split
        *_stats_before: Pre-balancing statistics
        *_stats_after: Post-balancing statistics
        output_dir: Directory to save files
    """
    print("\n" + "=" * 70)
    print("SAVING STAGE 2 BALANCED SPLITS")
    print("=" * 70)
    
    # Save balanced indices
    np.save(f'{output_dir}/train_indices_stage2.npy', train_indices)
    np.save(f'{output_dir}/val_indices_stage2.npy', val_indices)
    np.save(f'{output_dir}/test_indices_stage2.npy', test_indices)
    print(f"✓ Saved balanced indices (3 files)")
    
    # Create comprehensive metadata
    metadata = {
        'stage': 'Stage 2 - Within-Split Post-Level Rebalancing',
        'random_seeds': {
            'train': 123,
            'val': 456,
            'test': 789
        },
        'before_stage2': {
            'train': train_stats_before,
            'val': val_stats_before,
            'test': test_stats_before
        },
        'after_stage2': {
            'train': train_stats_after,
            'val': val_stats_after,
            'test': test_stats_after
        },
        'changes': {
            'train': {
                'posts_removed': train_stats_before['total_posts'] - train_stats_after['total_posts'],
                'control_removed': train_stats_before['control_posts'] - train_stats_after['control_posts'],
                'at_risk_removed': train_stats_before['at_risk_posts'] - train_stats_after['at_risk_posts']
            },
            'val': {
                'posts_removed': val_stats_before['total_posts'] - val_stats_after['total_posts'],
                'control_removed': val_stats_before['control_posts'] - val_stats_after['control_posts'],
                'at_risk_removed': val_stats_before['at_risk_posts'] - val_stats_after['at_risk_posts']
            },
            'test': {
                'posts_removed': test_stats_before['total_posts'] - test_stats_after['total_posts'],
                'control_removed': test_stats_before['control_posts'] - test_stats_after['control_posts'],
                'at_risk_removed': test_stats_before['at_risk_posts'] - test_stats_after['at_risk_posts']
            }
        }
    }
    
    with open(f'{output_dir}/stage2_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved Stage 2 metadata (stage2_metadata.json)")
    
    print(f"\n✓ All Stage 2 files saved to {output_dir}/")


def print_comparison_table(stats_before, stats_after, split_name):
    """Print before/after comparison for a split."""
    print(f"\n{split_name} Split:")
    print(f"  BEFORE Stage 2: {stats_before['total_posts']:,} posts "
          f"({stats_before['control_posts']:,} control + {stats_before['at_risk_posts']:,} at-risk)")
    print(f"  AFTER Stage 2:  {stats_after['total_posts']:,} posts "
          f"({stats_after['control_posts']:,} control + {stats_after['at_risk_posts']:,} at-risk)")
    print(f"  Posts removed:  {stats_before['total_posts'] - stats_after['total_posts']:,} "
          f"({stats_before['control_posts'] - stats_after['control_posts']:,} control, "
          f"{stats_before['at_risk_posts'] - stats_after['at_risk_posts']:,} at-risk)")


def verify_user_isolation(df, train_indices, val_indices, test_indices):
    """
    Verify that user-level isolation is preserved after Stage 2.
    
    Args:
        df: Full dataframe
        *_indices: Balanced indices for each split
        
    Returns:
        Boolean indicating whether isolation is preserved
    """
    print("\n" + "=" * 70)
    print("VERIFYING USER-LEVEL ISOLATION")
    print("=" * 70)
    
    train_users = set(df.iloc[train_indices]['user_id'].unique())
    val_users = set(df.iloc[val_indices]['user_id'].unique())
    test_users = set(df.iloc[test_indices]['user_id'].unique())
    
    overlap_train_val = train_users & val_users
    overlap_train_test = train_users & test_users
    overlap_val_test = val_users & test_users
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("❌ USER OVERLAP DETECTED!")
        print(f"  Train-Val: {len(overlap_train_val)} users")
        print(f"  Train-Test: {len(overlap_train_test)} users")
        print(f"  Val-Test: {len(overlap_val_test)} users")
        return False
    else:
        print("✓ VERIFIED: Zero user overlap between all splits")
        print(f"  Train: {len(train_users):,} unique users")
        print(f"  Val: {len(val_users):,} unique users")
        print(f"  Test: {len(test_users):,} unique users")
        print("=" * 70)
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Apply Stage 2 within-split post-level rebalancing'
    )
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to full balanced dataset CSV (same as used in Stage 1)')
    parser.add_argument('--splits_dir', type=str, default='data/splits',
                        help='Directory containing Stage 1 split files (default: data/splits)')
    parser.add_argument('--train_seed', type=int, default=123,
                        help='Random seed for training split rebalancing (default: 123)')
    parser.add_argument('--val_seed', type=int, default=456,
                        help='Random seed for validation split rebalancing (default: 456)')
    parser.add_argument('--test_seed', type=int, default=789,
                        help='Random seed for test split rebalancing (default: 789)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("STAGE 2: WITHIN-SPLIT POST-LEVEL REBALANCING")
    print("=" * 70)
    print(f"Random Seeds: Train={args.train_seed}, Val={args.val_seed}, Test={args.test_seed}")
    print("=" * 70)
    
    # Load data and Stage 1 splits
    df, train_idx_stage1, val_idx_stage1, test_idx_stage1 = load_splits_and_data(
        args.data_path, args.splits_dir
    )
    
    # Analyze BEFORE Stage 2
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION BEFORE STAGE 2")
    print("=" * 70)
    
    train_stats_before = analyze_split_distribution(df, train_idx_stage1, 'Train')
    val_stats_before = analyze_split_distribution(df, val_idx_stage1, 'Val')
    test_stats_before = analyze_split_distribution(df, test_idx_stage1, 'Test')
    
    for stats in [train_stats_before, val_stats_before, test_stats_before]:
        print(f"\n{stats['split_name']}: {stats['total_posts']:,} posts")
        print(f"  Control: {stats['control_posts']:,} ({stats['control_pct']:.2f}%)")
        print(f"  At-risk: {stats['at_risk_posts']:,} ({stats['at_risk_pct']:.2f}%)")
        print(f"  Imbalance: {stats['imbalance']:,} posts")
    
    # Apply Stage 2 balancing
    print("\n" + "=" * 70)
    print("APPLYING WITHIN-SPLIT BALANCING")
    print("=" * 70)
    
    print(f"\nBalancing Train split (seed={args.train_seed})...")
    train_idx_stage2 = apply_within_split_balancing(df, train_idx_stage1, args.train_seed)
    print(f"✓ Train: {len(train_idx_stage1):,} → {len(train_idx_stage2):,} posts")
    
    print(f"\nBalancing Val split (seed={args.val_seed})...")
    val_idx_stage2 = apply_within_split_balancing(df, val_idx_stage1, args.val_seed)
    print(f"✓ Val: {len(val_idx_stage1):,} → {len(val_idx_stage2):,} posts")
    
    print(f"\nBalancing Test split (seed={args.test_seed})...")
    test_idx_stage2 = apply_within_split_balancing(df, test_idx_stage1, args.test_seed)
    print(f"✓ Test: {len(test_idx_stage1):,} → {len(test_idx_stage2):,} posts")
    
    # Analyze AFTER Stage 2
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION AFTER STAGE 2")
    print("=" * 70)
    
    train_stats_after = analyze_split_distribution(df, train_idx_stage2, 'Train')
    val_stats_after = analyze_split_distribution(df, val_idx_stage2, 'Val')
    test_stats_after = analyze_split_distribution(df, test_idx_stage2, 'Test')
    
    for stats in [train_stats_after, val_stats_after, test_stats_after]:
        print(f"\n{stats['split_name']}: {stats['total_posts']:,} posts")
        print(f"  Control: {stats['control_posts']:,} ({stats['control_pct']:.2f}%)")
        print(f"  At-risk: {stats['at_risk_posts']:,} ({stats['at_risk_pct']:.2f}%)")
        print(f"  Perfect Balance: {'YES ✓' if stats['imbalance'] == 0 else 'NO ✗'}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("BEFORE vs AFTER COMPARISON")
    print("=" * 70)
    print_comparison_table(train_stats_before, train_stats_after, 'TRAIN')
    print_comparison_table(val_stats_before, val_stats_after, 'VAL')
    print_comparison_table(test_stats_before, test_stats_after, 'TEST')
    
    # Verify user isolation
    isolation_preserved = verify_user_isolation(df, train_idx_stage2, val_idx_stage2, test_idx_stage2)
    
    if not isolation_preserved:
        print("\n❌ ERROR: User-level isolation was compromised!")
        print("This should never happen. Check the code.")
        return
    
    # Save
    save_stage2_splits(
        train_idx_stage2, val_idx_stage2, test_idx_stage2,
        train_stats_before, val_stats_before, test_stats_before,
        train_stats_after, val_stats_after, test_stats_after,
        output_dir=args.splits_dir
    )
    
    print("\n" + "=" * 70)
    print("✓ STAGE 2 COMPLETE")
    print("=" * 70)
    print("\nFiles created:")
    print(f"  - {args.splits_dir}/train_indices_stage2.npy")
    print(f"  - {args.splits_dir}/val_indices_stage2.npy")
    print(f"  - {args.splits_dir}/test_indices_stage2.npy")
    print(f"  - {args.splits_dir}/stage2_metadata.json")
    print("\nThese files should now be used for model training instead of")
    print("the original Stage 1 indices.")
    print("=" * 70)


if __name__ == '__main__':
    main()
