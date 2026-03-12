

import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
import json
import os


def load_dataset(data_path):
    """
    Load the combined balanced dataset.
    
    Expected format: CSV with columns ['user_id', 'text', 'label']
    - user_id: unique identifier for each user
    - text: post content
    - label: 0 (control) or 1 (at-risk)
    
    Returns:
        DataFrame with user_id, text, and label columns
    """
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Validate required columns
    required_cols = ['user_id', 'text', 'label']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    print(f"✓ Loaded {len(df)} posts from {df['user_id'].nunique()} unique users")
    return df


def analyze_user_distribution(df):
    """
    Analyze user contribution patterns.
    
    Returns:
        Dictionary with distribution statistics
    """
    print("\n" + "="*60)
    print("USER DISTRIBUTION ANALYSIS")
    print("="*60)
    
    posts_per_user = df.groupby('user_id').size()
    
    stats = {
        'total_posts': len(df),
        'total_users': len(posts_per_user),
        'avg_posts_per_user': posts_per_user.mean(),
        'median_posts_per_user': posts_per_user.median(),
        'min_posts': posts_per_user.min(),
        'max_posts': posts_per_user.max(),
        'users_with_10plus_posts': (posts_per_user >= 10).sum(),
        'pct_users_with_10plus_posts': (posts_per_user >= 10).sum() / len(posts_per_user) * 100
    }
    
    print(f"Total Posts: {stats['total_posts']:,}")
    print(f"Total Unique Users: {stats['total_users']:,}")
    print(f"Average Posts/User: {stats['avg_posts_per_user']:.1f}")
    print(f"Median Posts/User: {stats['median_posts_per_user']:.0f}")
    print(f"Range: {stats['min_posts']} - {stats['max_posts']} posts")
    print(f"Users with 10+ posts: {stats['users_with_10plus_posts']:,} ({stats['pct_users_with_10plus_posts']:.1f}%)")
    print("="*60)
    
    return stats


def assign_user_labels(df):
    """
    Assign a single label to each user based on majority vote.
    
    For users with equal at-risk and control posts, conservatively
    assign at-risk label (minimize false negatives in screening).
    
    Returns:
        DataFrame with columns: user_id, user_label, post_count
    """
    print("\nAssigning user-level labels...")
    
    user_labels = []
    for user_id, group in df.groupby('user_id'):
        label_counts = group['label'].value_counts()
        
        # Majority vote (ties go to at-risk = 1)
        if len(label_counts) == 1:
            user_label = label_counts.index[0]
        elif label_counts[1] >= label_counts[0]:
            user_label = 1  # at-risk
        else:
            user_label = 0  # control
            
        user_labels.append({
            'user_id': user_id,
            'user_label': user_label,
            'post_count': len(group)
        })
    
    user_df = pd.DataFrame(user_labels)
    
    # Distribution
    label_dist = user_df['user_label'].value_counts()
    print(f"✓ User labels assigned:")
    print(f"  - Control users (0): {label_dist.get(0, 0):,}")
    print(f"  - At-risk users (1): {label_dist.get(1, 0):,}")
    
    return user_df


def create_user_level_splits(user_df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create user-level train/val/test splits with stratification.
    
    Args:
        user_df: DataFrame with user_id and user_label
        train_ratio: Proportion for training (default 0.70)
        val_ratio: Proportion for validation (default 0.15)
        test_ratio: Proportion for test (default 0.15)
        random_seed: Random seed for reproducibility (default 42)
        
    Returns:
        Dictionary with train_users, val_users, test_users (numpy arrays)
    """
    print("\n" + "="*60)
    print("USER-LEVEL SPLITTING")
    print("="*60)
    print(f"Random Seed: {random_seed}")
    print(f"Split Ratio: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    
    # First split: train vs. (val + test)
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=random_seed)
    train_idx, temp_idx = next(splitter1.split(user_df, user_df['user_label']))
    
    # Second split: val vs. test from temp
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=random_seed)
    val_idx, test_idx = next(splitter2.split(user_df.iloc[temp_idx], user_df.iloc[temp_idx]['user_label']))
    
    # Get actual indices
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]
    
    # Extract user IDs
    train_users = user_df.iloc[train_idx]['user_id'].values
    val_users = user_df.iloc[val_idx]['user_id'].values
    test_users = user_df.iloc[test_idx]['user_id'].values
    
    # Verify no overlap
    assert len(set(train_users) & set(val_users)) == 0, "User overlap between train and val!"
    assert len(set(train_users) & set(test_users)) == 0, "User overlap between train and test!"
    assert len(set(val_users) & set(test_users)) == 0, "User overlap between val and test!"
    
    print(f"✓ Train users: {len(train_users):,}")
    print(f"✓ Validation users: {len(val_users):,}")
    print(f"✓ Test users: {len(test_users):,}")
    print(f"✓ Verified: Zero user overlap between splits")
    print("="*60)
    
    return {
        'train_users': train_users,
        'val_users': val_users,
        'test_users': test_users
    }


def assign_posts_to_splits(df, user_splits):
    """
    Assign all posts to splits based on user assignment.
    
    Args:
        df: Original DataFrame with all posts
        user_splits: Dictionary with train_users, val_users, test_users
        
    Returns:
        Dictionary with train_df, val_df, test_df
    """
    print("\nAssigning posts to splits based on user assignment...")
    
    train_users_set = set(user_splits['train_users'])
    val_users_set = set(user_splits['val_users'])
    test_users_set = set(user_splits['test_users'])
    
    train_df = df[df['user_id'].isin(train_users_set)].copy()
    val_df = df[df['user_id'].isin(val_users_set)].copy()
    test_df = df[df['user_id'].isin(test_users_set)].copy()
    
    # Verify all posts assigned
    total_posts = len(train_df) + len(val_df) + len(test_df)
    assert total_posts == len(df), f"Posts lost! {total_posts} != {len(df)}"
    
    # Class distribution
    print("\nPost distribution per split:")
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        label_counts = split_df['label'].value_counts()
        print(f"  {name}: {len(split_df):,} posts "
              f"(Control: {label_counts.get(0, 0):,}, At-risk: {label_counts.get(1, 0):,})")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df
    }


def save_splits(user_splits, post_splits, output_dir='data/splits'):
    """
    Save user IDs and post indices to files.
    
    Args:
        user_splits: Dictionary with user arrays
        post_splits: Dictionary with post DataFrames
        output_dir: Directory to save split files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving splits to {output_dir}/...")
    
    # Save user IDs (for verification)
    np.save(f'{output_dir}/user_train_ids.npy', user_splits['train_users'])
    np.save(f'{output_dir}/user_val_ids.npy', user_splits['val_users'])
    np.save(f'{output_dir}/user_test_ids.npy', user_splits['test_users'])
    print(f"✓ Saved user ID arrays (3 files)")
    
    # Save post indices (for model training)
    train_indices = post_splits['train_df'].index.values
    val_indices = post_splits['val_df'].index.values
    test_indices = post_splits['test_df'].index.values
    
    np.save(f'{output_dir}/train_indices.npy', train_indices)
    np.save(f'{output_dir}/val_indices.npy', val_indices)
    np.save(f'{output_dir}/test_indices.npy', test_indices)
    print(f"✓ Saved post indices (3 files)")
    
    # Save metadata
    metadata = {
        'total_users': len(user_splits['train_users']) + len(user_splits['val_users']) + len(user_splits['test_users']),
        'train_users': len(user_splits['train_users']),
        'val_users': len(user_splits['val_users']),
        'test_users': len(user_splits['test_users']),
        'total_posts': len(train_indices) + len(val_indices) + len(test_indices),
        'train_posts': len(train_indices),
        'val_posts': len(val_indices),
        'test_posts': len(test_indices),
        'train_class_dist': post_splits['train_df']['label'].value_counts().to_dict(),
        'val_class_dist': post_splits['val_df']['label'].value_counts().to_dict(),
        'test_class_dist': post_splits['test_df']['label'].value_counts().to_dict(),
    }
    
    with open(f'{output_dir}/split_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata (split_metadata.json)")
    
    print(f"\n✓ All splits saved to {output_dir}/")


def verify_splits(output_dir='data/splits'):
    """
    Verify that splits have zero user overlap.
    """
    print("\n" + "="*60)
    print("VERIFICATION: Checking for user overlap...")
    print("="*60)
    
    train_users = np.load(f'{output_dir}/user_train_ids.npy', allow_pickle=True)
    val_users = np.load(f'{output_dir}/user_val_ids.npy', allow_pickle=True)
    test_users = np.load(f'{output_dir}/user_test_ids.npy', allow_pickle=True)
    
    train_set = set(train_users)
    val_set = set(val_users)
    test_set = set(test_users)
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    if overlap_train_val or overlap_train_test or overlap_val_test:
        print("❌ USER OVERLAP DETECTED!")
        print(f"  Train-Val overlap: {len(overlap_train_val)} users")
        print(f"  Train-Test overlap: {len(overlap_train_test)} users")
        print(f"  Val-Test overlap: {len(overlap_val_test)} users")
        return False
    else:
        print("✓ VERIFIED: Zero user overlap between all splits")
        print(f"  Train: {len(train_users):,} users")
        print(f"  Val: {len(val_users):,} users")
        print(f"  Test: {len(test_users):,} users")
        print(f"  Total: {len(train_users) + len(val_users) + len(test_users):,} users")
        print("="*60)
        return True


def main():
    parser = argparse.ArgumentParser(description='Create user-level data splits')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with user_id, text, and label columns')
    parser.add_argument('--output_dir', type=str, default='data/splits',
                        help='Output directory for split files (default: data/splits)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.70,
                        help='Training set ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Validate ratios
    assert abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Load data
    df = load_dataset(args.data_path)
    
    # Analyze distribution
    stats = analyze_user_distribution(df)
    
    # Assign user-level labels
    user_df = assign_user_labels(df)
    
    # Create splits
    user_splits = create_user_level_splits(
        user_df, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Assign posts to splits
    post_splits = assign_posts_to_splits(df, user_splits)
    
    # Save
    save_splits(user_splits, post_splits, output_dir=args.output_dir)
    
    # Verify
    verify_splits(output_dir=args.output_dir)
    
    print("\n✓ User-level data splitting complete!")
    print(f"\nFiles created in {args.output_dir}/:")
    print("  - user_train_ids.npy (training user IDs)")
    print("  - user_val_ids.npy (validation user IDs)")
    print("  - user_test_ids.npy (test user IDs)")
    print("  - train_indices.npy (training post indices)")
    print("  - val_indices.npy (validation post indices)")
    print("  - test_indices.npy (test post indices)")
    print("  - split_metadata.json (statistics and metadata)")


if __name__ == '__main__':
    main()
