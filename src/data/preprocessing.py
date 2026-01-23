"""
Phase 1: NLP Pipeline
Responsibility: Text Cleaning & Normalization for Clinical/Mental Health Analysis.

Features:
- Emoji-to-Text conversion (captures sentiment).
- Slang Normalization (internet lingo to full text).
- Anonymization (users/URLs).
- Temporal Windowing (if user/time metadata is available).
"""

import re
import pandas as pd
import emoji
import numpy as np

class ClinicalTextPreprocessor:
    def __init__(self):
        # Common internet slang map
        self.slang_map = {
            "idk": "i do not know",
            "imo": "in my opinion",
            "imho": "in my humble opinion",
            "tbh": "to be honest",
            "u": "you",
            "ur": "your",
            "r": "are",
            "btw": "by the way",
            "omg": "oh my god",
            "thx": "thanks",
            "plz": "please",
            "pls": "please",
            "dm": "direct message",
            "bc": "because",
            "b/c": "because",
            "fyi": "for your information",
            "irl": "in real life",
            "smh": "shaking my head",
            "nvm": "nevermind",
            "ttyl": "talk to you later"
        }
        
    def _normalize_slang(self, text):
        words = text.split()
        return " ".join([self.slang_map.get(w.lower(), w) for w in words])

    def clean_text(self, text: str) -> str:
        """
        Applies clinical text cleaning pipeline.
        """
        if not isinstance(text, str):
            return ""
            
        # 1. Emoji to Text (e.g. 😭 -> :loudly_crying_face:)
        text = emoji.demojize(text)
        
        # 2. Anonymization (Regex)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
        # Remove User Mentions
        text = re.sub(r'@\w+', '<USER>', text)
        
        # 3. Slang Normalization
        text = self._normalize_slang(text)
        
        # 4. Basic Cleanup
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def create_temporal_windows(self, df: pd.DataFrame, user_col='user_id', time_col='timestamp', text_col='text', window_size='7D') -> pd.DataFrame:
        """
        Groups posts by user and time window.
        
        NOTE: If user_col or time_col are missing/empty (common in Kaggle datasets), 
        this returns the original dataframe slightly modifed, treating each row as a window.
        """
        # Check availability
        if user_col not in df.columns or time_col not in df.columns:
            print(f"[!] Warning: Missing '{user_col}' or '{time_col}'. Skipping temporal grouping.")
            return df
            
        # Check validity (non-nulls)
        if df[user_col].isnull().all() or df[time_col].isnull().all():
             print(f"[!] Warning: Columns '{user_col}'/'{time_col}' are fully empty. Skipping temporal grouping.")
             return df

        # Convert to datetime
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        
        # Grouping Logic
        print(f"[-] Grouping texts by {window_size} windows...")
        grouped = df.sort_values(time_col).groupby([user_col, pd.Grouper(key=time_col, freq=window_size)])
        
        aggregated_data = []
        for (user, window), group in grouped:
            combined_text = " ".join(group[text_col].tolist())
            # Majority vote/logic for label
            label = group['label'].mode()[0] if 'label' in group.columns else None
            
            aggregated_data.append({
                'user_id': user,
                'timestamp': window,
                'text': combined_text,
                'label': label,
                'post_count': len(group)
            })
            
        return pd.DataFrame(aggregated_data)
