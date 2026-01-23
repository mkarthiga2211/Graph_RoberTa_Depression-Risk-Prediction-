"""
Phase 4: Comparative Benchmarking
Responsibility: Calculate rigorous classification metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report

class ClinicalEvaluator:
    """
    Evaluator specialized for Clinical Risk Prediction metrics.
    Focus: False Positive Reduction and High Recall for At-Risk Class.
    """
    def __init__(self):
        pass

    def calculate_standard_metrics(self, y_true, y_pred, y_probs=None):
        """
        Calculates core metrics: Accuracy, P, R, F1, AUC.
        """
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        
        metrics = {
            'Accuracy': acc,
            'Precision': p,
            'Recall': r,
            'F1-Score': f1
        }
        
        if y_probs is not None:
            try:
                # Handle binary case
                if len(y_probs.shape) > 1:
                    probs = y_probs[:, 1]
                else:
                    probs = y_probs
                metrics['AUC'] = roc_auc_score(y_true, probs)
            except ValueError:
                metrics['AUC'] = 0.0 # Handle case with single class in batch
                
        # FPR Calculation (False Positives / Negatives)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['False Positive Rate'] = fpr
        
        return metrics

    def calculate_false_positive_reduction(self, baseline_fpr, model_fpr):
        """
        Quantifies reduction in False Alarms.
        Target: >18% reduction.
        """
        reduction = (baseline_fpr - model_fpr) / baseline_fpr
        print(f"[-] False Positive Reduction: {reduction:.2%}")
        return reduction

    def generate_classification_report(self, y_true, y_pred, target_names=['Control', 'At-Risk']):
        """
        Returns classification report formatted for output.
        """
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("[-] Classification Report:\n")
        print(report)
        return report
