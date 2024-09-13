import os
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

## Classification Metric
    
def cls_metrics(pred, average:str='binary'):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

## NER Metric

import numpy as np
import evaluate


matscholar_labels = ['O', 'B-CMT' , 'I-CMT',
    'B-MAT', 'I-MAT', 'B-DSC', 'B-PRO', 'I-PRO', 'I-DSC',
    'B-SMT', 'I-SMT', 'B-APL', 'I-APL', 'B-SPL', 'I-SPL',]

SOFC_labels = [beg+end for beg in ['B-', 'I-'] for end in [str(i) for i in range(4)]] + ['O']
SOFC_slot_labels = [beg+end for beg in ['B-', 'I-'] for end in [str(i) for i in range(18)]] + ['O']

metric = evaluate.load("seqeval")

def matscholar_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[matscholar_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [matscholar_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def SOFC_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[SOFC_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [SOFC_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def SOFC_slot_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[SOFC_slot_labels[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [SOFC_slot_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

def NER_metrics(id2tag, detail=False):
    names = set([name[2:] for name in id2tag.values() if len(name)>2])

    def metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (special tokens) and convert to labels
        true_labels = [[id2tag[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        
        if detail:
            return all_metrics
        else:
            return { # Overall Score
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }

        for tag in names: # Entity-Wise Score
            if f"eval_{tag}" in all_metrics:
                result[f"f1_{tag}"] = all_metrics[f"eval_{tag}"]['f1']
    
        return result
    return metrics