import os
import random
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, path):
    torch.save(state, path)

def load_checkpoint(path, model, optimizer=None, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    return checkpoint

def compute_metrics(y_true, y_pred_probs, thresh=0.5):
    if np.isnan(y_pred_probs).any():
        return {"per_label_auc": [], "mean_auc": 0.5, "f1_macro": 0.0}

    aucs = []
    for i in range(y_true.shape[1]):
        try:
            if len(np.unique(y_true[:, i])) < 2:
                auc = float('nan')
            else:
                auc = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
        except ValueError:
            auc = float('nan')
        aucs.append(auc)
    
    mean_auc = np.nanmean(np.array(aucs))
    if np.isnan(mean_auc): mean_auc = 0.0

    y_pred = (y_pred_probs >= thresh).astype(int)
    f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='macro')
    
    return {"per_label_auc": aucs, "mean_auc": mean_auc, "f1_macro": f1}

def get_adj_matrix(csv_path, labels_list, threshold=0.05):
    """Tạo ma trận kề cho GCN từ xác suất đồng xuất hiện."""
    df = pd.read_csv(csv_path)
    num_classes = len(labels_list)
    
    data_labels = []
    for labels_str in df['Finding Labels']:
        row_vector = np.zeros(num_classes)
        for i, label in enumerate(labels_list):
            if label in labels_str: row_vector[i] = 1
        data_labels.append(row_vector)
    
    data_labels = np.array(data_labels)
    adj = np.dot(data_labels.T, data_labels) 

    nums = np.diag(adj) 
    with np.errstate(divide='ignore', invalid='ignore'):
        adj = adj / nums[:, None] 
    
    adj[np.isnan(adj)] = 0.0
    adj[adj < threshold] = 0
    
    row_sum = adj.sum(axis=1)
    row_sum[row_sum == 0] = 1 
    adj = adj / row_sum[:, None]
    
    return torch.FloatTensor(adj)