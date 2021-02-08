'''
Functions for evaluating trained models.
'''

import torch
import numpy as np

def feature_importance_score(model, x, device):
    '''
    Return feature importance/selection probability
    Args:
        - model: instantiated model
        - x: data
        - device: 'cpu'|'cuda'
    Returns:
        - selection probability
    '''
    with torch.no_grad():
        model.eval()
        x = torch.as_tensor(x, dtype=torch.float).to(device)
        _, _, _, s_probs = model(x)
        return s_probs

def predict(model, x, device):
    '''
    Return predictions
    Args:
        - model: instantiated model
        - x: data
        - device: 'cpu'|'cuda'
    Returns:
        - model predictions
    '''
    with torch.no_grad():
        model.eval()
        x = torch.as_tensor(x, dtype=torch.float).to(device)
        _, preds, _, _ = model(x)
    return preds