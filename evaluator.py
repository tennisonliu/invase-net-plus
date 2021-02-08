
import numpy as np
import torch

def feature_importance_score(model, x, device):
    with torch.no_grad():
        model.eval()
        x = torch.as_tensor(x, dtype=torch.float).to(device)
        _, _, _, s_probs = model(x)
        return s_probs

def predict(model, x, device):
    with torch.no_grad():
        model.eval()
        x = torch.as_tensor(x, dtype=torch.float).to(device)
        _, preds, _, _ = model(x)
    return preds