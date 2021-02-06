import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from pyro.distributions import RelaxedBernoulliStraightThrough

class invase_model(nn.Module):
    def __init__(self, model_args):
        super(invase_model, self).__init__()
        self.input_dim = model_args['input_dim']
        self.selector_hdim = model_args['selector_hdim']
        self.predictor_hdim = model_args['predictor_hdim']
        self.label_dim = model_args['output_dim']
        self.temp_anneal = model_args['temp_anneal']
        self.fc1 = nn.Linear(self.input_dim, self.selector_hdim)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(self.selector_hdim, self.selector_hdim)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(self.selector_hdim, self.input_dim)
        self.fc4 = nn.Linear(self.input_dim, self.predictor_hdim)
        self.bn4 = nn.BatchNorm1d(self.predictor_hdim)
        self.fc5 = nn.Linear(self.predictor_hdim, self.predictor_hdim)
        self.bn5 = nn.BatchNorm1d(self.predictor_hdim)
        self.fc6 = nn.Linear(self.predictor_hdim, self.label_dim)
    
    def forward(self, input, train_iteration=None):
        ## selector network
        x = F.relu(self.fc1(input))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        s_probs = torch.sigmoid(self.fc3(x))

        ## sampling layer
        if train_iteration:
            tau = np.max([0.1, np.exp(-self.temp_anneal*train_iteration)-0.5])
        else:
            tau = 0.
        bern_sampler = RelaxedBernoulliStraightThrough(tau, s_probs)
        binary_mask = bern_sampler.rsample()
        s_input = binary_mask * input.detach()

        ## predictor network
        x = F.relu(self.fc4(s_input))
        x = self.bn4(x)
        x = F.relu(self.fc5(x))
        x = self.bn5(x)
        out = self.fc6(x)
        preds = F.softmax(out, dim=1)

        return out, preds, s_probs

class ce_loss_with_reg():
    def __init__(self, l2, device):
        self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        self.l2 = l2
    
    def compute_loss(self, preds, targets, s_probs):
        l2_reg = self.l2 * torch.norm(s_probs, p=2)
        loss = self.ce_loss(preds, targets) + l2_reg
        return loss