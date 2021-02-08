'''
Custom model and loss definitions.
'''

import torch
import numpy as np
from torch import nn
from pyro.distributions import RelaxedBernoulliStraightThrough

class invase_plus(nn.Module):
    def __init__(self, model_args):
        '''
        Instantiate INVASE+ network
        '''
        super(invase_plus, self).__init__()
        self.input_dim = model_args['input_dim']
        self.selector_hdim = model_args['selector_hdim']
        self.predictor_hdim = model_args['predictor_hdim']
        self.label_dim = model_args['output_dim']
        self.temp_anneal = model_args['temp_anneal']
        self.selector_network = nn.Sequential(
            nn.Linear(self.input_dim, self.selector_hdim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.selector_hdim, self.selector_hdim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.selector_hdim, self.input_dim),
            nn.Sigmoid()
        )
        self.predictor_network = nn.Sequential(
            nn.Linear(self.input_dim, self.predictor_hdim),
            nn.ReLU(),
            nn.BatchNorm1d(self.predictor_hdim),
            nn.Linear(self.predictor_hdim, self.predictor_hdim),
            nn.ReLU(),
            nn.BatchNorm1d(self.predictor_hdim),
            nn.Linear(self.predictor_hdim, self.label_dim)
        )
    
    def forward(self, input, train_iteration=None):
        '''
        Forward pass through network
        Args:
            - input: mini-batch data
            - train_iteration: iteration number, computes annealing schedule
        Returns:
            - Out: un-normalised network predictions
            - preds: normalised network predictions (probability values)
            - binary_mask: sampled selection vector
            - s_probs: selection probability
        '''
        ## selector network
        s_probs = self.selector_network(input)

        ## sampling layer
        if self.selector_network.training:
            self.tau = np.max([0.5, np.exp(-self.temp_anneal*train_iteration)+1.0])
        else:
            self.tau = 0.
        bern_sampler = RelaxedBernoulliStraightThrough(self.tau, s_probs)
        binary_mask = bern_sampler.rsample()
        s_input = binary_mask * input

        ## predictor network
        out = self.predictor_network(s_input)
        preds = nn.functional.softmax(out, dim=1)

        return out, preds, binary_mask, s_probs

class ce_loss_with_reg():
    def __init__(self, l2, weight_decay, device, class_weight=None):
        '''
        Custom loss function
        Args: 
            - l2: activation regularisation factor
            - weight_decay: weight regularisation factor
            - device: 'cpu'|'cuda'
            - class_weight: class support for weighted CE loss
        '''
        if class_weight is not None:
            self.ce_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weight)).to(device)
        else:
            self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        self.l2 = l2
        self.wd = weight_decay
    
    def compute_loss(self, net, preds, targets, s_probs):
        '''
        Compute loss term = cross entropy loss + l2 weight regularisation + l2 activation regularisation
        Args:
            - net: model
            - preds: model predictions, for CE loss
            - targets: ground truth label, for CE loss
            - s_probs: variable selection probability, for activation regularisation
        '''
        selector_l2_reg = torch.tensor(0., requires_grad=True)
        predictor_l2_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)
        ## l2 weight regularisation in selector network
        for name, param in net.selector_network.named_parameters():
            if 'weight' in name:
                selector_l2_reg = selector_l2_reg + self.wd*torch.pow(param, 2).sum()
        ## l2 weight regularisation in predictor network
        for name, param in net.predictor_network.named_parameters():
            if 'weight' in name:
                predictor_l2_reg = predictor_l2_reg + 10*self.wd*torch.pow(param, 2).sum()
        ## l2 regularisation of selection probabilities
        l2_reg = self.l2 * torch.mean(torch.sum(s_probs**2, dim=1))
        ## final loss term
        loss = self.ce_loss(preds, targets) + l2_reg + selector_l2_reg + predictor_l2_reg
        return loss