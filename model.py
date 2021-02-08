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
        self.selector_network = nn.Sequential(
            nn.Linear(self.input_dim, self.selector_hdim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(self.selector_hdim),
            nn.Linear(self.selector_hdim, self.selector_hdim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # nn.BatchNorm1d(self.selector_hdim),
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
            # nn.Linear(self.predictor_hdim, self.predictor_hdim),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.predictor_hdim),
            nn.Linear(self.predictor_hdim, self.label_dim)
        )
    
    def forward(self, input, train_iteration=None):
        ## selector network
        s_probs = self.selector_network(input)

        ## sampling layer
        if self.selector_network.training:
            self.tau = np.max([1.0, np.exp(-self.temp_anneal*train_iteration)+1.0])
            # self.tau = 0.1
        else:
            self.tau = 0.
        bern_sampler = RelaxedBernoulliStraightThrough(self.tau, s_probs)
        binary_mask = bern_sampler.rsample()
        s_input = binary_mask * input

        ## predictor network
        out = self.predictor_network(s_input)
        preds = F.softmax(out, dim=1)

        return out, preds, binary_mask, s_probs

class ce_loss_with_reg():
    def __init__(self, l2, weight_decay, device, class_weight=None):
        if class_weight is not None:
            self.ce_loss = torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_weight)).to(device)
        else:
            self.ce_loss = torch.nn.CrossEntropyLoss().to(device)
        self.l2 = l2
        self.wd = weight_decay
    
    def compute_loss(self, net, preds, targets, s_probs):
        selector_l2_reg = torch.tensor(0., requires_grad=True)
        predictor_l2_reg = torch.tensor(0., requires_grad=True)
        l2_reg = torch.tensor(0., requires_grad=True)
        for name, param in net.selector_network.named_parameters():
            if 'weight' in name:
                selector_l2_reg = selector_l2_reg + self.wd*torch.pow(param, 2).sum()
        for name, param in net.predictor_network.named_parameters():
            if 'weight' in name:
                predictor_l2_reg = predictor_l2_reg + 10*self.wd*torch.pow(param, 2).sum()
        l2_reg = self.l2 * torch.mean(torch.sum(s_probs**2, dim=1))
        loss = self.ce_loss(preds, targets) + l2_reg + selector_l2_reg + predictor_l2_reg
        # loss = self.ce_loss(preds, targets) + l2_reg
        return loss