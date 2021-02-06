# from torch.distributions.bernoulli import Bernoulli, 
import torch
import torch.nn.functional as F
import numpy as np


# uni_sampler = torch.distributions.uniform.Uniform(torch.zeros(4), torch.ones(4))
# uni_rdn = uni_sampler.sample()

# uni_rdn = torch.Tensor([0.5])
# print('uniformly random')
# print(uni_rdn)
# parameter = torch.nn.Parameter(torch.rand(4), requires_grad=True) # parameter [0, 1] probability
# parameter = torch.nn.Parameter(torch.Tensor([0.71]), requires_grad=True)
# print('uniformly sampled probability')
# print(parameter)
# print(torch.log(uni_rdn)-torch.log(1-uni_rdn)+torch.log(parameter)-torch.log(1-parameter))
# bern_sample = F.sigmoid(torch.log(uni_rdn)-torch.log(1-uni_rdn)+torch.log(parameter)-torch.log(1-parameter))
# print('Bernoulli Sample')
# print(bern_sample)
# # parameter = torch.nn.Parameter(torch.ones(8), requires_grad=True) # parameters to learn
# # prob = torch.sigmoid(parameter) #sigmoid to make a probability [0,1]
# # bern_sample = torch.distributions.bernoulli.Bernoulli(prob).sample() # draw a sample 
# print(parameter.grad)
# loss = torch.sum(bern_sample)
# loss.backward()
# print(parameter.grad)


# pred = 0.95
# uni_rdn = np.random.uniform(0, 1)
# z = (np.log(uni_rdn)-np.log(1-uni_rdn)+np.log(pred)-np.log(1-pred))
# print(1/(1+np.exp(-z)))

# m = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(torch.tensor([0.5]), torch.tensor([0.9]))
# print(m.sample())
# print(m.rsample())

# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable

# def sample_gumbel(shape, eps=1e-20):
#     U = torch.rand(shape).cuda()
#     return -Variable(torch.log(-torch.log(U + eps) + eps))

# def gumbel_softmax_sample(logits, temperature):
#     y = logits + sample_gumbel(logits.size())
#     return F.softmax(y / temperature, dim=-1)

# def gumbel_softmax(logits, temperature):
#     """
#     input: [*, n_class]
#     return: [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(logits, temperature)
#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     return (y_hard - y).detach() + y

# if __name__ == '__main__':
#     import math
#     print(gumbel_softmax(Variable(torch.cuda.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 20000)),     0.8).sum(dim=0))
# @Baukebrenninkmeijer
import torch 
print(torch.__version__)
import pyro
print(pyro.__version__)

for i in range(10):
    parameter = torch.nn.Parameter(torch.Tensor([0.5]), requires_grad=True) 
    m = pyro.distributions.RelaxedBernoulliStraightThrough(0.1, parameter)
    # bern_sample = m.rsample()
    bern_sample = m.sample()
    print('#### SAMPLE ####')
    print(bern_sample)
    print(parameter.grad)
    loss = torch.sum(bern_sample)
    loss.backward()
    print(parameter.grad)

# temp_anneal = 1e-4
# for iteration in range(10000):
#     tau = np.max([0.5, np.exp(-temp_anneal*iteration)])
#     print(tau)