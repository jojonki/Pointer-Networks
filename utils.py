import torch
from torch.autograd import Variable


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
