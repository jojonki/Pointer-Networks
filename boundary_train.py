# Reference: https://github.com/guacomolia/ptr_net
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import generate_data
from utils import to_var
import random
from pointer_network import PointerNetwork

total_size = 10000
weight_size = 256                           # W
emb_size = 32
batch_size = 250                            # B
n_batches = total_size // batch_size        # NB
answer_seq_len = 2                          # M = 2
n_epochs = 100                              # NE

dataset, starts, ends = generate_data.generate_set_seq(total_size)
targets = np.vstack((starts, ends)).T  # [total_size, M]
dataset = np.array(dataset)# [total_size, L]

input_seq_len = dataset.shape[1]
inp_size = 11

# Convert to torch tensors
input = to_var(torch.LongTensor(dataset))     # [total_size, L]
targets = to_var(torch.LongTensor(targets))   # [total_size, 2]

train_batches = input.view(n_batches, batch_size, input_seq_len) # [NB, B, L]
targets = targets.view(n_batches, batch_size, answer_seq_len) # [NB, B, 2]

# from pointer_network import PointerNetwork
def train(n_epochs, model, train_batches, targets):
    model.train()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(n_epochs + 1):
        for i in range(len(train_batches)):
            input = train_batches[i] # [B, L, 2]
            target = targets[i] # [B, M]

            optimizer.zero_grad()

            L = input.data.shape[1]
            probs = model(input) # (L, N, M)
            probs = probs.view(L, -1).t().contiguous() # (N*M, L)
            target = target.view(-1) # (N*M)
            loss = F.nll_loss(probs, target)
            loss.backward()
            optimizer.step()

        pick = np.random.randint(0, batch_size)
        if epoch % 2 == 0:
            print('epoch: {}\t\t -- loss: {:.5f}'.format(epoch, loss.data[0]))
            print("trained ", probs.max(1)[1].data[pick], probs.max(1)[1].data[2*pick],
                  "target : ", target.data[pick], target.data[2*pick])

def predict(model, data):
    outputs = model(data) # (L, N, M)
    outputs = outputs.transpose(1, 0) # (N, L, M)
    outputs = outputs.transpose(2, 1) # (N, M, L)
    _v, indices = torch.max(outputs, 2) # indices: (N, M)
    return indices

def test(model):
    # Predictions
    test_id = random.randint(0, batch_size-1)
    test_data = train_batches[0] # (N, L)
    test_targets = targets[0] # (N, M)
    indices = predict(model, test_data) # (N, M)
    for i in range(len(indices)):
        print('-----')
        print('test', [v for v in test_data[i].data])
        print('label', [v for v in test_targets[i].data])
        print('pred', [v for v in indices[i].data])
        if i>20: break

model = PointerNetwork(inp_size, emb_size, weight_size, batch_size, input_seq_len, answer_seq_len)
if torch.cuda.is_available():
    model.cuda()
train(10, model, train_batches, targets)
test(model)
