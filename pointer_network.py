# Reference: https://github.com/guacomolia/ptr_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var


class PointerNetwork(nn.Module):
    def __init__(self, input_size, emb_size, weight_size, seq_len, answer_seq_len, hidden_size=512):
        super(PointerNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.answer_seq_len = answer_seq_len
        self.weight_size = weight_size
        self.seq_len = seq_len
        self.emb_size = emb_size

        self.emb = nn.Embedding(input_size, emb_size)  # embed inputs
        self.enc = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.dec = nn.LSTMCell(emb_size, hidden_size)
        self.W1 = nn.Linear(hidden_size, weight_size) # blending encoder
        self.W2 = nn.Linear(hidden_size, weight_size) # blending decoder
        self.vt = nn.Linear(weight_size, 1) # scaling sum of enc and dec by v.T

        self.tanh = nn.Tanh()

    def forward(self, input):
        batch_size = input.size(0)
        input = self.emb(input) # (bs, L, embd_size)

        # Encoding
        encoder_states, hc = self.enc(input) # encoder_state: (bs, L, H)
        encoder_states = encoder_states.transpose(1, 0) # (L, bs, H)

        # Decoding states initialization
        decoder_input = to_var(torch.Tensor(batch_size, self.emb_size).zero_()) # (bs, embd_size)
        hidden = to_var(torch.randn([batch_size, self.hidden_size]))            # (bs, h)
        cell_state = encoder_states[-1]                                         # (bs, h)

        probs = []
        # Decoding
        for i in range(self.answer_seq_len): # range(M)
            hidden, cell_state = self.dec(decoder_input, (hidden, cell_state)) # (bs, h), (bs, h)

            # Compute blended representation at each decoder time step
            blend1 = self.W1(encoder_states)          # (L, bs, W)
            blend2 = self.W2(hidden)                  # (bs, W)
            blend_sum = self.tanh(blend1 + blend2)    # (L, bs, W)
            out = self.vt(blend_sum).squeeze()        # (L, bs)
            out = F.log_softmax(out.t().contiguous()) # (bs, L)
            probs.append(out)

        probs = torch.stack(probs, dim=1)           # (bs, M, L)

        return probs
