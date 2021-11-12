import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertModel

class LSTMEncoder(nn.Module):
    def __init__(self, in_size, out_size, num_layers, drop_out, gpu=True,  bidirectional=True):
        super(LSTMEncoder, self).__init__()

        self.enc = nn.LSTM(input_size=in_size, hidden_size=out_size // 2, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=drop_out if num_layers > 1 else 0.)
        self.dropout = nn.Dropout(drop_out)
        if gpu:
            self.enc=self.enc.cuda()
            self.dropout=self.dropout.cuda()

    def forward(self, x, x_len: torch.Tensor):
        origin_len = x.shape[1]
        lengths, sorted_idx = x_len.sort(0, descending=True)
        x = x[sorted_idx]
        inp = pack_padded_sequence(x, lengths, batch_first=True)
        out, _ = self.enc(inp)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=origin_len)
        _, unsorted_idx = sorted_idx.sort(0)
        out = out[unsorted_idx]
        out = self.dropout(out)
        return out
