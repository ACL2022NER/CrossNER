#!/usr/bin/env python3
# _*_coding:utf-8 _*_
#@Time    :2019/7/2 9:04
#@Author  :Frances 
#@FileName: latticeonlstm.py
#@Software: PyCharm

import torch
from torch import nn
import torch.autograd as autograd
from torch.nn import init
import torch.nn.functional as F
import numpy as np


def cumsoftmax(x, dim=-1):
    return torch.cumsum(F.softmax(x, dim=dim), dim=dim)


class WordLSTMCell(nn.Module):
    """A basic LSTM cell."""

    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(WordLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal(self.weight_ih.data)
        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)
        if self.use_bias:
            init.constant_(self.bias.data, val=0)

    def forward(self, input_, hx):
        '''
        :param input_: (batch_size,input_size)tensor containing input features.
        :param hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        :return: (h_1, c_1),tensors containing the next hidden and cell state.
        '''

        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        f, i, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)
        c_1 = F.sigmoid(f) * c_0 + F.sigmoid(i) * torch.tanh(g)
        f_master_gate = cumsoftmax(f)
        i_master_gate = 1 - cumsoftmax(i)
        overlap = f_master_gate * i_master_gate
        forgetgate = f * overlap + (f_master_gate - overlap)
        ingate = i * overlap + (i_master_gate - overlap)
        cy = forgetgate * c_0 + ingate * c_1

        return cy

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, use_bias=True):
        """
        Most parts are copied from torch.nn.LSTMCell.
        """
        super(MultiInputLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """
        init.orthogonal(self.weight_ih.data)
        init.orthogonal(self.alpha_weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        with torch.no_grad():
            self.weight_hh.set_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)
        with torch.no_grad():
            self.alpha_weight_hh.set_(alpha_weight_hh_data)

        if self.use_bias:
            init.constant(self.bias.data, val=0)
            init.constant(self.alpha_bias.data, val=0)

    def forward(self, input_, c_input, hx, transformed_input=None):
        '''
        Get hidden state and memory state of next time according to h(t-1) and c(t-1) and x(t)(batch=1)
        :param input_: (batch,input_size)tensor containing input features.
        :param c_input: A  list with size c_num,each element is the input ct from skip word (batch, hidden_size).
        :param hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).
        :param transformed_input:
        :return: （h_1, c_1),tensors containing the next hidden and cell state.
        '''
        h_0, c_0 = hx
        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)
        i, o, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_num = len(c_input)
        if c_num == 0:
            f = 1 - i
            c_1 = f * c_0 + i * g
            h_1 = o * torch.tanh(c_1)
        else:
            c_input_var = torch.cat(c_input, 0)
            c_input_var = c_input_var.squeeze(1)
            alpha_wi = torch.addmm(self.alpha_bias, input_, self.alpha_weight_ih).expand(c_num, self.hidden_size)
            alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)
            alpha = torch.sigmoid(alpha_wi + alpha_wh)
            alpha = torch.exp(torch.cat([i, alpha], 0))
            alpha_sum = alpha.sum(0)
            alpha = torch.div(alpha, alpha_sum)
            merge_i_c = torch.cat([g, c_input_var], 0)
            c_1 = merge_i_c * alpha
            c_1 = c_1.sum(0).unsqueeze(0)
            h_1 = o * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):
    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_dim, hidden_dim, word_drop, word_alphabet_size, word_emb_dim, pretrain_word_emb=None,
                 left2right=True, fix_word_emb=True, gpu=True, use_bias=True):
        super(LatticeLSTM, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.word_emb = nn.Embedding(word_alphabet_size, word_emb_dim)

        if pretrain_word_emb is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(pretrain_word_emb))
        else:
            self.word_emb.weight.data.copy_(torch.from_numpy(self.random_embedding(word_alphabet_size, word_emb_dim)))

        if fix_word_emb:
            self.word_emb.weight.requires_grad = False

        self.word_dropout = nn.Dropout(word_drop)
        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim)
        self.word_rnn = WordLSTMCell(word_emb_dim, hidden_dim)

        self.left2right = left2right
        if self.gpu:
            self.rnn = self.rnn.cuda()
            self.word_emb = self.word_emb.cuda()
            self.word_dropout = self.word_dropout.cuda()
            self.word_rnn = self.word_rnn.cuda()

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input, skip_input_list, data_domain, hidden=None):
        '''
        :param input: (batch_size,seq_len,input_dim)
        :param skip_input_list: [skip_input, volatile_flag]
        skip_input:（batch_size,seq_len,)
        three dimension list, for each sentence length is seq_len.
        Each element is a list of matched word id and its length.
        example: [[], [[25,13],[2,3]]],[]] 25,13 are both word id, 2 and 3 are word length .
        :param hidden: None
        :return:out_hidden, out_memory,[batch_size,seq_len,hidden_dim]
        '''
        volatile_flag = skip_input_list[-1]
        skip_input = skip_input_list[0:-1]

        batch_size = input.size(0)
        max_seq_len = input.size(1)

        if hidden:
            (hx, cx) = hidden
        else:
            hx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))
            cx = autograd.Variable(torch.zeros(batch_size, self.hidden_dim))

        out_hidden = autograd.Variable(torch.zeros(batch_size, max_seq_len, self.hidden_dim))
        out_memory = autograd.Variable(torch.zeros(batch_size, max_seq_len, self.hidden_dim))

        if self.gpu:
            hx = hx.cuda()
            cx = cx.cuda()
            out_hidden = out_hidden.cuda()
            out_hidden = out_hidden.cuda()

        p_list2 = []
        type_list2 = []

        for index in range(batch_size):
            hidden_out = []
            memory_out = []
            input_ = input[index, :, :].unsqueeze(0).transpose(1, 0) #(seq_len, 1, 50)
            hx_ = hx[index, :].unsqueeze(0) # (1, hidden_dim)
            cx_ = cx[index, :].unsqueeze(0) # (1, hidden_dim)
            skip_input_ = skip_input[index]

            if not self.left2right:
                skip_input_ = convert_forward_gaz_to_backward(skip_input_)

            seq_len = len(skip_input_)
            id_list = range(seq_len)
            if not self.left2right:
                id_list = list(reversed(id_list))

            input_c_list = init_list_of_objects(seq_len) #存放以每个字结尾的词的word_embedding
            p_list=[]
            type_list=[]
            for t in id_list:
                (hx_, cx_) = self.rnn(input_[t], input_c_list[t], (hx_, cx_)) # hx_, cx_:(1, 100)  input_[t]:(1, 50)
                hidden_out.append(hx_)
                memory_out.append(cx_)
                if skip_input_[t]:
                    matched_num = len(skip_input_[t][0])
                    word_var = autograd.Variable(torch.LongTensor(skip_input_[t][0]), volatile=volatile_flag)
                    if self.gpu:
                        word_var = word_var.cuda()
                    word_emb = self.word_emb(word_var) #（每个字结尾的词数, 50)
                    word_emb = self.word_dropout(word_emb)
                    ct = self.word_rnn(word_emb, (hx_, cx_)) # (每个字结尾的词数, 100)

                    p_list.append(ct)
                    type_list.extend(skip_input_[t][2])

                    assert (ct.size(0) == len(skip_input_[t][2]))

                    for idx in range(matched_num):
                        length = skip_input_[t][1][idx]
                        if self.left2right:
                            input_c_list[t + length - 1].append(ct[idx, :].unsqueeze(0))
                        else:
                            input_c_list[t - length + 1].append(ct[idx, :].unsqueeze(0))

            if not self.left2right:
                hidden_out = list(reversed(hidden_out))
                memory_out = list(reversed(memory_out))

            hidden_out, memory_out = torch.cat(hidden_out, 0), torch.cat(memory_out, 0) # (seq_len, dim)
            out_hidden[index, 0:seq_len, :] = hidden_out
            out_memory[index, 0:seq_len, :] = memory_out
            if  len(type_list)!=0:
                p_list2.append(torch.cat(p_list,0))
                type_list2.append(type_list)
        return out_hidden, out_memory,  p_list2, type_list2# (b, seq_len, dim)


def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())
    return list_of_objects


def convert_forward_gaz_to_backward(forward_gaz):
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)
    for idx in range(length):
        if forward_gaz[idx]:
            assert (len(forward_gaz[idx]) == 3)
            num = len(forward_gaz[idx][0])
            for idy in range(num):
                the_id = forward_gaz[idx][0][idy]
                the_length = forward_gaz[idx][1][idy]
                type = forward_gaz[idx][2][idy]
                new_pos = idx + the_length - 1
                if backward_gaz[new_pos]:
                    backward_gaz[new_pos][0].append(the_id)
                    backward_gaz[new_pos][1].append(the_length)
                    backward_gaz[new_pos][2].append(type)
                else:
                    backward_gaz[new_pos] = [[the_id], [the_length],[type]]
    return backward_gaz


