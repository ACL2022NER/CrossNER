# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np
from model.charbilstm import CharBiLSTM
from model.charcnn import CharCNN
from model.latticeonlstm import LatticeLSTM
from model.lstmlayer import LSTMEncoder
from transformers.modeling_bert import BertModel


class BiLSTM(nn.Module):
    def __init__(self, data):
        super(BiLSTM, self).__init__()
        print ("build batched bi_lstm...")
        self.use_bert = data.use_bert
        self.use_bigram = data.use_bigram
        self.gpu = data.HP_gpu
        self.use_char = data.HP_use_char
        self.use_gaz = data.HP_use_gaz
        self.batch_size = data.HP_batch_size
        self.char_hidden_dim = 0
        if self.use_char:
            self.char_hidden_dim = data.HP_char_hidden_dim
            self.char_embedding_dim = data.char_emb_dim
            if data.char_features == "CNN":
                self.char_feature = CharCNN(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            elif data.char_features == "LSTM":
                self.char_feature = CharBiLSTM(data.char_alphabet.size(), self.char_embedding_dim, self.char_hidden_dim, data.HP_dropout, self.gpu)
            else:
                print ("Error char feature selection, please check parameter data.char_features (either CNN or LSTM).")
                exit(0)

        if self.use_bert:
            self.embedding_dim = 768+data.word_emb_dim
        else:
            self.embedding_dim = data.word_emb_dim
        if self.use_bert:
            self.bert_encoder = BertModel.from_pretrained(pretrained_model_name_or_path='pytorch_pretrained_model/pytorch_model.bin', config='pytorch_pretrained_model/bert_config.json')
            for p in self.bert_encoder.parameters():
                p.requires_grad = False

        self.hidden_dim = data.HP_hidden_dim
        self.drop = nn.Dropout(data.HP_dropout)
        self.droplstm = nn.Dropout(data.HP_dropout)
        self.word_embeddings = nn.Embedding(data.word_alphabet.size(),  data.word_emb_dim)
        self.biword_embeddings = nn.Embedding(data.biword_alphabet.size(), data.biword_emb_dim)
        self.bilstm_flag = data.HP_bilstm
        self.lstm_layer = data.HP_lstm_layer
        if data.pretrain_word_embedding is not None:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_word_embedding))
        else:
            self.word_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.word_alphabet.size(), self.embedding_dim)))
        if data.pretrain_biword_embedding is not None:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(data.pretrain_biword_embedding))
        else:
            self.biword_embeddings.weight.data.copy_(torch.from_numpy(self.random_embedding(data.biword_alphabet.size(), data.biword_emb_dim)))

        if self.bilstm_flag:
            lstm_hidden = data.HP_hidden_dim // 2
        else:
            lstm_hidden = data.HP_hidden_dim
        lstm_input = self.embedding_dim + self.char_hidden_dim
        if self.use_bigram:
            lstm_input += data.biword_emb_dim
        self.forward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, True, data.HP_fix_gaz_emb, self.gpu)
        if self.bilstm_flag:
            self.backward_lstm = LatticeLSTM(lstm_input, lstm_hidden, data.gaz_dropout, data.gaz_alphabet.size(), data.gaz_emb_dim, data.pretrain_gaz_embedding, False, data.HP_fix_gaz_emb, self.gpu)
        if self.use_bert:
            self.bilstm=LSTMEncoder(in_size=768, out_size=200, num_layers=1, drop_out=data.HP_dropout, gpu=self.gpu)
        else:
            self.bilstm = LSTMEncoder(in_size=50, out_size=200, num_layers=1, drop_out=data.HP_dropout, gpu=self.gpu)
        self.hidden2tag = nn.Linear(data.HP_hidden_dim, data.label_alphabet_size)
        if self.gpu:
            self.drop = self.drop.cuda()
            self.droplstm = self.droplstm.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.biword_embeddings = self.biword_embeddings.cuda()
            self.forward_lstm = self.forward_lstm.cuda()
            if self.bilstm_flag:
                self.backward_lstm = self.backward_lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            if self.use_bert:
                self.bert_encoder = self.bert_encoder.cuda()
            self.bilstm=self.bilstm.cuda()

    def forward(self, gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs, char_seq_lengths,
                char_seq_recover, mask):

        batch_size = word_inputs.size(0)
        seq_len = word_inputs.size(1)
        total_word = batch_size * seq_len

        outs= self.get_output_score(gaz_list, word_inputs, biword_inputs, word_seq_lengths, char_inputs,
                                     char_seq_lengths, char_seq_recover)

        outs = outs.view(total_word, -1)
        _, tag_seq = torch.max(outs, 1)
        tag_seq = tag_seq.view(batch_size, seq_len)
        decode_seq = mask.long() * tag_seq
        return decode_seq

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def get_output_score(self,   word_inputs, bert_seq_tensor, bert_mask,sourceVector,targetVector, batch_wordtype, alignFlag):
        lstm_out = self.get_lstm_features( word_inputs,bert_seq_tensor, bert_mask,sourceVector,targetVector, batch_wordtype, alignFlag)
        outputs = self.hidden2tag(lstm_out)
        return outputs,lstm_out

    def get_lstm_features(self, word_inputs, bert_input, bert_mask,sourceVector,targetVector, batch_wordtype, alignFlag):
        """
            input:
                word_inputs: (batch_size, sent_len)
                gaz_list:
                word_seq_lengths: list of batch_size, (batch_size,1)
                char_inputs: (batch_size*sent_len, word_length)
                char_seq_lengths: list of whole batch_size for char, (batch_size*sent_len, 1)
                char_seq_recover: variable which records the char order information, used to recover char order
            output: 
                Variable(sent_len, batch_size, hidden_dim)
        """

        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs_ = self.bert_encoder(bert_input, bert_mask, seg_id)
            word_embs= outputs_[0][:, 1:-1, :]

            if alignFlag:

                stacked_embs = word_embs.reshape([word_embs.size()[0] * word_embs.size()[1], word_embs.size()[-1]])

                source_entity_position=batch_wordtype.reshape([-1]) ==1
                source_position=source_entity_position.nonzero()
                stacked_embs[source_position, :]-=sourceVector

                target_entity_position = batch_wordtype.reshape([-1]) == 2
                target_position =target_entity_position.nonzero()
                stacked_embs[target_position, :] -= targetVector
                # stacked_embs[: , :] -= domainVector
                word_embs=stacked_embs.reshape([word_embs.size()[0], word_embs.size()[1], word_embs.size()[-1]])

        else:
            word_embs = self.word_embeddings(word_inputs)


        word_embs = self.drop(word_embs)


        seq_len=torch.sum(bert_mask, dim=-1)-2
        lstm_out = self.bilstm(word_embs, seq_len.cpu())

        return lstm_out

    def get_domainVector(self, batch_label,bert_input,bert_mask,data_domain):
        if self.use_bert:
            seg_id = torch.zeros(bert_mask.size()).long().cuda()
            outputs_ = self.bert_encoder(bert_input, bert_mask, seg_id)
            word_embs= outputs_[0][:, 1:-1, :]
            # 0:source data, 1:target data
            source_word_embs=word_embs[data_domain==0]
            target_word_embs = word_embs[data_domain == 1]
            source_label=batch_label[data_domain==0]
            target_label = batch_label[data_domain == 1]
            source_stacked_embs=source_word_embs.reshape([-1, word_embs.size()[-1]])
            target_stacked_embs = target_word_embs.reshape([-1, word_embs.size()[-1]])
            source_position=source_label.reshape([-1]) >=2
            target_position = target_label.reshape([-1]) >= 2
            source_domain_embs=source_stacked_embs[source_position]
            target_domain_embs = target_stacked_embs[target_position]
            return source_domain_embs, target_domain_embs






