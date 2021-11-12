# -*- coding: utf-8 -*-


import torch.nn as nn
from model.bilstm import BiLSTM
from model.crf import CRF
import torch
import torch.nn.functional as F


class BiLSTM_CRF(nn.Module):
    def __init__(self, data):
        super(BiLSTM_CRF, self).__init__()
        print("build batched lstm_crf...")
        self.gpu = data.HP_gpu
        label_size = data.label_alphabet_size
        data.label_alphabet_size += 2
        self.lstm = BiLSTM(data)
        self.crf = CRF(label_size, self.gpu)
        self.batch_size = data.HP_batch_size
        self.use_bert = data.use_bert
        self.align=data.alignment
        self.sep=data.separation
        self.sepLoss_weight = data.separation_loss_weight



    def forward(self, word_inputs, batch_wordtype, mask, bert_seq_tensor, bert_mask, sourceVector,targetVector):
        outs,lstm_out = self.lstm.get_output_score(word_inputs, bert_seq_tensor, bert_mask, sourceVector, targetVector, batch_wordtype, self.align)
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return tag_seq

    def neg_log_likelihood_loss(self, word_inputs,  batch_label,batch_wordtype, mask, data_domain,bert_seq_tensor, bert_mask,sourceVector,targetVector):

        outs,lstm_out = self.lstm.get_output_score(word_inputs,bert_seq_tensor, bert_mask,sourceVector,targetVector, batch_wordtype, self.align)

        sepLoss=self.separateLoss(lstm_out,data_domain, batch_wordtype)


        crf_loss = self.crf.neg_log_likelihood_loss(outs, mask, batch_label)
        if self.sep:
            total_loss = crf_loss + self.sepLoss_weight * sepLoss
        else:
            total_loss = crf_loss
        scores, tag_seq = self.crf._viterbi_decode(outs, mask)
        return total_loss, tag_seq, (crf_loss, self.sepLoss_weight * sepLoss)


    def domainVector(self, batch_label,bert_seq_tensor,bert_mask,data_domain):
        source_domain_embs, target_domain_embs = self.lstm.get_domainVector(batch_label,bert_seq_tensor,bert_mask,data_domain)
        return source_domain_embs, target_domain_embs

    def separateLoss(self, embs,data_domain, batch_wordtype):
        source_embs = embs[data_domain == 0]
        target_embs = embs[data_domain == 1]
        source_token_type =batch_wordtype[data_domain == 0]
        target_token_type = batch_wordtype[data_domain == 1]

        source_stacked_embs = source_embs.reshape([-1, embs.size()[-1]])
        target_stacked_embs = target_embs.reshape([-1, embs.size()[-1]])

        # 1:source，2:target， 3: others， 0:padding
        source_entity_position = source_token_type.reshape([-1]) == 1
        source_Noentity_position = source_token_type.reshape([-1]) != 1
        target_entity_position = target_token_type.reshape([-1]) == 2
        target_Noentity_position = target_token_type.reshape([-1]) != 2

        source_entity_embs=source_stacked_embs[source_entity_position]
        source_Noentity_embs = source_stacked_embs[source_Noentity_position]
        target_entity_embs=target_stacked_embs[target_entity_position]
        target_Noentity_embs = target_stacked_embs[target_Noentity_position]

        # Averge pooling
        sourcePro = torch.mean(source_entity_embs, 0)
        sourceNoPro = torch.mean(source_Noentity_embs, 0)
        targetPro = torch.mean(target_entity_embs, 0)
        targetNoPro = torch.mean( target_Noentity_embs, 0)


        loss1 = 1 / torch.norm(sourcePro - sourceNoPro, p=2)
        loss2 = 1 / torch.norm(targetPro - targetNoPro, p=2)

        if torch.isnan(loss2).item() ==True:
            loss= loss1 * self.batch_size
        elif torch.isnan(loss1).item() == True:
            loss = loss2 * self.batch_size
        else:
            loss = (loss1 + loss2) * self.batch_size
        return loss






