#!/usr/bin/env python3
# _*_coding:utf-8 _*_


import os
import sys
os.chdir(sys.path[0])
import warnings
import time
import argparse
import random
import torch
import gc
import pickle
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import datetime
import csv

from utils.metric import get_ner_fmeasure
from model.bilstmcrf import BiLSTM_CRF as SeqModel
from utils.data import Data
from utils.logger import init_logger

parser = argparse.ArgumentParser(description='Tuning with BERT-BiLSTM-CRF')
parser.add_argument('--source', default="fc",help="source domain")
parser.add_argument('--target', default="cars",help="target domain")
parser.add_argument('--alignment', default=True, type=bool)
parser.add_argument('--separation', default=True, type=bool)
parser.add_argument('--emb', default='data/ctb.50d.vec', type=str, help='Embedding')
parser.add_argument('--log_file', default="logs/log.txt", help="dir of log")
parser.add_argument('--separation_loss_weight', default='0.5', type=float, help='parameter of loss weight of separation')
parser.add_argument('--use_bert', action='store_true', default=True)
parser.add_argument('--status', choices=['train', 'test', 'decode', 'predict'], default='train',help='update algorithm')
nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
parser.add_argument('--savedir', default="output/model", help="dir path to save model")
parser.add_argument('--savedset', help='fir of saved data setting', default="output/dset/dset1")
parser.add_argument('--decodeOutput', default="")
parser.add_argument('--seg', default=True, type=bool, help="add segment information as a feature")
parser.add_argument('--extendalphabet', default="True")
parser.add_argument('--predictInput', default="")
parser.add_argument('--predictOutput', default="")
parser.add_argument('--load_model', default="", help="dir of trained model for prediction")
parser.add_argument('--load_dset', default="", help="dir of  data setting for prediction")
parser.add_argument('--gpu', default=torch.cuda.is_available(), type=bool, help="available to gpu")
args = parser.parse_args()
status = args.status.lower()

logger = init_logger(args.log_file)

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


def data_initialization(data, gaz_file, train_file, dev_file, test_file, pro_file):
    '''
    Build vocabulary and generate lattice
    :param data:
    :param gaz_file:
    :param train_file:
    :param dev_file:
    :param test_file:
    :return:
    '''
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    data.load_pro(pro_file)
    return data


def predict_check(pred_variable, gold_variable, mask_variable):
    '''
    Check data
    :param pred_variable: (batch_size, sent_len),pred tag result, in numpy format
    :param gold_variable: (batch_size, sent_len),gold  result variable
    :param mask_variable: mask_variable (batch_size, sent_len),mask variable
    :return:
    '''
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    right_token = np.sum(overlaped * mask)
    total_token = mask.sum()
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover):
    '''
    Recover the result with label
    :param pred_variable: (batch_size, sent_len),pred tag result
    :param gold_variable: (batch_size, sent_len),gold result variable
    :param mask_variable: (batch_size, sent_len),mask variable
    :param label_alphabet:
    :param word_recover:
    :return:
    '''
    pred_variable = pred_variable[word_recover]
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    seq_len = gold_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    gold_tag = gold_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    gold_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        assert (len(pred) == len(gold))
        pred_label.append(pred)
        gold_label.append(gold)
    return pred_label, gold_label


def recover_without_label(pred_variable, mask_variable, label_alphabet, word_recover):
    '''
    Recover the predict without label
    :param pred_variable: (batch_size, sent_len), pred tag result
    :param mask_variable: (batch_size, sent_len), mask variable
    :param label_alphabet:
    :param word_recover:
    :return:
    '''

    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
        logger.info("p:", pred, pred_tag.tolist())
        pred_label.append(pred)
    return pred_label


def save_data_setting(data, save_file):
    '''
    Save data setting when train
    :param data:
    :param save_file:
    :return:
    '''
    logger.info(save_file)
    with open(save_file, 'wb') as fp:
        pickle.dump(data, fp)
    # logger.info("Data setting saved to file: ", save_file)


def load_data_setting(save_file):
    '''
    Load data setting
    :param save_file: path of file to be loaded
    :return:
    '''
    with open(save_file, 'rb') as fp:
        data = pickle.load(fp)
    logger.info("Data setting loaded from file: ", save_file)
    data.show_data_summary()
    return data


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    '''
    Learning rate decay
    :param optimizer:
    :param epoch:
    :param decay_rate:
    :param init_lr:
    :return:
    '''
    lr = init_lr * ((1 - decay_rate) ** epoch)
    # logger.info(" Learning rate is setted as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def evaluate(data, model, name, source_domain_vector, target_domain_vector):
    '''
    Evaluate when train/test/decode
    :param data:
    :param model:
    :param name:
    :return:
    '''
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        logger.info("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = 10
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_wordtype, mask, data_domain,bert_seq_tensor, bert_mask = batchify_with_label(
            instance, data.HP_gpu, True)
        tag_seq = model(batch_word,batch_wordtype, mask,bert_seq_tensor, bert_mask, source_domain_vector, target_domain_vector)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results




def batchify_with_label(input_batch_list, gpu, volatile_flag=False):
    '''
    Padding the sequence with label for train/test/decode
    :param input_batch_list:
    :param gpu:
    :param volatile_flag:
    :return:
    '''
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    labels = [sent[4] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    data_domain = [sent[5] for sent in input_batch_list]  # 每一条的数据领域
    ### bert tokens
    bert_ids = [sent[6] for sent in input_batch_list]
    ##字的类型（源领域、目标领域实体的token）还是非实体token
    wordType_ids = [sent[7] for sent in input_batch_list]

    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    label_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    wordType_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    ### bert seq tensor
    bert_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len + 2)), volatile=volatile_flag).long()
    bert_mask = autograd.Variable(torch.zeros((batch_size, max_seq_len + 2))).byte()

    for idx, (seq,bert_id, biseq, label,wordType, seqlen) in enumerate(zip(words, bert_ids, biwords, labels,wordType_ids, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        wordType_seq_tensor[idx, :seqlen] = torch.LongTensor(wordType)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.item())
        bert_seq_tensor[idx, :seqlen + 2] = torch.LongTensor(bert_id)
        bert_mask[idx, :seqlen+2] = torch.LongTensor([1]*int(seqlen+2))

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    wordType_seq_tensor = wordType_seq_tensor[word_perm_idx]
    bert_seq_tensor=bert_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    bert_mask=bert_mask[word_perm_idx]

    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)),
                                        volatile=volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    data_domain = [data_domain[i] for i in word_perm_idx]
    data_domain=torch.LongTensor(data_domain)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        bert_seq_tensor=bert_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        wordType_seq_tensor = wordType_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
        bert_mask=bert_mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, wordType_seq_tensor,mask.bool(), data_domain, bert_seq_tensor, bert_mask


def batchify_without_label(input_batch_list, gpu, volatile_flag=False):
    '''
    Padding the sequence without label for predict
    :param input_batch_list:
    :param gpu:
    :param volatile_flag:
    :return:
    '''
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    biwords = [sent[1] for sent in input_batch_list]
    chars = [sent[2] for sent in input_batch_list]
    gazs = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    biword_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).long()
    mask = autograd.Variable(torch.zeros((batch_size, max_seq_len)), volatile=volatile_flag).byte()
    for idx, (seq, biseq, seqlen) in enumerate(zip(words, biwords, word_seq_lengths)):
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
        mask[idx, :seqlen] = torch.Tensor([1] * seqlen.item())

    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    pad_chars = [chars[idx] + [[0]] * (max_seq_len - len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len)),
                                        volatile=volatile_flag).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        biword_seq_tensor = biword_seq_tensor.cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return gaz_list, word_seq_tensor, biword_seq_tensor, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, mask


def train(data, save_model_dir, save_data_set, seg=True):
    '''
    Train model
    :param data:
    :param log_file:
    :param save_model_dir:
    :param save_data_set:
    :param seg:
    :return:
    '''
    data.show_data_summary()
    save_data_name = save_data_set
    save_data_setting(data, save_data_name)
    model = SeqModel(data)
    model.lstm.word_embeddings.weight.requires_grad = False
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    # optimizer = optim.Adam(parameters, lr=data.HP_lr)
    best_dev = -1

    logger.info("Now we define the model, and word lexcion gradient is %s." \
                % model.lstm.word_embeddings.weight.requires_grad)




    #caculate domain vector
    source_domain_embs = []
    target_domain_embs = []
    batch_size = data.HP_batch_size
    data_Ids=[]
    data_Ids.extend(data.train_Ids)
    data_num = len(data_Ids)
    total_batch = data_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > data_num:
            end = data_num
        instance = data_Ids[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_wordtype, mask, data_domain, bert_seq_tensor, bert_mask = batchify_with_label(
            instance, data.HP_gpu)
        source_domain_batch_embs, target_domain_batch_embs = model.domainVector(batch_label, bert_seq_tensor, bert_mask, data_domain)
        source_domain_embs.append(source_domain_batch_embs)
        target_domain_embs.append(target_domain_batch_embs)
    source_domain_vector = torch.cat(source_domain_embs, 0).mean(0)
    target_domain_vector = torch.cat(target_domain_embs, 0).mean(0)


    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        logger.info("Epoch: %s/%s" % (idx, data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0

        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        train_num = len(data.train_Ids)
        total_batch = train_num // batch_size + 1
        end = 0



        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, batch_wordtype, mask, data_domain,bert_seq_tensor, bert_mask = batchify_with_label(
                instance, data.HP_gpu)
            instance_count += 1
            loss, tag_seq, loss_list = model.neg_log_likelihood_loss( batch_word, batch_label, batch_wordtype, mask,
                                                          data_domain, bert_seq_tensor, bert_mask, source_domain_vector, target_domain_vector)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.data
            total_loss += loss.data
            batch_loss += loss
            if end % (data.HP_batch_size*5) == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                logger.info("Instance: %s; Time: %.2fs; loss: %.4f(crf_loss: %.4f; separation_loss: %.4f); acc: %s/%s=%.4f" % (
                    end, temp_cost, sample_loss,loss_list[0],loss_list[1], right_token, whole_token, (right_token + 0.) / whole_token))
                sample_loss = 0

            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0

        temp_time = time.time()
        temp_cost = temp_time - temp_start
        logger.info("Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (
            end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start

        logger.info("\nEpoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss of train: %s" % (
            idx, epoch_cost, train_num / epoch_cost, total_loss))



        speed, acc, p, r, f, _ = evaluate(data, model, "dev",source_domain_vector, target_domain_vector)
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if seg:
            current_score = f
            logger.info("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                dev_cost, speed, acc, p, r, f))

        else:
            current_score = acc
            logger.info("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))
        if current_score > best_dev:
            if seg:
                logger.info("Exceed previous best f score: %.4f" % (best_dev))
            else:
                logger.info("Exceed previous best acc score: %.4f" % (best_dev))
            model_name = save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_name)
            best_dev = current_score

        speed, acc, p, r, f, _ = evaluate(data, model, "test",source_domain_vector, target_domain_vector)
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            logger.info("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                test_cost, speed, acc, p, r, f))
        else:
            logger.info("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))

        gc.collect()



def load_model_decode(model_dir, data, name, gpu, seg=True):
    '''
    Load model when decode
    :param model_dir:
    :param data:
    :param name:
    :param gpu:
    :param seg:
    :return:
    '''
    data.HP_gpu = gpu
    logger.info("Load Model from file: ", model_dir)
    model = SeqModel(data)
    model.load_state_dict(torch.load(model_dir))
    logger.info("Decode %s data ..." % (name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        logger.info("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
            name, time_cost, speed, acc, p, r, f))
    else:
        logger.info("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results





if __name__ == '__main__':

    if status == 'train':
        if not os.path.exists(args.savedir):
            os.mkdir(args.savedir)

        args.char_emb = args.emb
        args.bichar_emb = args.emb
        args.gaz_emb = args.emb

        # train, valid, test file
        args.train_file=os.path.join('data', args.source, args.target, 'train.txt')
        args.dev_file = os.path.join('data', args.source, args.target, 'valid.txt')
        args.test_file = os.path.join('data', args.source, args.target, 'test.txt')
        # entity lexicon in source and target
        lexcion_dir=os.path.join('lexcion', args.source, args.target)
        args.pro_file=[os.path.join(lexcion_dir, 'source.txt'),os.path.join(lexcion_dir, 'target.txt')]

        savemodel = args.savedir + "/saved_model" + "BERTLstmCrf"
        logger.info("模型训练参数:\nCuDNN:{}\tGpu available:{}\t分词:{}".format(
            torch.backends.cudnn.enabled, args.gpu, args.seg
        ))
        logger.info("字向量路径:{}\nbi-字向量路径:{}\ngaz_embedding路径:{}\n"
                    .format(args.char_emb, args.bichar_emb, args.gaz_emb))
        logger.info("训练集路径:{}\n验证集路径:{}\n测试集路径{}".format(args.train_file
                                                         , args.dev_file, args.test_file))
        logger.info("模型保存路径:{}".format(savemodel))
        data = Data()
        data.separation = args.separation
        data.alignment = args.alignment
        data.HP_gpu = args.gpu
        data.use_bigram = False
        data.number_normalized = False
        data.separation_loss_weight = args.separation_loss_weight
        data.use_bert=args.use_bert
        data_initialization(data, args.gaz_emb, args.train_file, args.dev_file, args.test_file, args.pro_file)
        data.generate_instance_with_gaz(args.train_file, 'train')
        data.generate_instance_with_gaz(args.dev_file, 'dev')
        data.generate_instance_with_gaz(args.test_file, 'test')
        data.build_word_pretrain_emb(args.char_emb)
        if data.use_bigram:
            data.build_biword_pretrain_emb(args.bichar_emb)
        data.build_gaz_pretrain_emb(args.gaz_emb)
        train(data, savemodel, args.savedset, args.seg)

    elif status == 'test':
        data = load_data_setting(args.savedset)
        data.generate_instance_with_gaz(args.dev_file, 'dev')
        load_model_decode(args.load_model, data, 'dev', args.gpu, args.seg)
        data.generate_instance_with_gaz(args.test_file, 'test')
        load_model_decode(args.load_model, data, 'test', args.gpu, args.seg)

    elif status == 'decode':
        data = load_data_setting(args.savedset)
        data.generate_instance_with_gaz(args.raw_file, 'raw')
        decode_results = load_model_decode(args.load_model, data, 'raw',
                                           args.gpu, args.seg)
        data.write_decoded_results(args.decodeOutput, decode_results, 'raw')

    else:
        logger.info("Invalid argument! Please use valid arguments! (train/test/decode/predict)")
