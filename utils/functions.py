# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:23:06
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-05-12 22:09:37

import os
import numpy as np
from utils.alphabet import Alphabet
from transformers.tokenization_bert import BertTokenizer

from tqdm import *

NULLKEY = "-null-"


def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word


def read_instance(input_file, word_alphabet, char_alphabet, label_alphabet, number_normalized, max_sent_length,
                  char_padding_size=-1, char_padding_symbol='</pad>'):
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    chars = []
    labels = []
    word_Ids = []
    char_Ids = []
    label_Ids = []
    for line in in_lines:
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0].decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, chars, labels])
                instence_Ids.append([word_Ids, char_Ids, label_Ids])
            words = []
            chars = []
            labels = []
            word_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def read_seg_instance(input_file, word_alphabet, biword_alphabet, char_alphabet, label_alphabet, number_normalized,
                      max_sent_length, char_padding_size=-1, char_padding_symbol='</pad>'):
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in xrange(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0].decode('utf-8')
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            words.append(word)
            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                biword = word + in_lines[idx + 1].strip().split()[0].decode('utf-8')
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            labels.append(label)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if (max_sent_length < 0) or (len(words) < max_sent_length):
                instence_texts.append([words, biwords, chars, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, label_Ids])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
    return instence_texts, instence_Ids


def read_instance_with_gaz(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet,
                           number_normalized, max_sent_length,source_pro, target_pro,data_domain=0, char_padding_size=-1, char_padding_symbol='</pad>'):
    # print("input_file",str(input_file))

    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('pytorch_pretrained_model/bert-base-chinese-vocab.txt', do_lower_case=True)

    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    wordTypes=[]
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    # 默认为源领域，遇到‘-’行后切换为目标领域1
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            label = pairs[-1]
            # if word in source_pro and word in target_pro:
            #     wordType= data_domain+1
            # elif word in source_pro:
            #     #源领域
            #     wordType=1
            # elif word in target_pro:
            #     #目标领域
            #     wordType=2
            # else:
            #     #非entity token
            #     wordType=3
            #
            if word in source_pro or word in target_pro:
                wordType= data_domain+1
            else:
                #非entity token
                wordType=3

            # if label !='O':
            #     wordType= data_domain+1
            # else:
            #     #非entity token
            #     wordType=3

            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                biword = word + in_lines[idx + 1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            labels.append(label)
            wordTypes.append(wordType)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            label_Ids.append(label_alphabet.get_index(label))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            # char_padding_size=-1
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):
                gazs = []
                gaz_Ids = []
                w_length = len(words)
                for idx in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    # 词的类型，0为非产品， 1为source产品，2为target产品， 3为source和target公共产品
                    gaz_type=[]
                    for word in matched_list:
                        if word in source_pro and word in target_pro:
                            gaz_type.append(3)
                        elif word in source_pro:
                            gaz_type.append(1)
                        elif word in target_pro:
                            gaz_type.append(2)
                        else:
                            gaz_type.append(0)
                    # 调试
                    # if gaz_type != [0]*len(matched_list):
                    #     print(gaz_type)
                    #     print(matched_list)
                    gazs.append(matched_list)
                    matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_Id:
                        assert len(matched_Id)==len(gaz_type)
                        gaz_Ids.append([matched_Id, matched_length,gaz_type])
                    else:
                        gaz_Ids.append([])

                texts = ['[CLS]'] + words + ['[SEP]']
                bert_text_ids = tokenizer.convert_tokens_to_ids(texts)

                instence_texts.append([words, biwords, chars, gazs, labels, data_domain])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids, data_domain,bert_text_ids,wordTypes])
            # 数据切换为目标领域
            if len(line) == 2:
                data_domain = 1
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
            gazs = []
            gaz_Ids = []
            wordTypes = []
    return instence_texts, instence_Ids


def read_instance_with_gaz_predict(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet, label_alphabet,
                           number_normalized, max_sent_length, char_padding_size=-1, char_padding_symbol='</pad>'):
    # print("input_file",str(input_file))
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_Ids = []
    words = []
    biwords = []
    chars = []
    labels = []
    word_Ids = []
    biword_Ids = []
    char_Ids = []
    label_Ids = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) >= 2:
            pairs = line.strip().split()
            word = pairs[0]
            if number_normalized:
                word = normalize_word(word)
            if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                biword = word + in_lines[idx + 1].strip().split()[0]
            else:
                biword = word + NULLKEY
            biwords.append(biword)
            words.append(word)
            word_Ids.append(word_alphabet.get_index(word))
            biword_Ids.append(biword_alphabet.get_index(biword))
            char_list = []
            char_Id = []
            for char in word:
                char_list.append(char)
            # char_padding_size=-1
            if char_padding_size > 0:
                char_number = len(char_list)
                if char_number < char_padding_size:
                    char_list = char_list + [char_padding_symbol] * (char_padding_size - char_number)
                assert (len(char_list) == char_padding_size)
            else:
                ### not padding
                pass
            for char in char_list:
                char_Id.append(char_alphabet.get_index(char))
            chars.append(char_list)
            char_Ids.append(char_Id)
        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words) > 0):
                gazs = []
                gaz_Ids = []
                w_length = len(words)
                for idx in range(w_length):
                    matched_list = gaz.enumerateMatchList(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    gazs.append(matched_list)
                    matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])
                instence_texts.append([words, biwords, chars, gazs, labels])
                instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Ids])
            words = []
            biwords = []
            chars = []
            labels = []
            word_Ids = []
            biword_Ids = []
            char_Ids = []
            label_Ids = []
            gazs = []
            gaz_Ids = []
    return instence_texts, instence_Ids


def read_instance_with_gaz_in_sentence(input_file, gaz, word_alphabet, biword_alphabet, char_alphabet, gaz_alphabet,
                                       label_alphabet, number_normalized, max_sent_length, char_padding_size=-1,
                                       char_padding_symbol='</pad>'):
    in_lines = open(input_file, 'r', encoding='utf-8').readlines()
    instence_texts = []
    instence_Ids = []
    for idx in range(len(in_lines)):
        pair = in_lines[idx].strip().decode('utf-8').split()
        orig_words = list(pair[0])

        if (max_sent_length > 0) and (len(orig_words) > max_sent_length):
            continue
        biwords = []
        biword_Ids = []
        if number_normalized:
            words = []
            for word in orig_words:
                word = normalize_word(word)
                words.append(word)
        else:
            words = orig_words
        word_num = len(words)
        for idy in range(word_num):
            if idy < word_num - 1:
                biword = words[idy] + words[idy + 1]
            else:
                biword = words[idy] + NULLKEY
            biwords.append(biword)
            biword_Ids.append(biword_alphabet.get_index(biword))
        word_Ids = [word_alphabet.get_index(word) for word in words]
        label = pair[-1]
        label_Id = label_alphabet.get_index(label)
        gazs = []
        gaz_Ids = []
        word_num = len(words)
        chars = [[word] for word in words]
        char_Ids = [[char_alphabet.get_index(word)] for word in words]
        for idx in range(word_num):
            matched_list = gaz.enumerateMatchList(words[idx:])
            matched_length = [len(a) for a in matched_list]
            gazs.append(matched_list)
            matched_Id = [gaz_alphabet.get_index(entity) for entity in matched_list]
            if matched_Id:
                gaz_Ids.append([matched_Id, matched_length])
            else:
                gaz_Ids.append([])
        instence_texts.append([words, biwords, chars, gazs, label])
        instence_Ids.append([word_Ids, biword_Ids, char_Ids, gaz_Ids, label_Id])
    return instence_texts, instence_Ids


def build_pretrain_embedding(embedding_path, word_alphabet, embedd_dim=100, norm=True):
    path = embedding_path.split('\\')
    embedd_dict = dict()
    if path[-1] == 'Tencent_AILab_ChineseEmbedding.txt':
        embedd_dict, embedd_dim = load_pretrain_emb_tencent(embedding_path)
    else:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([word_alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_alphabet.iteritems():
        if word in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word])
            else:
                pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[word.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(pretrain_emb)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / word_alphabet.size()))
    information = "Embedding:\n  pretrain word:" + str(pretrained_size) + ", prefect match" + str(
        perfect_match) + ":, case_match:" + str(case_match) + ", oov:" + str(not_match) + ",+oov%%:" + str(
        (not_match + 0.) / word_alphabet.size()) + "\n"

    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict, embedd_dim


def load_pretrain_emb_tencent(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding='utf-8') as f:
        next(f)
        i = 0
        for line in tqdm(f, total=8824330):
            e = line[:-1].split(' ')
            w = e[0]
            embedd_dim = len(e[1:])
            embedd = []
            # print(e[1:])
            vec = [float(data) for data in e[1:]]
            embedd.append(np.array(vec, dtype=float))
            embedd_dict[w] = np.array(embedd)
    return embedd_dict, embedd_dim


if __name__ == '__main__':
    a = np.arange(9.0)
    print(a)
    print(norm2one(a))
