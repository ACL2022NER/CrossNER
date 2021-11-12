# -*- coding: utf-8 -*-

import sys
from utils.alphabet import Alphabet
from utils.functions import *
from utils.gazetteer import Gazetteer
from tqdm import *
import jieba

START = "</s>"
UNKNOWN = "</unk>"
PADDING = "</pad>"
NULLKEY = "-null-"


class Data:

    def __init__(self):
        self.MAX_SENTENCE_LENGTH = 250
        self.MAX_WORD_LENGTH = -1
        self.number_normalized = False

        # lexcion config
        self.norm_word_emb = True
        self.norm_biword_emb = True
        self.norm_gaz_emb = False
        self.word_alphabet = Alphabet('word')
        self.biword_alphabet = Alphabet('biword')
        self.char_alphabet = Alphabet('character')
        # self.word_alphabet.add(START)
        # self.word_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(START)
        # self.char_alphabet.add(UNKNOWN)
        # self.char_alphabet.add(PADDING)
        self.label_alphabet = Alphabet('label', True)
        self.gaz_lower = False
        self.gaz = Gazetteer(self.gaz_lower)
        self.gaz_alphabet = Alphabet('gaz')
        self.HP_fix_gaz_emb = False
        self.HP_use_gaz = True
        self.tagScheme = "NoSeg"
        self.char_features = "LSTM"

        self.train_texts = []
        self.dev_texts = []
        self.test_texts = []
        self.raw_texts = []

        self.train_Ids = []
        self.dev_Ids = []
        self.test_Ids = []
        self.raw_Ids = []

        self.source_pro = []
        self.target_pro = []

        self.use_bigram = True
        self.word_emb_dim = 50
        self.biword_emb_dim = 200
        self.char_emb_dim = 30

        self.gaz_emb_dim = 50
        self.gaz_dropout = 0.5

        self.pretrain_word_embedding = None
        self.pretrain_biword_embedding = None
        self.pretrain_gaz_embedding = None

        self.label_size = 0

        # dictionary size
        self.word_alphabet_size = 0
        self.biword_alphabet_size = 0
        self.char_alphabet_size = 0
        self.label_alphabet_size = 0

        self.HP_iteration = 80
        self.HP_batch_size = 10

        # hidden layer of character
        self.HP_char_hidden_dim = 50
        self.HP_hidden_dim = 200
        self.HP_dropout = 0.5
        self.HP_lstm_layer = 1
        self.HP_bilstm = True

        self.HP_use_char = False
        self.HP_gpu = False

        # learning rate
        self.HP_lr = 0.005
        # learning decay
        self.HP_lr_decay = 0.05
        # gradient range
        self.HP_clip = 5.0
        # momentum
        self.HP_momentum = 0

    def show_data_summary(self):
        '''
        Show data information
        :return:
        '''
        print("DATA SUMMARY START:")
        print("     Tag          scheme: %s" % (self.tagScheme))
        print("     MAX SENTENCE LENGTH: %s" % (self.MAX_SENTENCE_LENGTH))
        print("     MAX   WORD   LENGTH: %s" % (self.MAX_WORD_LENGTH))
        print("     Number   normalized: %s" % (self.number_normalized))
        print("     Use          bigram: %s" % (self.use_bigram))
        print("     Word  alphabet size: %s" % (self.word_alphabet_size))
        print("     Biword alphabet size: %s" % (self.biword_alphabet_size))
        print("     Char  alphabet size: %s" % (self.char_alphabet_size))
        print("     Gaz   alphabet size: %s" % (self.gaz_alphabet.size()))
        print("     Label alphabet size: %s" % (self.label_alphabet_size))
        print("     Word lexcion size: %s" % (self.word_emb_dim))
        print("     Biword lexcion size: %s" % (self.biword_emb_dim))
        print("     Char lexcion size: %s" % (self.char_emb_dim))
        print("     Gaz lexcion size: %s" % (self.gaz_emb_dim))
        print("     Norm     word   emb: %s" % (self.norm_word_emb))
        print("     Norm     biword emb: %s" % (self.norm_biword_emb))
        print("     Norm     gaz    emb: %s" % (self.norm_gaz_emb))
        print("     Norm   gaz  dropout: %s" % (self.gaz_dropout))
        print("     Train instance number: %s" % (len(self.train_texts)))
        print("     Dev   instance number: %s" % (len(self.dev_texts)))
        print("     Test  instance number: %s" % (len(self.test_texts)))
        print("     Raw   instance number: %s" % (len(self.raw_texts)))
        print("     Hyperpara  iteration: %s" % (self.HP_iteration))
        print("     Hyperpara  batch size: %s" % (self.HP_batch_size))
        print("     Hyperpara          lr: %s" % (self.HP_lr))
        print("     Hyperpara    lr_decay: %s" % (self.HP_lr_decay))
        print("     Hyperpara     HP_clip: %s" % (self.HP_clip))
        print("     Hyperpara    momentum: %s" % (self.HP_momentum))
        print("     Hyperpara  hidden_dim: %s" % (self.HP_hidden_dim))
        print("     Hyperpara     dropout: %s" % (self.HP_dropout))
        print("     Hyperpara  lstm_layer: %s" % (self.HP_lstm_layer))
        print("     Hyperpara      bilstm: %s" % (self.HP_bilstm))
        print("     Hyperpara         GPU: %s" % (self.HP_gpu))
        print("     Hyperpara     use_gaz: %s" % (self.HP_use_gaz))
        print("     Hyperpara fix gaz emb: %s" % (self.HP_fix_gaz_emb))
        print("     Hyperpara    use_char: %s" % (self.HP_use_char))
        if self.HP_use_char:
            print("             Char_features: %s" % (self.char_features))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    def refresh_label_alphabet(self, input_file):
        old_size = self.label_alphabet_size
        self.label_alphabet.clear(True)
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        for line in in_lines:
            if len(line) > 2:
                pairs = line.strip().split()
                label = pairs[-1]
                self.label_alphabet.add(label)
        self.label_alphabet_size = self.label_alphabet.size()
        startS = False
        startB = False
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True

        if startB:
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"
        self.fix_alphabet()
        print("Refresh label alphabet finished: old:%s -> new:%s" % (old_size, self.label_alphabet_size))

    def build_alphabet(self, input_file):
        """  Build dictionary for train/test/dev data
        """
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        for idx in range(len(in_lines)):
            line = in_lines[idx]
            if len(line) > 2:
                pairs = line.strip().split()
                word = pairs[0]
                if self.number_normalized:
                    word = normalize_word(word)
                label = pairs[-1]
                self.label_alphabet.add(label)
                self.word_alphabet.add(word)

                if idx < len(in_lines) - 1 and len(in_lines[idx + 1]) > 2:
                    biword = word + in_lines[idx + 1].strip().split()[0]
                else:
                    biword = word + NULLKEY
                self.biword_alphabet.add(biword)
                for char in word:
                    self.char_alphabet.add(char)
            # elif len(line) == 2:
            #     print(line)

        self.word_alphabet_size = self.word_alphabet.size()
        self.biword_alphabet_size = self.biword_alphabet.size()
        self.char_alphabet_size = self.char_alphabet.size()
        self.label_alphabet_size = self.label_alphabet.size()
        # #
        startS = False
        startB = False
        # instance2index.items()
        for label, _ in self.label_alphabet.iteritems():
            if "S-" in label.upper():
                startS = True
            elif "B-" in label.upper():
                startB = True
        if startB:
            self.tagScheme = "BMES"
            if startS:
                self.tagScheme = "BMES"
            else:
                self.tagScheme = "BIO"

    def build_gaz_file(self, gaz_file):
        '''
        Build dictionary for gaz_file
        :param gaz_file:
        :return:
        '''
        ## build gaz file,initial read gaz lexcion file
        path = gaz_file.split('\\')
        if path[-1] == 'Tencent_AILab_ChineseEmbedding.txt':
            with open(gaz_file, 'r', encoding='utf-8') as f:
                next(f)
                i = 0
                for line in tqdm(f, total=8824330):
                    e = line[:-1].split(' ')
                    w = e[0]
                    self.gaz.insert(w, "one_source")

            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())
        else:
            with open(gaz_file, 'r', encoding='utf-8') as f:
                fins = f.readlines()
            for fin in fins:
                fin = fin.strip().split()[0]
                if fin:
                    self.gaz.insert(fin, "one_source")

            print("Load gaz file: ", gaz_file, " total size:", self.gaz.size())

    def build_gaz_alphabet(self, input_file):
        """ Build dictionary for gaz(在train,dev,test file在embedding中匹配到的词语)
        """
        in_lines = open(input_file, 'r', encoding='utf-8').readlines()
        word_list = []
        for line in in_lines:
            if len(line) > 3:
                word = line.split()[0]
                if self.number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                w_length = len(word_list)
                for idx in range(w_length):
                    matched_entity = self.gaz.enumerateMatchList(word_list[idx:])
                    for entity in matched_entity:
                        self.gaz_alphabet.add(entity)
                word_list = []
        print("gaz alphabet size:", self.gaz_alphabet.size())

    def load_pro(self, pro_file):
        # txt format
        with open(pro_file[0], 'r', encoding='utf-8') as f:
            for item in f.readlines():
                phases=item.strip().split()
                for phase in phases:
                    self.source_pro.extend(list(phase))

        with open(pro_file[1], 'r', encoding='utf-8') as f:
            for item in f.readlines():
                phases = item.strip().split()
                for phase in phases:
                    self.target_pro.extend(list(phase))


    def fix_alphabet(self):
        self.word_alphabet.close()  # not allow growing
        self.biword_alphabet.close()
        self.char_alphabet.close()
        self.label_alphabet.close()
        self.gaz_alphabet.close()

    def build_word_pretrain_emb(self, emb_path):
        """ Build word pre-train lexcion
        """
        print("build word pretrain emb...")
        self.pretrain_word_embedding, self.word_emb_dim = build_pretrain_embedding(emb_path, self.word_alphabet,
                                                                                   self.word_emb_dim,
                                                                                   self.norm_word_emb)

    def build_biword_pretrain_emb(self, emb_path):
        '''
        Build bi-word pre-train lexcion
        :param emb_path:
        :return:
        '''
        print("build biword pretrain emb...")
        self.pretrain_biword_embedding, self.biword_emb_dim = build_pretrain_embedding(emb_path, self.biword_alphabet,
                                                                                       self.biword_emb_dim,
                                                                                       self.norm_biword_emb)

    def build_gaz_pretrain_emb(self, emb_path):
        '''
        Build gaz pre-train lexcion
        :param emb_path:
        :return:
        '''
        print("build gaz pretrain emb...")
        self.pretrain_gaz_embedding, self.gaz_emb_dim = build_pretrain_embedding(emb_path, self.gaz_alphabet,
                                                                                 self.gaz_emb_dim, self.norm_gaz_emb)

    def generate_instance(self, input_file, name):
        '''
        Read instance
        :param input_file:
        :param name:
        :return:
        '''
        self.fix_alphabet()
        if name == "train":
            self.train_texts, self.train_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                                 self.char_alphabet, self.label_alphabet,
                                                                 self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.char_alphabet, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "test":
            self.test_texts, self.test_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                               self.char_alphabet, self.label_alphabet,
                                                               self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_seg_instance(input_file, self.word_alphabet, self.biword_alphabet,
                                                             self.char_alphabet, self.label_alphabet,
                                                             self.number_normalized, self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def generate_instance_with_gaz(self, input_file, name):
        self.fix_alphabet()  # not allow growing
        if name == "train":
            self.train_texts, self.train_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                      self.biword_alphabet, self.char_alphabet,
                                                                      self.gaz_alphabet, self.label_alphabet,
                                                                      self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                                      self.source_pro, self.target_pro,data_domain=0)
        elif name == "dev":
            self.dev_texts, self.dev_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                  self.biword_alphabet, self.char_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                                  self.source_pro, self.target_pro,data_domain=1)
        elif name == "test":
            self.test_texts, self.test_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                    self.biword_alphabet, self.char_alphabet,
                                                                    self.gaz_alphabet, self.label_alphabet,
                                                                    self.number_normalized, self.MAX_SENTENCE_LENGTH,
                                                                    self.source_pro, self.target_pro,data_domain=1)
        elif name == "raw":
            self.raw_texts, self.raw_Ids = read_instance_with_gaz(input_file, self.gaz, self.word_alphabet,
                                                                  self.biword_alphabet, self.char_alphabet,
                                                                  self.gaz_alphabet, self.label_alphabet,
                                                                  self.number_normalized, self.MAX_SENTENCE_LENGTH)
        elif name == "predict":
            self.predict_texts, self.predict_Ids = read_instance_with_gaz_predict(input_file, self.gaz,
                                                                                  self.word_alphabet,
                                                                                  self.biword_alphabet,
                                                                                  self.char_alphabet, self.gaz_alphabet,
                                                                                  self.label_alphabet,
                                                                                  self.number_normalized,
                                                                                  self.MAX_SENTENCE_LENGTH)
        else:
            print("Error: you can only generate train/dev/test instance! Illegal input:%s" % (name))

    def write_decoded_results(self, output_file, predict_results, name):
        '''
        Output decode result to file
        :param output_file:
        :param predict_results:
        :param name:
        :return:
        '''
        fout = open(output_file, 'w', encoding='utf-8')
        sent_num = len(predict_results)
        content_list = []
        if name == 'raw':
            content_list = self.raw_texts
        elif name == 'test':
            content_list = self.test_texts
        elif name == 'dev':
            content_list = self.dev_texts
        elif name == 'train':
            content_list = self.train_texts
        else:
            print("Error: illegal name during writing predict result, name should be within train/dev/test/raw !")
        assert (sent_num == len(content_list))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))

    def write_predict_results(self, output_file, predict_results, name):
        '''
        Output predict result to file
        :param output_file:
        :param predict_results:
        :param name:
        :return:
        '''
        fout = open(output_file, 'w', encoding='utf-8')
        sent_num = len(predict_results)
        content_list = []
        if name == 'predict':
            content_list = self.predict_texts
        else:
            print("Error: illegal name during writing predict result, name should be predict !")
        assert sent_num == len(content_list), print(
            "sent_num:%d\nlen of content_list:%d" % (sent_num, len(content_list)))
        for idx in range(sent_num):
            sent_length = len(predict_results[idx])
            for idy in range(sent_length):
                ## content_list[idx] is a list with [word, char, label]
                fout.write(content_list[idx][0][idy] + " " + predict_results[idx][idy] + '\n')

            fout.write('\n')
        fout.close()
        print("Predict %s result has been written into file. %s" % (name, output_file))
