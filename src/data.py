# !/usr/bin/python3
# -*- coding:utf-8 -*-




import re
import codecs
import torch
import random
from math import ceil
from numpy import array
from utils import color
# from random import shuffle
# from torch.utils.data import DataLoader





class DataUtils(object):
    """
    Documentation for DataLoader.
    """
    def __init__(self, args):
        super(DataUtils, self).__init__()
        print("\nInitializing Data Utils Object ...\n")
        self.set_random_seed(args)
        self.args = args
        self.train_data_path = args.train_data_path
        self.valid_data_path = args.valid_data_path
        self.test_data_path  = args.test_data_path
        self.max_seq_len = args.max_seq_len
        self.max_token_len = args.max_token_len
        self.max_target_len = args.max_target_len

        self.PAD_idx = args.PAD_idx
        self.BOS_idx = args.BOS_idx
        self.EOS_idx = args.EOS_idx
        self.OOV_idx = args.OOV_idx
        self.PAD_token = args.PAD_token
        self.BOS_token = args.BOS_token
        self.EOS_token = args.EOS_token
        self.OOV_token  =args.OOV_token
        
        self.flag_constructed_dict = False



    def set_random_seed(self, args):
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)



    def read_lines_from_file(self, input_path, with_target):
        input_file = codecs.open(input_path, "r", "utf-8")
        lines = input_file.readlines()
        pattern = re.compile("[^\u4E00-\u9FA5]+")
        map_fun = lambda word: len(word) >1 and word != "".join(re.findall(pattern,word))
        new_lines = []

        if with_target:
            for idx, line in enumerate(lines):
                line = line.rstrip().split("\t")
                if len(line) !=3:
                    print(f"wrong data in line: {idx}, should contain source1、source2 and target data.")
                    continue
                source1_words = [word for word in line[0].split(" ") if map_fun(word)] #[self.BOS_token] + 
                source2_words = [word for word in line[1].split(" ") if map_fun(word)]
                target_words  = [word for word in line[2].split(" ") if map_fun(word)]
                
                # 过滤source1、source2或target太短的数据,source2至少包含1个token，target至少包含2个有效token。
                if len(source1_words) <2 or len(source2_words) <1 or len(target_words) < 2:
                    continue
                # Any token in target should be contained in source1 or source2.
                if len(set(target_words) - set(source1_words + source2_words)) > 0:
                    print(f"wrong data in line: {idx}, target words should be contained in source1 and source2.")
                    continue
                new_lines.append([source1_words, source2_words, target_words])
        else:
            for idx, line in enumerate(lines):
                line = line.rstrip().split("\t")
                if len(line) !=2:
                    print(f"wrong data in line: {idx}, should contain source1、source2 data")
                    continue
                source1_words = [word for word in line[0].split(" ") if map_fun(word)]
                source2_words = [word for word in line[1].split(" ") if map_fun(word)]
                if len(source1_words) <2 or len(source2_words) <1:
                    continue
                new_lines.append([source1_words, source2_words])

        return new_lines





    def construct_dict(self):
        """从训练数据中构建词典"""
        print("\nConstructing Dict ...\n")
        self.train_lines = self.read_lines_from_file(self.train_data_path, with_target=True)
        self.valid_lines = self.read_lines_from_file(self.valid_data_path, with_target=True)
        if self.test_data_path is not None:
            self.test_lines  = self.read_lines_from_file(self.test_data_path, with_target=True)
        else:
            self.test_lines  = None

        self.word2index = {self.PAD_token: self.PAD_idx, self.BOS_token:self.BOS_idx, self.EOS_token:self.EOS_idx, self.OOV_token:self.OOV_idx}
        self.char2index = {self.PAD_token: self.PAD_idx, self.BOS_token:self.BOS_idx, self.EOS_token:self.EOS_idx, self.OOV_token:self.OOV_idx}

        def update_dict(lines):
            for line in lines:
                source1_words, source2_words = line[0], line[1]
                for word in source1_words + source2_words:
                    if len(word) < 1 or word in self.word2index or word == self.BOS_token:
                        continue
                    self.word2index[word] = len(self.word2index)
                    for char in word:
                        if char in self.char2index:
                            continue
                        self.char2index[char] = len(self.char2index)
        
        update_dict(self.train_lines)
        # update_dict(self.valid_lines)
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.index2char = {v:k for k,v in self.char2index.items()}

        self.flag_constructed_dict = True
        vocab_file = {"word2index": self.word2index, "char2index": self.char2index}
        torch.save(vocab_file, self.args.vocab_file)
        print(color("\nSuccessfully Constructed Dict.", 2))
        print(color("\nVocabulary File Has Been Saved.", 1))
        print("\nNumber of words: ", color(len(self.word2index), 2))
        print("Number of chars: ", color(len(self.char2index), 2),"\n\n")

        return self.word2index, self.char2index, self.index2word, self.index2char






    def construct_dict_v2(self):
        """从训练数据中构建词典"""
        print("\nConstructing Dict ...\n")
        self.train_lines = self.read_lines_from_file(self.train_data_path, with_target=True)
        self.valid_lines = self.read_lines_from_file(self.valid_data_path, with_target=True)

        self.special_tokens = {self.PAD_token:self.PAD_idx, self.BOS_token:self.BOS_idx, 
                               self.EOS_token:self.EOS_idx, self.OOV_token:self.OOV_idx}

        word_indexer = list({word.strip() for line in self.train_lines for word in line[0]+line[1] if len(word.strip()) >1})
        char_indexer = list(set("".join(word_indexer)))

        for token, index in self.special_tokens.items():
            word_indexer.insert(index, token)
            char_indexer.insert(index, token)

        self.index2word = {index:word for index, word in enumerate(word_indexer)}
        self.word2index = {word:index for index, word in enumerate(word_indexer)}
        self.index2char = {index:word for index, word in enumerate(char_indexer)}
        self.char2index = {word:index for index, word in enumerate(char_indexer)}

        self.flag_constructed_dict = True
        vocab_file = {"word2index": self.word2index, "char2index": self.char2index}
        torch.save(vocab_file, self.args.vocab_file)
        print(color("\nSuccessfully Constructed Dict.", 2))
        print(color("\nVocabulary File Has Been Saved.\n", 1))
        print("Number of words: ", color(len(self.word2index), 2))
        print("Number of chars: ", color(len(self.char2index), 2),"\n\n")

        return self.word2index, self.char2index, self.index2word, self.index2char






    def padding_and_covert_to_idx(self, line, data_type):
        """
        args:
            line: List[str], the input line, a list of tokens (words).
            data_type: str, should be in ['train', 'valid', 'test']
        return:
            new_char_list: List[[int, int,...],[int, int, ...],[int, int, ...],..]
                           A list if list contains char id, the outer list has maximum sequence length size,
                           the inner list has maximum token length size.
            new_word_list: List[int, int, int, ...] , 
                           A list of token id with maximum sequence length.
        """

        assert data_type in ["source1", "source2", "target"], "'data_type' should be in ['source1', 'source2', 'target']"
        
        max_token_len = self.max_token_len
        if data_type in ["source1", "source2"]:
            max_seq_len = self.max_seq_len
        else:
            max_seq_len = self.max_target_len

        padded_word_ids = [self.BOS_idx]
        padded_char_ids = [max_token_len * [self.BOS_idx]]

        seq_len = len(line) + 1  if len(line) < max_seq_len else max_seq_len #此处加1是因为前面有'#BOS#'

        for word in line:
            if not len(word) >0: continue
            char_list = [self.char2index.get(char, self.OOV_idx) for char in list(word)]
            N = max_token_len - len(char_list)
            char_list = char_list + N * [self.PAD_idx] if N>0 else char_list[:max_token_len]

            padded_char_ids.append(char_list)
            padded_word_ids.append(self.word2index.get(word, self.OOV_idx))
        padded_word_ids.insert(0, self.BOS_idx)
        padded_char_ids.insert(0, self.max_target_len * [self.BOS_idx])


        # TODO 此处在后面填充的字符。
        N = max_seq_len - len(padded_word_ids)
        padded_word_ids = padded_word_ids + N * [self.PAD_idx] if N>0 else padded_word_ids[:max_seq_len]

        N = max_seq_len - len(padded_char_ids)
        padded_char_ids = padded_char_ids + N * [max_token_len *[self.PAD_idx]] if N>0 else padded_char_ids[: max_seq_len]

        seq_mask = [1 if idx != self.PAD_idx else 0 for idx in padded_word_ids]

        return (padded_word_ids, padded_char_ids, seq_mask, seq_len)




    def get_merged_source_word_ids(self, source1_word_list, source2_word_list):

        merged_source_word_list = list(set(source1_word_list).union(set(source2_word_list))) 
        #TODO 注意此处的填充符号。
        merged_source_word_list = merged_source_word_list + (2* self.max_seq_len - len(merged_source_word_list)) * [self.PAD_token]
        merged_source_local_ids = [self.word2index.get(word, self.OOV_idx) for word in merged_source_word_list]
        merged_source_local_mask = [1 if idx != self.PAD_idx and idx != self.BOS_idx else 0 for idx in merged_source_local_ids]
        merged_source_global_ids = [self.word2index.get(word, self.OOV_idx) for word in merged_source_word_list]

        source1_local_word_ids = [merged_source_word_list.index(word) for word in source1_word_list]
        source2_local_word_ids = [merged_source_word_list.index(word) for word in source2_word_list]

        source1_local_word_ids = source1_local_word_ids + (self.max_seq_len - len(source1_local_word_ids)) * [2* self.max_seq_len-1] \
                                  if(len(source1_local_word_ids) < self.max_seq_len) else source1_local_word_ids[0:self.max_seq_len]
        source2_local_word_ids = source2_local_word_ids + (self.max_seq_len - len(source2_local_word_ids)) * [2* self.max_seq_len-1] \
                                  if(len(source2_local_word_ids) < self.max_seq_len) else source2_local_word_ids[0:self.max_seq_len]

        return (merged_source_word_list, merged_source_local_ids, source1_local_word_ids, 
                source2_local_word_ids, merged_source_local_mask, merged_source_global_ids)




    def obtain_formatted_line(self, line, with_target):
        source1_word_list, source2_word_list = line[0], line[1]
        source1_input_words_ids, source1_input_chars_ids, source1_input_seq_mask, source1_input_seq_len = self.padding_and_covert_to_idx(source1_word_list, "source1")
        source2_input_words_ids, source2_input_chars_ids, source2_input_seq_mask, source2_input_seq_len = self.padding_and_covert_to_idx(source2_word_list, "source2")
        merged_source_word_list, merged_source_local_ids, source1_local_words_ids, source2_local_words_ids, merged_source_local_mask, merged_source_global_ids = \
                                                                      self.get_merged_source_word_ids(source1_word_list, source2_word_list)

        target_word_list, target_words_ids, target_chars_ids, target_seq_mask = None, None, None, None

        if with_target:
            assert len(line) ==3, "wrong data format encountered, target tokens should be given or target tokens should not be None."
            target_word_list = line[2]
            target_words_ids, target_chars_ids, target_seq_mask, target_seq_len = self.padding_and_covert_to_idx(target_word_list, "target")
        
        return (source1_input_words_ids, source1_input_chars_ids, source2_input_words_ids, source2_input_chars_ids, 
                source1_local_words_ids, source2_local_words_ids, merged_source_word_list, merged_source_local_ids, 
                source1_input_seq_mask,  source2_input_seq_mask,  source1_input_seq_len,   source2_input_seq_len, 
                merged_source_local_mask, target_word_list, target_words_ids, target_chars_ids, target_seq_mask, 
                merged_source_global_ids)




    def obtain_formatted_lines(self, lines, with_target):
        formatted_lines = []
        if with_target:
            for idx, line in enumerate(lines):
                formatted_line = self.obtain_formatted_line(line, with_target=with_target)
                formatted_lines.append(formatted_line)
        else:
            for idx, line in enumerate(lines):
                formatted_line = self.obtain_formatted_line(line, with_target=with_target)
                formatted_lines.append(formatted_line)
        return formatted_lines




    def obtain_formatted_data(self, ):
        if not self.flag_constructed_dict:
            self.construct_dict_v2()

        train_data = self.obtain_formatted_lines(self.train_lines, with_target=True)
        valid_data = self.obtain_formatted_lines(self.valid_lines, with_target=True)
        if self.test_data_path:
            test_data  = self.obtain_formatted_lines(self.test_lines, with_target=False)
        else:
            test_data = None

        return (train_data, valid_data, test_data)





    def get_batch_data(self, formatted_lines, with_target, batch_size=50, shuffle=False, device=None):
        single_batch_data = {}
        all_batch_data = []
        N = len(formatted_lines)

        device = device if device else torch.device("cpu")
        all_index = list(range(N)) 
        if shuffle: 
            random.shuffle(all_index)
        all_batch_index = [all_index[start:min(start + batch_size, N)] for start in range(0, N, batch_size)]

        for batch_index in all_batch_index:
            source1_input_words_ids, source1_input_chars_ids = [], []
            source2_input_words_ids, source2_input_chars_ids = [], []
            source1_local_words_ids, source2_local_words_ids = [], []
            merged_source_word_list, merged_source_local_ids = [], []
            source1_input_seq_mask,  source2_input_seq_mask  = [], []
            source1_input_seq_len,   source2_input_seq_len   = [], []
            target_word_list,        target_words_ids,       = [], []
            target_chars_ids,        target_seq_mask,        = [], []
            merged_source_local_mask, merged_source_global_ids = [], []

            for index in batch_index:
                formatted_line = formatted_lines[index]
                source1_input_words_ids.append(formatted_line[0])
                source1_input_chars_ids.append(formatted_line[1])
                source2_input_words_ids.append(formatted_line[2])
                source2_input_chars_ids.append(formatted_line[3])
                source1_local_words_ids.append(formatted_line[4])
                source2_local_words_ids.append(formatted_line[5])
                
                merged_source_word_list.append(formatted_line[6])
                merged_source_local_ids.append(formatted_line[7])
                source1_input_seq_mask.append(formatted_line[8])
                source2_input_seq_mask.append(formatted_line[9])
                source1_input_seq_len.append(formatted_line[10])
                source2_input_seq_len.append(formatted_line[11])
                merged_source_local_mask.append(formatted_line[12])
                merged_source_global_ids.append(formatted_line[17])
                
                if with_target:
                    target_word_list.append(formatted_line[13])
                    target_words_ids.append(formatted_line[14])
                    target_chars_ids.append(formatted_line[15])
                    target_seq_mask.append(formatted_line[16])
                else:
                    target_word_list = target_words_ids = target_chars_ids = target_seq_mask = None
            
            single_batch_data = {"source1_input_words_ids": torch.LongTensor(source1_input_words_ids).to(device), 
                                 "source1_input_chars_ids": torch.LongTensor(source1_input_chars_ids).to(device), 
                                 "source2_input_words_ids": torch.LongTensor(source2_input_words_ids).to(device), 
                                 "source2_input_chars_ids": torch.LongTensor(source2_input_chars_ids).to(device), 
                                 "source1_local_words_ids": torch.LongTensor(source1_local_words_ids).to(device), 
                                 "source2_local_words_ids": torch.LongTensor(source2_local_words_ids).to(device), 
                                 "merged_source_word_list": array(merged_source_word_list), # 不涉及到计算，所以放在CPU中。
                                 "merged_source_local_ids": torch.LongTensor(merged_source_local_ids).to(device), 
                                 "source1_input_seq_len":   torch.LongTensor(source1_input_seq_len).to(device), 
                                 "source2_input_seq_len":   torch.LongTensor(source2_input_seq_len).to(device), 
                                 "source1_input_seq_mask":  torch.LongTensor(source1_input_seq_mask).to(device), 
                                 "source2_input_seq_mask":  torch.LongTensor(source2_input_seq_mask).to(device), 
                                 "merged_source_local_mask": torch.LongTensor(merged_source_local_mask).to(device), 
                                 "merged_source_global_ids": torch.LongTensor(merged_source_global_ids).to(device), 
                                 "target_words_ids": None if target_words_ids is None else torch.LongTensor(target_words_ids).to(device),
                                 "target_chars_ids": None if target_chars_ids is None else torch.LongTensor(target_chars_ids).to(device),
                                 "target_seq_mask":  None if target_seq_mask is  None else torch.LongTensor(target_seq_mask).to(device),
                                 "target_word_list": target_word_list  #不涉及到计算，主要用于评估BLEU值，所以放在CPU中。
                                }
                                
            all_batch_data.append(single_batch_data)
            #yield single_batch_data

        return all_batch_data





class TestDataUtils(DataUtils):
    """docstring for testDataUtils"""

    def __init__(self, args, word2index, char2index):
        #super(testDataUtils, self).__init__()
        self.word2index = word2index
        self.char2index = char2index
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.index2char = {v:k for k,v in self.char2index.items()}

        self.max_seq_len = args.max_seq_len
        self.max_token_len = args.max_token_len
        
        self.PAD_idx = args.PAD_idx
        self.BOS_idx = args.BOS_idx
        self.EOS_idx = args.EOS_idx
        self.OOV_idx = args.OOV_idx
        self.PAD_token = args.PAD_token
        self.BOS_token = args.BOS_token
        self.EOS_token = args.EOS_token
        self.OOV_token  =args.OOV_token
        
        self.pattern = re.compile("[^\u4E00-\u9FA5]+")
        self.filter_fun = lambda word: len(word) >1 and word != "".join(re.findall(self.pattern,word))



    def get_batch_formatted_test_data(self, raw_test_data, with_target=False, batch_size=50, device=torch.device("cpu")):

        formatted_lines = self.obtain_formatted_lines(raw_test_data, with_target=with_target)
        batch_formatted_data = self.get_batch_data(formatted_lines, with_target, batch_size=batch_size, device=device)

        return batch_formatted_data







if __name__ == "__main__":
    from config import config
    args = config()
    args.train_data_path = "../data/train.dat"
    args.valid_data_path = "../data/valid.dat"
    args.test_data_path  = "../data/test.dat"

    data_loader = DataUtils(args)
    data_loader.construct_dict_v2()
    train_data, valid_data, test_data = data_loader.obtain_data()

    word2index, index2word, char2index, index2char = data_loader.word2index, data_loader.index2word, \
                                                     data_loader.char2index, data_loader.index2char

    test_data_utils = testDataUtils(args, word2index, index2word, char2index, index2char)
    print(test_data_utils.get_batch_formatted_test_data([[['哈哈'],['好的']]]), torch.device("cpu"))

    print("Train data num: ", len(train_data),"\n")

    print("source1_input_words_ids:\n", train_data[0][0], "\n\n")
    print("source1_input_chars_ids:\n", train_data[0][1], "\n\n")
    print("source2_input_words_ids:\n", train_data[0][2], "\n\n")
    print("source2_input_chars_ids:\n", train_data[0][3], "\n\n")
    print("merged_source_word_list:\n", train_data[0][4], "\n\n")
    print("merged_source_local_ids:\n", train_data[0][5], "\n\n")
    print("source1_local_words_ids:\n", train_data[0][6], "\n\n")
    print("source2_local_words_ids:\n", train_data[0][7], "\n\n")
    print("target_words_ids:\n", train_data[0][8], "\n\n")
    print("target_chars_ids:\n", train_data[0][9], "\n\n")



