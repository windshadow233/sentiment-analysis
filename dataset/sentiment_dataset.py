from torch.utils.data import Dataset
import tqdm
import torch
import random
from sklearn.utils import shuffle
import re


class CLSDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_seq_len, data_regularization=False):

        self.data_regularization = data_regularization
        self.tokenizer = tokenizer
        # define max length
        self.max_seq_len = max_seq_len
        # directory of corpus dataset
        self.corpus_path = corpus_path
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.lines = [eval(line) for line in tqdm.tqdm(f, desc="Loading Dataset")]
            self.lines = shuffle(self.lines)
            self.corpus_lines = len(self.lines)

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        # 得到tokenize之后的文本和与之对应的情感分类
        text, label = self.get_text_and_label(item)

        if self.data_regularization:
            # 数据正则, 有10%的几率再次分句
            if random.random() < 0.1:
                split_spans = [i.span() for i in re.finditer("，|；|。|？|!", text)]
                if len(split_spans) != 0:
                    span_idx = random.randint(0, len(split_spans) - 1)
                    cut_position = split_spans[span_idx][1]
                    if random.random() < 0.5:
                        if len(text) - cut_position > 2:
                            text = text[cut_position:]
                        else:
                            text = text[:cut_position]
                    else:
                        if cut_position > 2:
                            text = text[:cut_position]
                        else:
                            text = text[cut_position:]

        text_input = self.tokenize_char(text)

        output = {"text_input": torch.tensor(text_input),
                  "label": torch.tensor([label])}
        return output

    def get_text_and_label(self, item):
        text = self.lines[item]["text"][:self.max_seq_len]
        label = self.lines[item]["label"]
        return text, label

    def tokenize_char(self, segments):
        return self.tokenizer.encode(segments)