# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.utils.data as Data
from cope_with_data import *

import os.path

src_vocab_size = 1500
tgt_vocab_size = 6          # len(tgt_vocab)
src_len = 1200              # Encoder输入的最大长度
tgt_len = 5                 # Decoder输入输出最大长度


signal_good_path = 'signal_good'
signal_disease_path = 'signal_disease'
signal_bad_path = 'signal_bad'

def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []

    for name in os.listdir(signal_good_path):
        enc_input_long, dec_input, dec_output = show_good_signal(signal_good_path + '/' + name, 1, 1200)
        for i in range(1):
            enc_input = enc_input_long
            enc_inputs.extend(enc_input)
            dec_inputs.extend(dec_input)
            dec_outputs.extend(dec_output)

    for name in os.listdir(signal_bad_path):
        enc_input_long, dec_input, dec_output = show_bad_signal(signal_bad_path + '/' + name, 1, 1200)
        for i in range(1):
            enc_input = enc_input_long
            enc_inputs.extend(enc_input)
            dec_inputs.extend(dec_input)
            dec_outputs.extend(dec_output)

    for name in os.listdir(signal_disease_path):
        enc_input_long, dec_input, dec_output = show_disease_signal(signal_disease_path + '/' + name, 1, 1200)
        for i in range(1):
            enc_input = enc_input_long
            enc_inputs.extend(enc_input)
            dec_inputs.extend(dec_input)
            dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
