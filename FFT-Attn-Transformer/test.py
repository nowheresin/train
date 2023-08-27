import numpy as np

from datasets import *
import argparse
import xlwt
import os.path
import Confusion_matrix
import matplotlib.pyplot as plt

import time

def evaluate(model, enc_input, start_symbol):
    # Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    enc_outputs, enc_self_attns = model.Encoder(enc_input)
    dec_input = torch.zeros(1, tgt_len).type_as(enc_input.data)
    next_symbol = start_symbol
    for i in range(0, tgt_len):
        dec_input[0][i] = next_symbol

        dec_outputs, _, _ = model.Decoder(dec_input, enc_input, enc_outputs)

        projected = model.projection(dec_outputs)
        # print("projected=",projected.squeeze(0))
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # print("prob=",prob)
        next_word = prob.data[i]
        # print(next_word)
        next_symbol = next_word.item()
        # print(next_symbol)
    return dec_input

if __name__ == "__main__":

    model_path = 'model'
    trial_path = 'trials'

    error_workbook = xlwt.Workbook()
    error_01 = error_workbook.add_sheet('good_to_bad_error')
    error_02 = error_workbook.add_sheet('good_to_disease_error')
    error_10 = error_workbook.add_sheet('bad_to_good_error')
    error_12 = error_workbook.add_sheet('bad_to_disease_error')
    error_20 = error_workbook.add_sheet('disease_to_good_error')
    error_21 = error_workbook.add_sheet('disease_to_bad_error')
    error_else = error_workbook.add_sheet('else_error')

    indx_hang = 0
    indx_lie = 0

    for model_name in os.listdir(model_path):
        begin_time = time.time()
        model = model_path + '/' + model_name
        print(model)

        [counts_00, counts_01, counts_02] = [0, 0, 0]
        [counts_10, counts_11, counts_12] = [0, 0, 0]
        [counts_20, counts_21, counts_22] = [0, 0, 0]
        counts_else = 0

        model = torch.load(model)
        indx_hang = 0


        for name in os.listdir(trial_path):
            enc_input = xshow(trial_path + '/' + name, 1, 1200)
            enc_inputs = torch.LongTensor((enc_input)[0])

            predict_dec_input = evaluate(model, enc_inputs.view(1, -1).cuda(), start_symbol=1)
            predict, _, _, _ = model(enc_inputs.view(1, -1).cuda(), predict_dec_input)
            predict = predict.data.max(1, keepdim=True)[1]

            aa = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
            bb = {aa[key]: key for key in aa}  # 把目标字典转换成 索引：字的形式

            print('%-47s' % name, end=':')
            print([bb[n.item()] for n in predict.squeeze()])

            indx_lie = 1
            outputs = []
            for n in predict.squeeze():
                output = int(bb[n.item()])
                indx_lie = indx_lie + 1
                outputs.append(output)

            indx_hang = indx_hang + 1
            if "good" in name:
                if sum(outputs) / len(outputs) == 3:
                    counts_00 += 1
                elif sum(outputs) / len(outputs) == 4:
                    error_01.write(counts_01, 0, name)
                    counts_01 += 1
                elif sum(outputs) / len(outputs) == 5:
                    error_02.write(counts_02, 0, name)
                    counts_02 += 1
                else:
                    error_else.write(counts_else, 0, name)
                    counts_else += 1
            if "bad" in name:
                if sum(outputs) / len(outputs) == 4:
                    counts_11 += 1
                elif sum(outputs) / len(outputs) == 3:
                    error_10.write(counts_10, 0, name)
                    counts_10 += 1
                elif sum(outputs) / len(outputs) == 5:
                    error_12.write(counts_12, 0, name)
                    counts_12 += 1
                else:
                    error_else.write(counts_else, 0, name)
                    counts_else += 1
            if "disease" in name:
                if sum(outputs) / len(outputs) == 5:
                    counts_22 += 1
                elif sum(outputs) / len(outputs) == 3:
                    error_20.write(counts_20, 0, name)
                    counts_20 += 1
                elif sum(outputs) / len(outputs) == 4:
                    error_21.write(counts_21, 0, name)
                    counts_21 += 1
                else:
                    error_else.write(counts_else, 0, name)
                    counts_else += 1

        matrix = [[counts_00, counts_01, counts_02], [counts_10, counts_11, counts_12], [counts_20, counts_21, counts_22]]
        Confusion_matrix.show(matrix)
        plt.savefig(model_name[:-4] + '.png', dpi=300, bbox_inches="tight")

        end_time = time.time()
        run_time = round(end_time - begin_time)
        print(run_time / (
                    counts_00 + counts_01 + counts_02 + counts_10 + counts_11 + counts_12 + counts_20 + counts_21 + counts_22 + counts_else))

    error_workbook.save('error.xls')




