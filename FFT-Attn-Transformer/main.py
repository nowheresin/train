# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer import Transformer
import xlwt
import Predict_output
import Confusion_matrix

import matplotlib.pyplot as plt
import time


if __name__ == "__main__":

    begin_time = time.time()

    model_workbook = xlwt.Workbook()
    model_worksheet = model_workbook.add_sheet("Evaluate the model")
    category = ["Accaracy", "Precision_good", "Recall_good", "Precision_bad", "Recall_bad", "Precision_disease",
                "Recall_disease", "F1_good", "F1_bad", "F1_disease", "score"]
    for model_hang in range(len(category)):
        model_worksheet.write(0, model_hang, category[model_hang])

    GE_workbook = xlwt.Workbook()
    GE_worksheet = GE_workbook.add_sheet("GE")
    GE_category = ["var", "bias", "GE"]
    for GE_hang in range(len(GE_category)):
        GE_worksheet.write(0, GE_hang, GE_category[GE_hang])

    enc_inputs, dec_inputs, dec_outputs = make_data()
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 8, True)

    model = Transformer().cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)
    alf = 5e-5
    end_times = 0
    delta_loss_lists = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    best_loss = 10
    best_epoch = 0
    past_epoch_loss = 10


    for epoch in range(20):
        good_list = []
        bad_list = []
        disease_list = []

        loss_list = []

        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
            # dec_inputs : [batch_size, tgt_len]
            # dec_outputs: [batch_size, tgt_len]
            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            # print(len(enc_self_attns))   # 6
            # print(len(enc_self_attns[0]))   # batch_size
            # print(len(enc_self_attns[0][0]))   # 8
            # print(len(enc_self_attns[0][0][0]))   # 1200
            # print(len(enc_self_attns[0][0][0][0]))   # 1200

            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))

            good, bad, disease = Predict_output.predict(outputs, dec_outputs)
            good_list = good_list + good
            bad_list = bad_list + bad
            disease_list = disease_list + disease

            loss_list.append(loss)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = sum(loss_list) / len(loss_list)

        if epoch_loss < best_loss:
            best_model = model
            best_loss = epoch_loss
            best_epoch = epoch
            torch.save(best_model, 'best_model.pth')

        delta_loss = abs(epoch_loss - past_epoch_loss)
        delta_loss_lists[epoch % 10] = delta_loss

        counts = [1 for data in delta_loss_lists if data <= alf]
        if sum(counts) >= 7:
            end_times += 1
        else:
            end_times = 0

        past_epoch_loss = epoch_loss

        position_00, position_01, position_02 = Predict_output.prdicted_label(good_list)
        position_10, position_11, position_12 = Predict_output.prdicted_label(bad_list)
        position_20, position_21, position_22 = Predict_output.prdicted_label(disease_list)

        matrix = [[position_00, position_01, position_02], [position_10, position_11, position_12], [
            position_20, position_21, position_22]]

        Accaracy, Precision_good, Recall_good, Precision_bad, Recall_bad, Precision_disease, Recall_disease, \
            F1_good, F1_bad, F1_disease, score = Confusion_matrix.show(matrix)

        for model_hang in range(len(category)):
            model_worksheet.write(epoch + 1, model_hang, [Accaracy, Precision_good, Recall_good, Precision_bad,
                                                          Recall_bad, Precision_disease, Recall_disease,
                                                          F1_good, F1_bad, F1_disease, score][model_hang])

        good_means_list = Predict_output.mean_column(good_list)
        bad_means_list = Predict_output.mean_column(bad_list)
        disease_means_list = Predict_output.mean_column(disease_list)

        good_var = Predict_output.var_means(good_list, good_means_list)
        bad_var = Predict_output.var_means(bad_list, bad_means_list)
        disease_var = Predict_output.var_means(disease_list, disease_means_list)

        good_bias = Predict_output.bias_means(good_means_list, true_list=[3, 3, 3, 3, 2])
        bad_bias = Predict_output.bias_means(bad_means_list, true_list=[4, 4, 4, 4, 2])
        disease_bias = Predict_output.bias_means(disease_means_list, true_list=[5, 5, 5, 5, 2])

        good_GE = Predict_output.generalization_error(good_bias, good_var)
        bad_GE = Predict_output.generalization_error(bad_bias, bad_var)
        disease_GE = Predict_output.generalization_error(disease_bias, disease_var)

        for GE_hang in range(len(GE_category)):
            GE_worksheet.write(epoch + 1, GE_hang, [good_var + bad_var + disease_var, good_bias + bad_bias +
                                                    disease_bias, good_GE + bad_GE + disease_GE][GE_hang])

        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(epoch_loss), end='  ')

        print("delta_loss= ", '%-11s' % (float('{:.6f}'.format(delta_loss))), end='')

        end_time = time.time()
        run_time = end_time - begin_time
        hours = run_time // 3600
        minutes = run_time // 60 - hours * 60
        seconds = run_time - minutes * 60 - hours * 3600
        print("run_time= ", '%02d' % (hours), ':', '%02d' % (minutes), ':', '%02d' % (seconds))

        if end_times > 9:
            break

    print("run " + '%04d' % (epoch + 1) + " times, save " + '%04d' % (best_epoch + 1) + " as best model")

    model_workbook.save('Evaluate the model' + '_%04d' % (epoch + 1) + '.xls')
    GE_workbook.save('GE' + '_%04d' % (epoch + 1) + '.xls')

    torch.save(model, 'last_model.pth')

    plt.savefig('%04d' % (epoch + 1) + '.png', dpi=300, bbox_inches="tight")