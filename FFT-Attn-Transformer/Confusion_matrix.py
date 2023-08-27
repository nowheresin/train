import matplotlib.pyplot as plt
import numpy as np
import itertools
# from torch import tensor
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = np.sum(cm) / 6
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show(matrix):
    cnf_matrix = np.matrix(matrix)
    class_names = ['good_signal', 'abnormal_signal', 'disease_signal']
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')  # 绘制混淆矩阵
    np.set_printoptions(precision=2)
    Accaracy = (cnf_matrix[1, 1] + cnf_matrix[0, 0] + cnf_matrix[2, 2]) / (cnf_matrix.sum())

    try:
        Precision_good = cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[1, 0] + cnf_matrix[2, 0])
    except:
        Precision_good  = 0
    try:
        Recall_good = cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[0, 1] + cnf_matrix[0, 2])
    except:
        Recall_good  = 0

    try:
        Precision_bad = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1] + cnf_matrix[2, 1])
    except:
        Precision_bad = 0
    try:
        Recall_bad = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[1, 0] + cnf_matrix[1, 2])
    except:
        Recall_bad = 0

    try:
        Precision_disease = cnf_matrix[2, 2] / (cnf_matrix[2, 2] + cnf_matrix[0, 2] + cnf_matrix[1, 2])
    except:
        Precision_disease = 0
    try:
        Recall_disease = cnf_matrix[2, 2] / (cnf_matrix[2, 2] + cnf_matrix[2, 0] + cnf_matrix[2, 1])
    except:
        Recall_disease = 0

    """
    print('Accaracy:', Accaracy)
    print('Precision_good:', Precision_good)
    print('Recall_good:', Recall_good)
    print('Precision_bad:', Precision_bad)
    print('Recall_bad:', Recall_bad)
    """
    try:
        F1_good = (2 * Precision_good * Recall_good) / (Precision_good + Recall_good)
    except:
        F1_good = 0
    try:
        F1_bad  = (2 * Precision_bad * Recall_bad) / (Precision_bad + Recall_bad)
    except:
        F1_bad = 0
    try:
        F1_disease  = (2 * Precision_disease * Recall_disease) / (Precision_disease + Recall_disease)
    except:
        F1_disease = 0

    """
    print('F1_good:', F1_good)
    print('F1_bad:', F1_bad)
    """

    try:
        score = (np.mean([F1_good, F1_bad, F1_disease]))**2
    except:
        score = 0

    """
    print('score:', score)
    """

    # print('Specificity:', cnf_matrix[0, 0] / (cnf_matrix[0, 1] + cnf_matrix[0, 0]))
    # plt.show()

    return Accaracy, Precision_good, Recall_good, Precision_bad, Recall_bad, Precision_disease, \
        Recall_disease, F1_good, F1_bad, F1_disease, score

# show([[320, 4, 5], [43, 161, 22], [7, 2, 200]])