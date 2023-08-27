
import matplotlib.pyplot as plt

from cope_with_data import *

import os.path

from sklearn import manifold, datasets

plt.rc('font',family='Times New Roman')

X=[]
y=[]
for name in os.listdir('trials'):
    enc_input_long, _, _ = show_good_signal('trials/' + name, 1, 1200)
    X.extend(enc_input_long)
    if 'good' in name:
        y.append(0)
    if 'bad' in name:
        y.append(1)
    if 'disease' in name:
        y.append(2)


# #加载PCA算法，设置降维后主成分数目为2，即降为2维
# pca=PCA(n_components=100)

tsne = manifold.TSNE(n_components=2, init='pca', random_state=333)
reduced_X = tsne.fit_transform(np.array(X))

x_min, x_max = reduced_X.min(0), reduced_X.max(0)

X_norm = (reduced_X - x_min) / (x_max - x_min)
reduced_X = X_norm

red_x,red_y=[],[]
blue_x,blue_y=[],[]
green_x,green_y=[],[]

for i in range(len(reduced_X)):
    if y[i]==0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i]==1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
#4.降维后数据点的可视化
plt.figure(figsize=(5, 4.5))
plt.scatter(red_x,red_y,c='r',marker='x',label='Normal')
plt.scatter(blue_x,blue_y,c='b',marker='.',label='Abnormal')
plt.scatter(green_x,green_y,c='g',marker='D',label='Disease')

plt.ylabel('Dimension 2', size='12')
plt.xlabel('Dimension 1', size='12')

plt.rcParams.update({'font.size': 12})
plt.legend(loc='upper left')
plt.savefig("save.eps", dpi=300, bbox_inches="tight")

