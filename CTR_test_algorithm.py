'''
日常控制手势分类
浅层学习
深度学习
优化
'''
import keras.callbacks
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sklearn.preprocessing as sp
import datetime, time
import lightgbm as lgbm
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, LeaveOneOut, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy import stats
from keras.optimizers import RMSprop


def post_mode_proc(y_test, y_pred, k):
    # 策略 众数
    y_pred_post = np.zeros((len(y_test) - k-1))
    y_test_post = np.zeros((len(y_test)-k-1))

    for i in range(len(y_test)-k-1):
        y_pred_post[i] = stats.mode(y_pred[i:i+k])[0][0]
        y_test_post[i] = stats.mode(y_test[i:i+k])[0][0]
    return y_test_post, y_pred_post


# 产生带step数据集
def gen_step_data(x, y):
    x_step = []
    y_step = []
    for i in range(5):
        temp_index = np.where(y == i+1)
        temp_data = x[temp_index[0], :]
        for k in range(len(temp_index[0]) - 4):
            temp_arr = temp_data[k: k+5]
            x_step.append(temp_arr)
            y_step.append(i+1)
    x_step = np.array(x_step)
    y_step = np.array(y_step).reshape((len(y_step), 1))
    return x_step, y_step

# 合并数据集：激活特征集 + 分类标签
# ctr_label = np.loadtxt('ctr_test_label.txt')
# sti_dataset = np.loadtxt('sti_dataset_test.txt')
# ctr_dataset = np.concatenate((sti_dataset[:, 0:-1], np.array(ctr_label).reshape((len(ctr_label), 1))), axis=1)
#
# np.savetxt('ctr_dataset_test.txt', ctr_dataset)


# 加载数据集
ctr_dataset_train = np.loadtxt('dataset_txt/sub3_ctr_dataset_training.txt')
ctr_dataset_valid = np.loadtxt('dataset_txt/sub3_ctr_dataset_validating.txt')
ctr_dataset_test = np.loadtxt('dataset_txt/sub3_ctr_dataset_testing.txt')

# 划分数据集
# 4 / 4 / 3
train_data_1 = ctr_dataset_train[:, 0:-1]
train_data_2 = ctr_dataset_valid[:, 0:-1]
train_data = np.concatenate((train_data_1, train_data_2), axis=0)

std_scale = sp.StandardScaler().fit(train_data)
x_train_ = std_scale.transform(train_data)

y_train_1 = ctr_dataset_train[:, -1]
y_train_2 = ctr_dataset_valid[:, -1]
y_train_ = np.concatenate((y_train_1, y_train_2), axis=0)

test_data = ctr_dataset_test[:, 0:-1]
# std_scale = sp.StandardScaler().fit(test_data)
x_test_ = std_scale.transform(test_data)
y_test_ = ctr_dataset_test[:, -1]
# dump(std_scale, 'ctr_std.joblib')
# z = load('ctr_std.joblib')

# t-SNE 可视化
# tsne = TSNE()
# X_embedded = tsne.fit_transform(x_train_)
# plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train_)
# cbar = plt.colorbar()
# cbar.set_label('手势类比')

# pca 降维
# pca = PCA(n_components=0.99)
# x_train_ = pca.fit_transform(x_train_)
# x_test_ = pca.transform(x_test_)
# from joblib import dump, load
# dump(pca, 'ctr_pca.joblib')
# z = load('ctr_pca.joblib')
# plt.scatter(x_train_[:, 1], x_train_[:, 3], c=y_train_)
# cbar = plt.colorbar()
# cbar.set_label('手势类比')

# 浅层学习
metric_all = np.array([[]])
# for i in range(5):
#     if i == 0:
#         # svm
#         print("svm")
#         svm = SVC(decision_function_shape='ovr')
#         svm.fit(x_train_, y_train_)
#         y_pred = svm.predict(x_test_)
#     elif i == 4:
#         # knn
#         print("knn")
#         knn = KNeighborsClassifier(n_neighbors=10)
#         knn.fit(x_train_, y_train_)
#         y_pred = knn.predict(x_test_)
#     elif i == 1:
#         # mlp
#         print("mlp")
#         mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 100, 100), random_state=1)
#         mlp.fit(x_train_, y_train_)
#         y_pred = mlp.predict(x_test_)
#     elif i == 5:
#         # rf
#         print("rf")
#         rf = RandomForestClassifier(max_depth=20, random_state=0, n_estimators=1999, oob_score=True)
#         rf.fit(x_train_, y_train_)
#         y_pred = rf.predict(x_test_)
#     elif i == 2:
#         # lgbm
#         print("lgbm")
#         gbm = lgbm.LGBMClassifier()
#         gbm.fit(x_train_, y_train_)
#         y_pred = gbm.predict(x_test_)
#     elif i == 3:
#         # xgboost
#         print("xgb")
#         num_round = 10 # 二分类：objective='binary:logitraw', 多分类：objective='binary:logitraw',
#         xgbo = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=2000, objective='binary:logitraw' ,
#                                  tree_method='gpu_hist', subsample=0.8, colsample_bytree=0.8, min_child_samples=3,
#                                  eval_metric='auc', reg_lambda=0.5)
#         xgbo.fit(x_train_, y_train_, eval_set=[(x_test_, y_test_)], eval_metric='auc', early_stopping_rounds=10,
#                  verbose=False)
#         y_pred = xgbo.predict(x_test_)
#     else:
#         # adaboost
#         adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=15, min_samples_split=30, min_samples_leaf=10), # adb 第一版参数
#                                  algorithm="SAMME", n_estimators=2000, learning_rate=0.5)
#         print("adaboost")
#         # adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8, min_samples_split=30, min_samples_leaf=10), # adb 第二版参数
#         #                          algorithm="SAMME", n_estimators=1000, learning_rate=0.5)
#         adb.fit(x_train_, y_train_)
#         y_pred = adb.predict(x_test_)
#
#     y_test_post_3, y_pred_post_3 = post_mode_proc(y_test_, y_pred, 3)
#     y_test_post_5, y_pred_post_5 = post_mode_proc(y_test_, y_pred, 5)
#     accuracy1 = accuracy_score(y_test_, y_pred)
#     accuracy2 = accuracy_score(y_test_post_3, y_pred_post_3)
#     accuracy3 = accuracy_score(y_test_post_5, y_pred_post_5)
#     accuracy = np.array([[accuracy1, accuracy2, accuracy3]])
#     metric_all = np.column_stack((metric_all, accuracy))
#     print("accuracy1: %.2f%%" % (accuracy1*100))
#     print("accuracy2: %.2f%%" % (accuracy2*100))
#     print("accuracy3: %.2f%%" % (accuracy3*100))
#
#     confusion_m = np.array(confusion_matrix(y_test_, y_pred))
#     print('confusion_matrix : \n', confusion_m)  # 纵轴是真实标签，横轴是预测标签


# 深度学习
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical

# n_dims = pca.n_components_
n_dims = 140
x_train = np.array(x_train_).reshape((len(x_train_), 1, n_dims))
y_train = np.array(y_train_).reshape((len(y_train_), 1))
x_train = x_train.reshape((len(x_train), n_dims))

x_train_step, y_train_step = gen_step_data(x_train, y_train)
x_train_step = x_train_step.reshape((len(x_train_step), 5, n_dims))

y_train_one_hot = to_categorical(y_train_step-1)
# y_train_one_hot = y_train_one_hot.reshape((len(y_train_one_hot),  5))

x_test = np.array(x_test_).reshape((len(x_test_), 1, n_dims))
y_test = np.array(y_test_).reshape((len(y_test_), 1))
x_test = x_test.reshape((len(x_test), n_dims))

x_test_step, y_test_step = gen_step_data(x_test, y_test)
x_test_step = x_test_step.reshape((len(x_test_step), 5, n_dims))

y_test_one_hot = to_categorical(y_test_step-1)
y_test_one_hot = y_test_one_hot.reshape((len(y_test_one_hot),  5))


# 深度学习 - 5 steps 双向堆叠LSTM 90% - 92% (最高有个93)
from keras.models import *
from keras.layers import Input, Dense
# inputs = Input(shape=(None, n_dims))
# x = layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(inputs)
# x = layers.BatchNormalization()(x)
# x = layers.LSTM(64, activation='relu', dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Bidirectional(layers.LSTM(64, activation='relu', dropout=0.5, recurrent_dropout=0.5))(x)
# x = layers.BatchNormalization()(x)
# output = Dense(5, activation='sigmoid')(x)
# model_stebistalstm = Model(input=inputs, output=output)
# callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=10),
#                   keras.callbacks.ModelCheckpoint(filepath='saves/mct_sub3_v2_111.h5', monitor='val_acc', save_best_only=True)]
#
#
# model_stebistalstm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# history_stebistalstm = model_stebistalstm.fit(x_train_step, y_train_one_hot, epochs=100, batch_size=256,
#                                               # callbacks=callbacks_list, validation_split=0.3)
#                                   callbacks=callbacks_list, validation_data=(x_test_step, y_test_one_hot))


# # 加载.H5 模型
model_load = load_model("saves\ctr_sub3_52.h5")
# model.summary()


pre_onehot = model_load.predict(x_test_step)
pre_fin = [np.argmax(x)+1 for x in pre_onehot]
pre_fin = np.array(pre_fin).reshape((len(pre_fin), 1))
acc_sin = (pre_fin == y_test_step).mean()


y_test_post_3, y_pred_post_3 = post_mode_proc(y_test_step, pre_fin, 3)
y_test_post_5, y_pred_post_5 = post_mode_proc(y_test_step, pre_fin, 5)
accuracy1 = accuracy_score(y_test_step, pre_fin)
accuracy2 = accuracy_score(y_test_post_3, y_pred_post_3)
accuracy3 = accuracy_score(y_test_post_5, y_pred_post_5)
accuracy = np.array([[accuracy1, accuracy2, accuracy3]])
metric_all = np.column_stack((metric_all, accuracy))
print("accuracy1: %.2f%%" % (accuracy1*100))
print("accuracy2: %.2f%%" % (accuracy2*100))
print("accuracy3: %.2f%%" % (accuracy3*100))

confusion_m = np.array(confusion_matrix(y_test_step, pre_fin))
print('confusion_matrix : \n', confusion_m) # 纵轴是真实标签，横轴是预测标签