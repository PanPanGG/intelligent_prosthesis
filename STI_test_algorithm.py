"""
肌电激活 - 连体网络算法实现
"""
import keras.models
from keras import layers
from keras import Input
from keras.models import Model
import numpy as np
import random
import sklearn.preprocessing as sp
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from joblib import load, dump
from keras.models import *

# 打印指标
def plot_cm_acc_tpr_fpr(test_targets, res):
    acc_sin = (res == test_targets).mean()
    confusion_m = np.array(confusion_matrix(test_targets, res))
    print(str(k) + ':' + '\n')
    print('confusion_matrix : \n', confusion_m)  # 纵轴是真实标签，横轴是预测标签
    print('ACC : \n', ("%.2f"%(acc_sin*100)))
    print('TPR : \n', ("%.4f"%(confusion_m[1, 1] / (confusion_m[1, 1] + confusion_m[1, 0]))))
    print('FPR : \n', ("%.4f"%(confusion_m[0, 1] / (confusion_m[0, 1] + confusion_m[0, 0]))))
    # C = confusion_matrix(test_targets, res)
    # plt.matshow(C, cmap=plt.cm.GnBu)  # 根据最下面的图按自己需求更改颜色
    # for i in range(len(C)):
    #     for j in range(len(C)):
    #         plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    #
    # # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。
    #
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    return None


# 连体网络 训练数据集 制作：left_data, right_data, targets
def Siamese_gen_train_ds(data):

    # 归一化
    std_scale = sp.StandardScaler().fit(data)
    data_std = std_scale.transform(data)

    pos_data = data_std[:, 0:140][np.where(data[:, -1] == 1)]
    neg_data = data_std[:, 0:140][np.where(data[:, -1] == 0)]

    temp = len(pos_data[:, 1])

    # 均衡样本集合
    left_data_1_1_1 = pos_data[0:int(temp/2), :]
    left_data_1_1_2 = pos_data[0:int(temp/2), :]
    left_data_1_0 = pos_data[0:int(temp/2)*2, :]
    # left_data_1 = pos_data[9:10, :]

    right_data_1_1_1 = pos_data[int(temp/2):int(temp/2)*2, :]
    right_data_1_1_2 = pos_data[int(temp/2)*2-1:int(temp/2)-1:-1, :]

    # 这个随机结果需要重复多次，取统计值 / 置信区间
    rd = list(range(len(neg_data)))
    random.shuffle(rd)
    right_data_1_0 = neg_data[rd[0: int(temp/2)*2]]

    left_data = np.concatenate((left_data_1_1_1, left_data_1_1_2, left_data_1_0), axis=0)
    right_data = np.concatenate((right_data_1_1_1, right_data_1_1_2, right_data_1_0), axis=0)

    # left_data = np.repeat(left_data_1, len(right_data), axis=0)

    targets = np.concatenate((np.ones((int(temp/2)*2, 1)), np.zeros((int(temp/2)*2, 1))), axis=0)

    return left_data.reshape((len(left_data), 1, 140)), right_data.reshape((len(right_data), 1, 140)), targets


# 连体网络 测试数据集 制作：left_data, right_data, targets
def Siamese_gen_test_ds(traindata, testdata, k):
    # 归一化
    std_scale = sp.StandardScaler().fit(traindata)
    traindata_std = std_scale.transform(traindata)

    # std_scale = sp.StandardScaler().fit(testdata)
    testdata_std = std_scale.transform(testdata)

    pos_traindata = traindata_std[:, 0:140][np.where(traindata[:, -1] == 1)]

    pos_testdata = testdata_std[:, 0:140][np.where(testdata[:, -1] == 1)]
    neg_testdata = testdata_std[:, 0:140][np.where(testdata[:, -1] == 0)]

    temp1 = len(pos_testdata[:, 1])

    # left 模板 - 训练数据
    left_data_1 = pos_traindata[k:k+1, :]

    # right 新样本 - 测试数据
    right_data_1 = pos_testdata

    # 这个随机结果需要重复多次，取统计值 / 置信区间
    temp2 = int(temp1 / 2) * 2
    rd = list(range(len(neg_testdata)))
    random.shuffle(rd)
    right_data_0 = neg_testdata[rd[0: temp2]]

    left_data = np.repeat(left_data_1, temp1 + temp2, axis=0)  # 左边模板数据都是一样的，因此就是随机选择一个模板，然后整体测试该模板的平均准确率
    right_data = np.concatenate((right_data_1, right_data_0), axis=0)
    targets = np.concatenate((np.ones((temp1, 1)), np.zeros((temp2, 1))), axis=0)

    return left_data.reshape((len(left_data), 1, 140)), right_data.reshape((len(right_data), 1, 140)), targets, temp1


dataset_train = np.loadtxt('dataset_txt/sub3_sti_dataset_training.txt')
dataset_valid = np.loadtxt('dataset_txt/sub3_sti_dataset_validating.txt')
dataset_test = np.loadtxt('dataset_txt/sub3_sti_dataset_testing.txt')


train_left_data, train_right_data, train_targets = Siamese_gen_train_ds(dataset_train)


# 随机打乱顺序
rd = list(range(len(train_targets)))
random.shuffle(rd)
train_left_data = train_left_data[rd]
train_right_data = train_right_data[rd]
train_targets = train_targets[rd]
#
# # 模型设计及训练
# lstm = layers.LSTM(64)
# lstm = layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5)
# # lstm = layers.Bidirectional(layers.LSTM(64, activation='relu', dropout=0.5, recurrent_dropout=0.5))
#
# left_input = Input(shape=(1, 64))
# left_output = lstm(left_input)
#
# right_input = Input(shape=(1, 64))
# right_output = lstm(right_input)
#
# merged = layers.concatenate([left_output, right_output], axis=-1)
# predictions = layers.Dense(1, activation='sigmoid')(merged)
#
# model = Model([left_input, right_input], predictions)
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#
# callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=100),
#                   keras.callbacks.ModelCheckpoint(filepath='saves/STI_sub3_322_v1.h5',
#                   monitor='val_acc', save_best_only=True)]
# history = model.fit([train_left_data, train_right_data], train_targets, epochs=100, batch_size=128,
#                   callbacks=callbacks_list, validation_split = 0.3)


# 选择validate数据进行模板选择

# # 训练集d1-d3，调参d4-d5

test_left_data, test_right_data, test_targets, temp = Siamese_gen_test_ds(dataset_train, dataset_valid, 5)

rd = list(range(temp-1))
random.shuffle(rd)
acc = np.zeros((200, 1))
res = np.copy(test_targets)
thr_all = np.zeros((200, 1))

# 加载模型
model_load = load_model("saves\STI_sub3_322_v1.h5")
# 随机选择200个模板

for k in range(200):
    test_left_data, test_right_data, test_targets, temp = Siamese_gen_test_ds(dataset_train, dataset_valid, rd[k])
    preds = model_load.predict([test_left_data, test_right_data])
    # 概率门限 [0.1: 0.9: 0.1]
    # 取中位数 test_right_data 一半为正一半为负
    thr = np.median(preds)
    thr_all[k] = thr

    for i, item in enumerate(preds):
        if item > thr:
            res[i] = 1
        else:
            res[i] = 0

    acc_sin = (res == test_targets).mean()
    # print(str(k) + ':' + str(acc_sin))
    acc[k] = acc_sin

print(np.mean(acc))  # 该200个模板的平均准确率
# #
# 选择准确率最高的5个
acc_higest_index = acc[:, 0].argsort()[195: 200]
acc_higest = acc[acc_higest_index]
thr_higest = thr_all[acc_higest_index]
print(acc_higest)
print(thr_higest)
temp_higest = np.array(rd)[acc_higest_index]
#
#
#
# # 保存模型
# # model.save('sti_model_3_1')
#
# # 加载模型
# model_load = keras.models.load_model('STI_sub3_322')
test_num = 3544
acc_sec = np.zeros((5, 1))
res_all = np.zeros((test_num, 5))
res = np.zeros((test_num, 1))
temp_data = []  # 保存模板数据
temp_thr = []  # 保存模板数据

for k in range(5):
    test_left_data, test_right_data, test_targets, temp = Siamese_gen_test_ds(dataset_train, dataset_test, temp_higest[k])
    preds = model_load.predict([test_left_data, test_right_data])

    temp_data.append(test_left_data[0])
    temp_thr.append(thr_higest[k])

    # 概率门限 [0.1: 0.9: 0.1]
    # 取中位数
    # thr = np.median(preds)
    for i, item in enumerate(preds):
        if item > thr_higest[k]:
            res[i] = 1
        else:
            res[i] = 0

    res_all[:, k] = res[:, 0]

    plot_cm_acc_tpr_fpr(test_targets, res)
    acc_sin = (res == test_targets).mean()
    acc_sec[k] = acc_sin

# 保存模板
# sti_std = sp.StandardScaler().fit(train_data[:, 0:168])
# dump(sti_std, 'sti_std.joblib')
# np.save('temp_data.npy', np.array(temp_data).reshape((5, 168)))  # 保存模板数据，需要进行归一化
# np.save('temp_thr.npy', np.array(temp_thr).reshape((5, 1)))  # 保存模板数据

print(np.mean(acc_sec))

# 集成学习，取众数
res_inte = np.zeros((test_num, 1))
for j in range(test_num):
    res_inte[j] = stats.mode(res_all[j, :])[0]

# acc_int = (res_inte == test_targets).mean()
# print(acc_int)
plot_cm_acc_tpr_fpr(test_targets, res_inte)

# 后处理，持续“OK”手势3帧 / 1s
res_mode = np.zeros((test_num-4, 1))
for j in range(len(res_inte) - 4):
    res_mode[j] = stats.mode(res_inte[j: j+5])[0]
res_mode = np.array(res_mode)

# target_mode = test_targets[3:-1]
target_mode = np.zeros((test_num-4, 1))
for j in range(len(res_inte) - 4):
    target_mode[j] = stats.mode(test_targets[j: j+5])[0]
target_mode = np.array(target_mode)

plot_cm_acc_tpr_fpr(target_mode, res_mode)
# acc_mode = (res_mode == target_mode).mean()
# print(acc_mode)