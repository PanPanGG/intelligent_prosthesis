# EMG Activation (EA) - training, validating, and testing

import keras.models
from keras import layers
import numpy as np
import random
import sklearn.preprocessing as sp
from scipy import stats
from sklearn.metrics import confusion_matrix
from keras.models import *


# print the confusion matrix
def plot_cm_acc_tpr_fpr(test_targets, res):
    acc_sin = (res == test_targets).mean()
    confusion_m = np.array(confusion_matrix(test_targets, res))
    print(str(k) + ':' + '\n')
    print('confusion_matrix : \n', confusion_m)  # 纵轴是真实标签，横轴是预测标签
    print('ACC : \n', ("%.2f"%(acc_sin*100)))
    print('TPR : \n', ("%.4f"%(confusion_m[1, 1] / (confusion_m[1, 1] + confusion_m[1, 0]))))
    print('FPR : \n', ("%.4f"%(confusion_m[0, 1] / (confusion_m[0, 1] + confusion_m[0, 0]))))
    return None


# generate training set：left_data, right_data, targets
def Siamese_gen_train_ds(data):

    # normalization
    std_scale = sp.StandardScaler().fit(data)
    data_std = std_scale.transform(data)

    pos_data = data_std[:, 0:140][np.where(data[:, -1] == 1)]
    neg_data = data_std[:, 0:140][np.where(data[:, -1] == 0)]

    temp = len(pos_data[:, 1])

    # balance the sample
    left_data_1_1_1 = pos_data[0:int(temp/2), :]
    left_data_1_1_2 = pos_data[0:int(temp/2), :]
    left_data_1_0 = pos_data[0:int(temp/2)*2, :]

    right_data_1_1_1 = pos_data[int(temp/2):int(temp/2)*2, :]
    right_data_1_1_2 = pos_data[int(temp/2)*2-1:int(temp/2)-1:-1, :]

    rd = list(range(len(neg_data)))
    random.shuffle(rd)
    right_data_1_0 = neg_data[rd[0: int(temp/2)*2]]

    left_data = np.concatenate((left_data_1_1_1, left_data_1_1_2, left_data_1_0), axis=0)
    right_data = np.concatenate((right_data_1_1_1, right_data_1_1_2, right_data_1_0), axis=0)

    targets = np.concatenate((np.ones((int(temp/2)*2, 1)), np.zeros((int(temp/2)*2, 1))), axis=0)

    return left_data.reshape((len(left_data), 1, 140)), right_data.reshape((len(right_data), 1, 140)), targets


# generate testing set：left_data, right_data, targets
def Siamese_gen_test_ds(traindata, testdata, k):
    # normalization
    std_scale = sp.StandardScaler().fit(traindata)
    traindata_std = std_scale.transform(traindata)

    testdata_std = std_scale.transform(testdata)

    pos_traindata = traindata_std[:, 0:140][np.where(traindata[:, -1] == 1)]

    pos_testdata = testdata_std[:, 0:140][np.where(testdata[:, -1] == 1)]
    neg_testdata = testdata_std[:, 0:140][np.where(testdata[:, -1] == 0)]

    temp1 = len(pos_testdata[:, 1])

    left_data_1 = pos_traindata[k:k+1, :]

    right_data_1 = pos_testdata

    temp2 = int(temp1 / 2) * 2
    rd = list(range(len(neg_testdata)))
    random.shuffle(rd)
    right_data_0 = neg_testdata[rd[0: temp2]]

    left_data = np.repeat(left_data_1, temp1 + temp2, axis=0)
    right_data = np.concatenate((right_data_1, right_data_0), axis=0)
    targets = np.concatenate((np.ones((temp1, 1)), np.zeros((temp2, 1))), axis=0)

    return left_data.reshape((len(left_data), 1, 140)), right_data.reshape((len(right_data), 1, 140)), targets, temp1


# import EA datasets
dataset_train = np.loadtxt('dataset_txt/sub1_ea_dataset_training.txt')
dataset_valid = np.loadtxt('dataset_txt/sub1_ea_dataset_validating.txt')
dataset_test = np.loadtxt('dataset_txt/sub1_ea_dataset_testing.txt')

# generate training set
train_left_data, train_right_data, train_targets = Siamese_gen_train_ds(dataset_train)


rd = list(range(len(train_targets)))
random.shuffle(rd)
train_left_data = train_left_data[rd]
train_right_data = train_right_data[rd]
train_targets = train_targets[rd]

# design and train a siamese network
lstm = layers.LSTM(64)
lstm = layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5)

left_input = Input(shape=(1, 64))
left_output = lstm(left_input)

right_input = Input(shape=(1, 64))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
predictions = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], predictions)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# save the EA model
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=100),
                  keras.callbacks.ModelCheckpoint(filepath='saves/EA_sub1.h5',
                  monitor='val_acc', save_best_only=True)]
history = model.fit([train_left_data, train_right_data], train_targets, epochs=100, batch_size=128,
                  callbacks=callbacks_list, validation_split=0.3)


# choose the template data from the validating set
test_left_data, test_right_data, test_targets, temp = Siamese_gen_test_ds(dataset_train, dataset_valid, 5)

rd = list(range(temp-1))
random.shuffle(rd)
acc = np.zeros((200, 1))
res = np.copy(test_targets)
thr_all = np.zeros((200, 1))


# load the EA model
model_load = load_model("saves\SEA_sub1.h5")

# randomly choose 200 templates
for k in range(200):
    test_left_data, test_right_data, test_targets, temp = Siamese_gen_test_ds(dataset_train, dataset_valid, rd[k])
    preds = model_load.predict([test_left_data, test_right_data])
    thr = np.median(preds)
    thr_all[k] = thr

    for i, item in enumerate(preds):
        if item > thr:
            res[i] = 1
        else:
            res[i] = 0

    acc_sin = (res == test_targets).mean()
    acc[k] = acc_sin

print(np.mean(acc))  # the average accuracy on the 200 templates

# choose the best 5 templates according to the accuray
acc_higest_index = acc[:, 0].argsort()[195: 200]
acc_higest = acc[acc_higest_index]
thr_higest = thr_all[acc_higest_index]
print(acc_higest)
print(thr_higest)
temp_higest = np.array(rd)[acc_higest_index]

# test the EA model
test_num = 3544
acc_sec = np.zeros((5, 1))
res_all = np.zeros((test_num, 5))
res = np.zeros((test_num, 1))
temp_data = []
temp_thr = []

# individual template judgment
for k in range(5):
    test_left_data, test_right_data, test_targets, temp = Siamese_gen_test_ds(dataset_train, dataset_test, temp_higest[k])
    preds = model_load.predict([test_left_data, test_right_data])

    temp_data.append(test_left_data[0])
    temp_thr.append(thr_higest[k])

    for i, item in enumerate(preds):
        if item > thr_higest[k]:
            res[i] = 1
        else:
            res[i] = 0

    res_all[:, k] = res[:, 0]

    plot_cm_acc_tpr_fpr(test_targets, res)
    acc_sin = (res == test_targets).mean()
    acc_sec[k] = acc_sin

print(np.mean(acc_sec))

# template ensemble
res_inte = np.zeros((test_num, 1))
for j in range(test_num):
    res_inte[j] = stats.mode(res_all[j, :])[0]
plot_cm_acc_tpr_fpr(test_targets, res_inte)

# 5-frame post-processing
res_mode = np.zeros((test_num-4, 1))
for j in range(len(res_inte) - 4):
    res_mode[j] = stats.mode(res_inte[j: j+5])[0]
res_mode = np.array(res_mode)

target_mode = np.zeros((test_num-4, 1))
for j in range(len(res_inte) - 4):
    target_mode[j] = stats.mode(test_targets[j: j+5])[0]
target_mode = np.array(target_mode)

plot_cm_acc_tpr_fpr(target_mode, res_mode)
