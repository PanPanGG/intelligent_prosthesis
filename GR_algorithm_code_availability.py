# gesture recognition (GR) - training and testing

import keras.callbacks
import sklearn.preprocessing as sp
import lightgbm as lgbm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
import numpy as np
from scipy import stats
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.models import *
from keras.layers import Input, Dense


# post-processing
def post_mode_proc(y_test, y_pred, k):
    y_pred_post = np.zeros((len(y_test) - k-1))
    y_test_post = np.zeros((len(y_test)-k-1))

    for i in range(len(y_test)-k-1):
        y_pred_post[i] = stats.mode(y_pred[i:i+k])[0][0]
        y_test_post[i] = stats.mode(y_test[i:i+k])[0][0]
    return y_test_post, y_pred_post


# generate datasets for deep learning model
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


# import datasets
ctr_dataset_train = np.loadtxt('dataset_txt/sub1_ctr_dataset_training.txt')
ctr_dataset_valid = np.loadtxt('dataset_txt/sub1_ctr_dataset_validating.txt')
ctr_dataset_test = np.loadtxt('dataset_txt/sub1_ctr_dataset_testing.txt')

# generate training set and testing set
train_data_1 = ctr_dataset_train[:, 0:-1]
train_data_2 = ctr_dataset_valid[:, 0:-1]
train_data = np.concatenate((train_data_1, train_data_2), axis=0)

# normalization
std_scale = sp.StandardScaler().fit(train_data)
x_train_ = std_scale.transform(train_data)

y_train_1 = ctr_dataset_train[:, -1]
y_train_2 = ctr_dataset_valid[:, -1]
y_train_ = np.concatenate((y_train_1, y_train_2), axis=0)

test_data = ctr_dataset_test[:, 0:-1]
x_test_ = std_scale.transform(test_data)
y_test_ = ctr_dataset_test[:, -1]


# shallow learning: SVM, KNN, MLP, LGBM, XGBoost
# train and test shallow learning models
metric_all = np.array([[]])
for i in range(5):
    if i == 0:
        # SVM
        print("svm")
        svm = SVC(decision_function_shape='ovr')
        svm.fit(x_train_, y_train_)
        y_pred = svm.predict(x_test_)
    elif i == 1:
        # KNN
        print("knn")
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(x_train_, y_train_)
        y_pred = knn.predict(x_test_)
    elif i == 2:
        # MLP
        print("mlp")
        mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(200, 100, 100), random_state=1)
        mlp.fit(x_train_, y_train_)
        y_pred = mlp.predict(x_test_)
    elif i == 3:
        # LGBM
        print("lgbm")
        gbm = lgbm.LGBMClassifier()
        gbm.fit(x_train_, y_train_)
        y_pred = gbm.predict(x_test_)
    else:
        # XBGoost
        print("xgb")
        num_round = 10
        xgbo = xgb.XGBClassifier(max_depth=6, learning_rate=0.05, n_estimators=2000, objective='binary:logitraw' ,
                                 tree_method='gpu_hist', subsample=0.8, colsample_bytree=0.8, min_child_samples=3,
                                 eval_metric='auc', reg_lambda=0.5)
        xgbo.fit(x_train_, y_train_, eval_set=[(x_test_, y_test_)], eval_metric='auc', early_stopping_rounds=10,
                 verbose=False)
        y_pred = xgbo.predict(x_test_)

    y_test_post_3, y_pred_post_3 = post_mode_proc(y_test_, y_pred, 3) # 3-frame post-processing
    y_test_post_5, y_pred_post_5 = post_mode_proc(y_test_, y_pred, 5) # 5-frame post-processing
    accuracy1 = accuracy_score(y_test_, y_pred)
    accuracy2 = accuracy_score(y_test_post_3, y_pred_post_3)
    accuracy3 = accuracy_score(y_test_post_5, y_pred_post_5)
    accuracy = np.array([[accuracy1, accuracy2, accuracy3]])
    metric_all = np.column_stack((metric_all, accuracy))
    print("accuracy1: %.2f%%" % (accuracy1*100))
    print("accuracy2: %.2f%%" % (accuracy2*100))
    print("accuracy3: %.2f%%" % (accuracy3*100))

    confusion_m = np.array(confusion_matrix(y_test_, y_pred))
    print('confusion_matrix : \n', confusion_m)  # confusion matrix


# deep learning: Stacked-BiLSTM
# generate datasets for deep learning model
n_dims = 140
x_train = np.array(x_train_).reshape((len(x_train_), 1, n_dims))
y_train = np.array(y_train_).reshape((len(y_train_), 1))
x_train = x_train.reshape((len(x_train), n_dims))

x_train_step, y_train_step = gen_step_data(x_train, y_train)
x_train_step = x_train_step.reshape((len(x_train_step), 5, n_dims))

y_train_one_hot = to_categorical(y_train_step-1)

x_test = np.array(x_test_).reshape((len(x_test_), 1, n_dims))
y_test = np.array(y_test_).reshape((len(y_test_), 1))
x_test = x_test.reshape((len(x_test), n_dims))

x_test_step, y_test_step = gen_step_data(x_test, y_test)
x_test_step = x_test_step.reshape((len(x_test_step), 5, n_dims))

y_test_one_hot = to_categorical(y_test_step-1)
y_test_one_hot = y_test_one_hot.reshape((len(y_test_one_hot),  5))


# design and train a Stacked-BiLSTM
inputs = Input(shape=(None, n_dims))
x = layers.LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(inputs)
x = layers.BatchNormalization()(x)
x = layers.LSTM(64, activation='relu', dropout=0.5, recurrent_dropout=0.5, return_sequences=True)(x)
x = layers.BatchNormalization()(x)
x = layers.Bidirectional(layers.LSTM(64, activation='relu', dropout=0.5, recurrent_dropout=0.5))(x)
x = layers.BatchNormalization()(x)
output = Dense(5, activation='sigmoid')(x)
model_stebistalstm = Model(input=inputs, output=output)
# save the GR model
callbacks_list = [keras.callbacks.EarlyStopping(monitor='acc', patience=10),
                  keras.callbacks.ModelCheckpoint(filepath='saves/ctr_sub1.h5', monitor='val_acc', save_best_only=True)]


model_stebistalstm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
history_stebistalstm = model_stebistalstm.fit(x_train_step, y_train_one_hot, epochs=100, batch_size=256,
                                  callbacks=callbacks_list, validation_data=(x_test_step, y_test_one_hot))


# load the GR model
model_load = load_model("saves\ctr_sub3_52.h5")

# test the GR model
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
print('confusion_matrix : \n', confusion_m)