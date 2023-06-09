# generate handwriting dataset (testing set, for example)


import pandas as pd
import scipy.fft
import scipy
from keras.models import *
from Fun_set import preprocess, GEM
import pickle
import numpy as np
from scipy import stats, fft
import math
from sklearn import metrics
from statsmodels.tsa.ar_model import AutoReg
from scipy import signal
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import keras


# EMG activity detection
def EAD_index(data, EAD_gem_load):
    wlen = 128
    inc = 64
    signal_length = len(data)
    nf = int((1.0 * signal_length - wlen + inc) / inc)
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                             (wlen, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = data[indices]

    frames_pre = preprocess(frames, 20, 450, 1000, 4)

    windown = np.hanning(wlen)
    frames_win = frames_pre * windown

    sub_sig_1 = preprocess(frames_win, 80, 150, 1000, 4)
    sub_sig_2 = preprocess(frames_win, 150, 300, 1000, 4)
    sub_sig_3 = preprocess(frames_win, 300, 450, 1000, 4)

    sub_eng_1 = np.sum(sub_sig_1 * sub_sig_1, axis=1)
    sub_eng_2 = np.sum(sub_sig_2 * sub_sig_2, axis=1)
    sub_eng_3 = np.sum(sub_sig_3 * sub_sig_3, axis=1)
    sub_eng = np.vstack((sub_eng_1, sub_eng_2, sub_eng_3)).T

    res = []
    for j in range(len(sub_eng)):
        res.append(EAD_gem_load.predict(sub_eng[j, :]))
    res = np.array(res)
    res_thr = res[:, 1]

    res_thr_mode_3 = []
    for j in range(len(sub_eng) - 4):
        res_thr_mode_3.append(stats.mode(res_thr[j:j + 5])[0])
    res_thr_mode_3 = np.array(res_thr_mode_3)

    # the start point and the end point of the activity
    if min(res_thr_mode_3) != 0:
        act_index = np.where(res_thr_mode_3 == 1)
        start_point = act_index[0][0] * 64
        end_point = (act_index[0][-1] + 5) * 64 + 128
    else:
        act_index = np.where(res_thr_mode_3 == 1)
        start_point = act_index[0][0] * 64
        end_point = (act_index[0][-1]) * 64 + 128

    return start_point, end_point


# feature extraction - EMG spectrogram
def EAD_tf_ext(data, start_p, end_p, fftlen=64):
    ead_tf = list()
    frame_data = []
    for j in range(8):
        frame_data.append(data[j][start_p:end_p])

    pre_data = preprocess(frame_data, 20, 450, 1000, 4)
    windown = np.hanning(end_p - start_p)
    pre_data_win = pre_data * windown

    for j in range(8):
        S = librosa.stft(pre_data_win[j, :], n_fft=fftlen, window='hann')
        ead_tf.append(np.log(np.abs(S)))

    return np.array(ead_tf)


# data augmentation - perturb the detection result
def thr_aug(start, end):
    # 判断 end 有没有超出范围
    if start >= 4000:
        end_aug = np.array([end, end-8, end-16, end-24, end-32, end-40, end-48,
                            end-56, end-64, end-72, end-80, end-88, end-96, end-104, end-112, end-120])
        start_aug = end_aug - 2000
    elif start <= 200:
        start_aug = np.array([start, start-8, start-16, start-24, start-32, start-40, start-48,
                              start-56, start-64, start+8, start+16, start+24, start+32, start+40,
                              start+48, start+56])
        end_aug = start_aug + 2000
    else:
        start_aug = np.array([start, start-8, start-16, start-24, start-32, start-40, start-48,
                              start-56, start-64, start-72, start-80, start-88, start-96, start-104, start-112, start-120])
        end_aug = start_aug + 2000
    return start_aug, end_aug


# label
cha2lab_dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
               'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
               'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
               '&': 27, '~': 28, '!': 29, '%': 30}

# import the testing handwriting content
with open('txt/test-text-ds.txt', encoding='utf-8') as file:
    c_text = file.read()
cont_text = c_text.split('\n')


# generate datasets
gmm_open = open('saves/' + 'CTR_EAD_sub2' + '.p', 'rb')
EAD_gem_load = pickle.load(gmm_open)
feat_set_all = np.zeros((1, 8, 33, 126))
label_all = np.zeros((1, 1))

# import the testing EMG signals
for z, item in enumerate(cont_text):
    path_name = r'C:\Users\research\prosthesis\data_collection\amp_data_csv'
    all_data = pd.read_csv(path_name + '\\' + 'sub2_hwr_d5_' + item + '.csv', header=None, skiprows=1)
    feature_set_sig = []
    label_sig = []
    print(item)
    m = 0
    print(item)
    for k, cha in enumerate(item):
        if cha == ' ':  # ignore the space
            pass
        else:
            # EMG activity detection
            start, end = EAD_index(all_data.values[k * 8], EAD_gem_load)
            # perform the data augmentation
            start_aug, end_aug = thr_aug(start, end)
            for aug in range(12):
                # feature extraction
                tf_fs_ori = EAD_tf_ext(all_data.values[k * 8:8 * (k + 1)], start_aug[aug], end_aug[aug])
                feature_set_sig.append(tf_fs_ori)
                label_sig.append(cha2lab_dic[cha])
                m = m + 1

    feat_set_arr = np.array(feature_set_sig).reshape((m, 8, 33, 126))

    # labeling
    label_sig = np.array(label_sig).reshape((len(label_sig), 1))
    label_all = np.concatenate((label_all, label_sig), axis=0)

    feat_set_all = np.concatenate((feat_set_all, feat_set_arr), axis=0)


np.save('hdw_sub2_feat_testing_aug.npy', feat_set_all[1:len(feat_set_all)])
np.save('hdw_sub2_label_testing_aug.npy', label_all[1:len(feat_set_all)])
