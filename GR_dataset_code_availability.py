# generate datasets of the five static gestures

import pandas as pd
import scipy.fft
import scipy
from Fun_set import preprocess
import pickle
import numpy as np
from scipy import stats
import math
from statsmodels.tsa.ar_model import AutoReg
from scipy import signal


def preprocess_one(raw_signal, lowcut, highcut, samplerate, order):
    semisamplerate = samplerate*0.5
    low = lowcut / semisamplerate
    high = highcut / semisamplerate
    b, a = signal.butter(order, [low, high], btype='bandpass')
    preprocessed = signal.filtfilt(b, a, raw_signal)
    return np.array(preprocessed)

# mean amplitude spectrum (MAS)
def mas_func(data_fun):
    return sum(abs(scipy.fft.fft(data_fun)))


# sub-band energy (SBE)
def sub_eng_func(data_fun):
    # sub-bands
    sub_sig_1 = preprocess_one(data_fun, 80, 150, 1000, 4)
    sub_sig_2 = preprocess_one(data_fun, 150, 300, 1000, 4)
    sub_sig_3 = preprocess_one(data_fun, 300, 450, 1000, 4)

    # short-term energy
    sub_eng_1 = np.sum(sub_sig_1 * sub_sig_1)
    sub_eng_2 = np.sum(sub_sig_2 * sub_sig_2)
    sub_eng_3 = np.sum(sub_sig_3 * sub_sig_3)

    return np.vstack((sub_eng_1, sub_eng_2, sub_eng_3)).T


# 4-order autoregression coefficients (ARC)
def ar4_func(data_fun):
    ar4_model = AutoReg(data_fun, 4, old_names=False).fit()
    return ar4_model.params


# root mean square (RMS)
def rms_func(data_fun):
    return math.sqrt(np.mean([x ** 2 for x in data_fun]))


# waveform length (WL)
def wl_func(data_fun):
    return sum([abs(data_fun[i+1]-data_fun[i]) for i in range(len(data_fun)-1)])


# mean absolute value (MAV)
def mav_func(data_fun):
    return np.mean(np.abs(data_fun))


# zero crossings (ZC)
def zc_func(data_fun):
    return sum([1 if data_fun[i]*data_fun[i+1] < 0 else 0 for i in range(len(data_fun)-1)])


# slope sign changes (SSC)
def ssc_func(data_fun):
    return sum([1 if (data_fun[i+1]-data_fun[i])*(data_fun[i+1]-data_fun[i+2]) < 0 else 0 for i in range(len(data_fun)-2)])


# Pearson correlation
def Corr_fea_ext_func(semg):
    fea_corr_sig_cha = []
    for i in range(8):
        for j in range(i+1, 8):
            fea_corr_sig_cha.append(abs(np.corrcoef(semg[i], semg[j])[0, 1]))
    return np.array(fea_corr_sig_cha).reshape(1, 28)


# intra-channel feature extraction
def intra_cha_fea_ext_func(semg):
    fea_intra_sig_cha = []
    for i in range(8):
        f1 = rms_func(semg[i])  # root mean square (RMS) 1
        f2 = wl_func(semg[i])  # waveform length (WL) 2
        f3 = mav_func(semg[i])  # mean absolute value (MAV) 3
        f4 = zc_func(semg[i])  # zero crossings (ZC) 4
        f5 = ssc_func(semg[i])  # slope sign changes (SSC) 5
        f6 = ar4_func(semg[i])  # 4-order autoregression coefficients (ARC) 6 7 8 9 10
        f7 = mas_func(semg[i])  # mean amplitude spectrum (MAS) 11
        f8 = sub_eng_func(semg[i])  # sub-band energy (SBE) 12 13 14
        fea_intra_sig_cha.append([f1, f2, f3, f4, f5, f6[0], f6[1], f6[2], f6[3], f6[4], f7, f8[0, 0], f8[0, 1], f8[0, 2]])

    return np.array(fea_intra_sig_cha).reshape(1, 112)


# extract the activity frame
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

    res_thr = []
    res_thr_mode = []

    for j in range(len(sub_eng)):
        res_thr.append(EAD_gem_load.predict_thr(sub_eng[j, :]))
    res_thr = np.array(res_thr)

    for j in range(len(sub_eng) - 4):
        if stats.mode(res_thr[j:j + 5])[0] == 0:
            res_thr_mode.append(0)
        elif stats.mode(res_thr[j:j + 5])[0] == 1:
            if stats.mode(res_thr[j:j + 5])[1] > 3:
                res_thr_mode.append(1)
            else:
                res_thr_mode.append(0)

    return indices[np.where(np.array(res_thr_mode) == 1)]


# feature_extraction
def EAD_feat_ext(data, sig_index):
    feat_set = []
    for m in range(len(sig_index)):
        frame_data = []
        for j in range(8):
            frame_data.append(data[j][sig_index[m]])

        pre_data = preprocess(frame_data, 20, 450, 1000, 4)

        windown = np.hanning(128)
        pre_data_win = pre_data * windown

        # intra-channel feature
        fea_intra = intra_cha_fea_ext_func(pre_data_win)

        # inter-channel feature
        fea_Corr = Corr_fea_ext_func(pre_data_win)

        feat_set.append(list(np.concatenate((fea_intra, fea_Corr), axis=1)))

    return feat_set


# import CSV files
file_name_ctr = list()
for i in range(5):
    file_name_ctr.append('sub1_sti_mot1_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot1_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot2_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot3_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot4_d' + str(i + 1))


# load the EAD model
gmm_open = open('saves/' + 'CTR_EAD_sub1' + '.p', 'rb')
EAD_gem_load = pickle.load(gmm_open)

# generate static gestures dataset (for example, training set)
dataset_all = np.zeros((1, 141))
for i in range(15):
    path_name = r'C:\Users\research\prosthesis\data_collection\amp_data_csv'
    all_data = pd.read_csv(path_name + '\\' + file_name_ctr[i] + '.csv', header=None, skiprows=1)
    print(file_name_ctr[i])
    feature_set_sig = []
    for k in range(20):
        # extract Activity frames
        act_index = EAD_index(all_data.values[k*8], EAD_gem_load)

        # feature extraction
        feat_set = EAD_feat_ext(all_data.values[k*8:8*(k+1)], act_index)
        feature_set_sig.append(feat_set)
    feat_set_arr = feature_set_sig[0]
    for n in range(len(feature_set_sig)-1):
        feat_set_arr = np.concatenate((feat_set_arr, feature_set_sig[n+1]), axis=0)

    feat_set_arr = feat_set_arr.reshape((len(feat_set_arr), 140))

    # labelling
    label = np.ones((len(feat_set_arr), 1)) * int(i%5 + 1)

    dataset_all = np.concatenate((dataset_all, np.concatenate((feat_set_arr, label), axis=1)), axis=0)

# save the dataset
np.savetxt(r'dataset_txt\sub1_ctr_dataset_training.txt', dataset_all[1:dataset_all.shape[0], :])


# generate datasets for the EMG activation algorithm
dataset_train = np.loadtxt(r'dataset_txt\sub1_ctr_dataset_training.txt')

for i in range(len(dataset_train)):
    if dataset_train[i, -1] > 1:
        dataset_train[i, -1] = 0

np.savetxt(r'dataset_txt\sub1_ea_dataset_training.txt', dataset_train)
