# EMG Activity Detection (EAD) - testing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
from scipy import stats
from scipy import signal


# signal pre-processing
def preprocess(raw_signal, lowcut, highcut, samplerate, order):
    semisamplerate = samplerate*0.5
    low = lowcut / semisamplerate
    high = highcut / semisamplerate
    b, a = signal.butter(order, [low, high], btype='bandpass')
    preprocessed = [signal.filtfilt(b, a, raw_signal[i]) for i in range(len(raw_signal))]
    return np.array(preprocessed)


# feature extraction (shor-term energy of sub-bands)
def EAD_feat_ext(prepro_data):
    # framing
    wlen = 128  # length of a frame
    inc = 64  # moving length
    signal_length = len(prepro_data)
    nf = int((1.0 * signal_length - wlen + inc) / inc)  #


    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                             (wlen, 1)).T
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = prepro_data[indices]

    # pre-processing
    frames_pre = preprocess(frames, 20, 450, 1000, 4)

    # windowing
    windown = np.hanning(wlen)
    frames_win = frames_pre * windown

    # sub-bands
    sub_sig_1 = preprocess(frames_win, 80, 150, 1000, 4)
    sub_sig_2 = preprocess(frames_win, 150, 300, 1000, 4)
    sub_sig_3 = preprocess(frames_win, 300, 450, 1000, 4)

    # calculate the short-term energy of sub-bands
    sub_eng_1 = np.sum(sub_sig_1 * sub_sig_1, axis=1)
    sub_eng_2 = np.sum(sub_sig_2 * sub_sig_2, axis=1)
    sub_eng_3 = np.sum(sub_sig_3 * sub_sig_3, axis=1)

    return np.vstack((sub_eng_1, sub_eng_2, sub_eng_3)).T


# import the GMM model
gmm_open = open('saves/' + 'CTR_EAD_sub1' + '.p', 'rb')
EAD_gem_load = pickle.load(gmm_open)


# import CSV files
file_name_ctr = list()
for i in range(3):
    file_name_ctr.append('sub1_sti_mot1_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot1_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot2_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot3_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot4_d' + str(i + 1))


# EMG activity detection
for i in range(15):
    path_name = r'C:\Users\research\prosthesis\data_collection\amp_data_csv'
    all_data = pd.read_csv(path_name + '\\' + file_name_ctr[i] + '.csv', header=None, skiprows=1)
    for k in range(20):
        # feature extraction
        feature_set = EAD_feat_ext(all_data.values[k*8])
        # pre-processing
        prepro_data = preprocess(all_data.values[k*8:(k+1)*8], 20, 450, 1000, 4)
        # activity detection
        res = []
        for j in range(len(feature_set)):
            res.append(EAD_gem_load.predict(feature_set[j, :]))
        res = np.array(res)
        res_ori = res[:, 0]
        res_thr = res[:, 1]

        # 5-frame post-processing
        res_thr_mode = []
        for j in range(len(feature_set) - 4):
            res_thr_mode.append(stats.mode(res_thr[j:j + 5])[0])
        res_thr_mode = np.array(res_thr_mode)

        plt.figure()
        plt.subplot(311)
        plt.plot(prepro_data[0, :])

        plt.subplot(312)
        plt.plot(res_thr)

        plt.subplot(313)
        plt.plot(res_thr_mode)


