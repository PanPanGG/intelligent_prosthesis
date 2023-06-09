# EMG Activity Detection (EAD) - training

import pandas as pd
import numpy as np
from scipy import signal
import copy
from collections import defaultdict
from sklearn.cluster import KMeans
import math
import pickle


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


# Gaussian Mixture Models (GMM)
class GEM:
    def __init__(self, K, maxstep=1000, epsilon=1e-3):
        self.maxstep = maxstep
        self.epsilon = epsilon
        self.K = K

        self.alpha = None
        self.mu = None
        self.sigma = None
        self.gamma_all_final = None

        self.D = None
        self.N = None

    def inin_param(self, data):
        # initialization
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.init_param_helper(data)
        return

    def init_param_helper(self, data):
        # KMeans
        KMEANS = KMeans(n_clusters=self.K).fit(data)
        clusters = defaultdict(list)
        for ind, label in enumerate(KMEANS.labels_):
            clusters[label].append(ind)
        mu = []
        alpha = []
        sigma = []
        for inds in clusters.values():
            partial_data = data[inds]
            mu.append(partial_data.mean(axis=0))
            alpha.append(len(inds) / self.N)
            sigma.append(np.cov(partial_data.T))
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        return

    def _phi(self, y, mu, sigma):

        s1 = 1.0 / math.sqrt(np.linalg.det(sigma)) # |Σ|
        s2 = np.linalg.inv(sigma)  # d*d Σ^(-1)
        delta = np.array([y - mu])  # 1*d
        return s1 * math.exp(-1.0 / 2 * delta @ s2 @ delta.T)

    def fit(self, data):
        # training
        self.inin_param(data)
        step = 0
        gamma_all_arr = None
        while step < self.maxstep:
            step += 1
            old_alpha = copy.copy(self.alpha)
            # E-step
            gamma_all = []
            for j in range(self.N):
                gamma_j = []

                for k in range(self.K):
                    gamma_j.append(self.alpha[k] * self._phi(data[j], self.mu[k], self.sigma[k]))

                s = sum(gamma_j)
                gamma_j = [item/s for item in gamma_j]
                gamma_all.append(gamma_j)

            gamma_all_arr = np.array(gamma_all)
            # M-step
            for k in range(self.K):
                gamma_k = gamma_all_arr[:, k]
                SUM = np.sum(gamma_k)
                # update weights
                self.alpha[k] = SUM / self.N
                # updated mean vector
                new_mu = sum([gamma * y for gamma, y in zip(gamma_k, data)]) / SUM  # 1*d
                self.mu[k] = new_mu
                # update covariance
                delta_ = data - new_mu   # n*d
                self.sigma[k] = sum([gamma * (np.outer(np.transpose([delta]), delta)) for gamma, delta in zip(gamma_k, delta_)]) / SUM  # d*n * n*d = d*d
            alpha_delta = self.alpha - old_alpha
            if np.linalg.norm(alpha_delta, 1) < self.epsilon:
                break
        self.gamma_all_final = gamma_all_arr
        return

    def predict_all(self):
        cluster = defaultdict(list)
        for j in range(self.N):
            max_ind = np.argmax(self.gamma_all_final[j])
            cluster[max_ind].append(j)
        return cluster

    # decision
    def predict(self, y):
        result = []
        for k in range(self.K):
            result.append(self.alpha[k] * self._phi(y, self.mu[k], self.sigma[k]))
        return np.argmax(result), 1 if result[0] < 1e-128 else 0, result


# import CSV files
file_name_ctr = list()
for i in range(3):
    file_name_ctr.append('sub1_sti_mot1_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot1_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot2_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot3_d' + str(i + 1))
    file_name_ctr.append('sub1_ctr_mot4_d' + str(i + 1))

# feature extraction
feature_set_all = []
for i in range(15):
    path_name = r'C:\Users\research\prosthesis\data_collection\amp_data_csv'
    all_data = pd.read_csv(path_name + '\\' + file_name_ctr[i] + '.csv', header=None, skiprows=1)
    print(file_name_ctr[i])
    feature_set_sig = []
    for k in range(20):
        # feature extraction on channel 1
        feature_set_sig.append(EAD_feat_ext(all_data.values[k*8]))

    feature_set_all.append(feature_set_sig)

# feature matrix
feature_set_all_arr = np.array(feature_set_all[0])
for i in range(14):
    feature_set_all_arr = np.vstack((feature_set_all_arr, feature_set_all[i+1]))

feature_set_all_arr = feature_set_all_arr.reshape((350*92, 3))

# GMM training
EAD_gem = GEM(K=2)
EAD_gem.fit(feature_set_all_arr)

# save the GMM model
gmm_save = open('saves/' + 'CTR_EAD_sub1' + '.p', 'wb')
pickle.dump(EAD_gem, gmm_save)
gmm_save.close()

