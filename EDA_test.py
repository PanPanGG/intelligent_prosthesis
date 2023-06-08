# EAD test
# 最终测试结果

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import copy
from collections import defaultdict
from sklearn.cluster import KMeans
import math
import pickle
from scipy import stats


# 信号预处理
def preprocess(raw_signal, lowcut, highcut, samplerate, order):
    # preprocessed = copy.deepcopy(raw_signal)
    semisamplerate = samplerate*0.5
    low = lowcut / semisamplerate
    high = highcut / semisamplerate
    b, a = signal.butter(order, [low, high], btype='bandpass')
    preprocessed = [signal.filtfilt(b, a, raw_signal[i]) for i in range(len(raw_signal))]
    return np.array(preprocessed)


# 子带特征提取
def EAD_feat_ext(prepro_data):
    # 分帧
    wlen = 128  # 帧长
    inc = 64  # 帧移
    signal_length = len(prepro_data)
    nf = int((1.0 * signal_length - wlen + inc) / inc)  # 分帧数量，np.ceil 向上取整
    # pad_length = int((nf - 1) * inc + wlen)  # 所有帧加起来总的铺平后的长度
    # zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    # pad_signal = np.concatenate((prepro_data, zeros))  # 填补后的信号记为pad_signal

    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                             (wlen, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*wlen长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = prepro_data[indices]  # 得到帧信号

    frames_pre = preprocess(frames, 20, 450, 1000, 4)

    # 加窗
    windown = np.hanning(wlen)
    frames_win = frames_pre * windown

    # 子带分割
    sub_sig_1 = preprocess(frames_win, 80, 150, 1000, 4)
    sub_sig_2 = preprocess(frames_win, 150, 300, 1000, 4)
    sub_sig_3 = preprocess(frames_win, 300, 450, 1000, 4)

    # 子带能量
    sub_eng_1 = np.sum(sub_sig_1 * sub_sig_1, axis=1)
    sub_eng_2 = np.sum(sub_sig_2 * sub_sig_2, axis=1)
    sub_eng_3 = np.sum(sub_sig_3 * sub_sig_3, axis=1)

    return np.vstack((sub_eng_1, sub_eng_2, sub_eng_3)).T


# GMM_EM
class GEM:
    def __init__(self, K, maxstep=1000, epsilon=1e-3):
        self.maxstep = maxstep
        self.epsilon = epsilon
        self.K = K  # 混合模型中的分模型的个数

        self.alpha = None  # 每个分模型前系数
        self.mu = None  # 每个分模型的均值向量
        self.sigma = None  # 每个分模型的协方差
        self.gamma_all_final = None  # 存储最终的每个样本对分模型的响应度，用于最终的聚类

        self.D = None  # 输入数据的维度
        self.N = None  # 输入数据总量

    def inin_param(self, data):
        # 初始化参数
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.init_param_helper(data)
        return

    def init_param_helper(self, data):
        # KMeans初始化模型参数
        KMEANS = KMeans(n_clusters=self.K).fit(data)
        clusters = defaultdict(list)
        for ind, label in enumerate(KMEANS.labels_): # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中
            clusters[label].append(ind)
        mu = []
        alpha = []
        sigma = []
        for inds in clusters.values():
            partial_data = data[inds]
            mu.append(partial_data.mean(axis=0))  # 分模型的均值向量
            alpha.append(len(inds) / self.N)  # 权重
            sigma.append(np.cov(partial_data.T))  # 协方差,D个维度间的协方差 分母为n-1
        self.mu = np.array(mu)
        self.alpha = np.array(alpha)
        self.sigma = np.array(sigma)
        return

    def _phi(self, y, mu, sigma):
        # 获取分模型的概率, 将y带入多维高斯分布模型，得到概率值
        # np.linalg.det():矩阵求行列式
        # np.linalg.inv()：矩阵求逆
        s1 = 1.0 / math.sqrt(np.linalg.det(sigma)) # |Σ|
        s2 = np.linalg.inv(sigma)  # d*d Σ^(-1)
        delta = np.array([y - mu])  # 1*d
        return s1 * math.exp(-1.0 / 2 * delta @ s2 @ delta.T) # 多维高斯分布函数

    def fit(self, data):
        # 迭代训练
        self.inin_param(data)
        step = 0
        gamma_all_arr = None
        while step < self.maxstep:
            step += 1
            old_alpha = copy.copy(self.alpha)
            # E步
            gamma_all = []
            for j in range(self.N):
                gamma_j = []    # 依次求每个样本对K个分模型的响应度

                for k in range(self.K):
                    gamma_j.append(self.alpha[k] * self._phi(data[j], self.mu[k], self.sigma[k]))

                s = sum(gamma_j)
                gamma_j = [item/s for item in gamma_j]
                gamma_all.append(gamma_j)

            gamma_all_arr = np.array(gamma_all)
            # M步
            for k in range(self.K):
                gamma_k = gamma_all_arr[:, k]
                SUM = np.sum(gamma_k)
                # 更新权重
                self.alpha[k] = SUM / self.N  # 更新权重
                # 更新均值向量
                new_mu = sum([gamma * y for gamma, y in zip(gamma_k, data)]) / SUM  # 1*d
                self.mu[k] = new_mu
                # 更新协方差阵
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

    # 纯概率大小进行判断
    def predict(self, y):
        result = []
        for k in range(self.K):
            result.append(self.alpha[k] * self._phi(y, self.mu[k], self.sigma[k]))
        return np.argmax(result), 1 if result[0] < 1e-128 else 0, result

    # 概率大小 + 判决门限，result[0] < 1e-80, 则判断为语音信号；result[噪声概率，语音概率]
    def predict_thr(self, y):
        result = []
        for k in range(self.K):
            result.append(self.alpha[k] * self._phi(y, self.mu[k], self.sigma[k]))
        return 1 if result[0] < 1e-128 else 0

    # 均值自适应
    def predict_sef_adap(self, y):
        result = []
        for k in range(self.K):
            result.append(self.alpha[k] * self._phi(y, self.mu[k], self.sigma[k]))
        if np.argmax(result) == 0:  # 如果是0类 / 噪声信号
            self.mu = self.mu + (y - self.mu) / self.N * self.alpha[0] * self._phi(y, self.mu[0], self.sigma[0])
        else:  # 如果是1类 / 语音信号
            self.mu = self.mu + (y - self.mu) / self.N * self.alpha[1] * self._phi(y, self.mu[1], self.sigma[1])
        return np.argmax(result)


# 建立文件名矩阵
file_name = list()
a = 9
for i in range(3):
    file_name.append('sub1_sti_mot1_d' + str(i + a))
    file_name.append('sub1_ctr_mot1_d' + str(i + a))
    file_name.append('sub1_ctr_mot2_d' + str(i + a))
    file_name.append('sub1_ctr_mot3_d' + str(i + a))
    file_name.append('sub1_ctr_mot4_d' + str(i + a))


# 导入文件
feature_set_all = []
#
# for i in range(20):
#     path_name = r'C:\Users\盼哥哥\Desktop\个人经验管理体系\1-项目课题\3.5-基于肌电唤醒技术的多场景应用-腕关节断离患者的智能助手\数据采集'
#     all_data = pd.read_csv(path_name + '\\' + file_name[i] + '.csv', header=None, skiprows=1)
#     feature_set_sig = []
#     for k in range(20):
#         # 特征提取
#         feature_set_sig.append(EAD_feat_ext(all_data.values[k*8]))
#
#     feature_set_all.append(feature_set_sig)
#
# # # 特征矩阵拼接
# feature_set_all_arr = np.array(feature_set_all[0])
# for i in range(19):
#     feature_set_all_arr = np.vstack((feature_set_all_arr, feature_set_all[i+1]))
#
# feature_set_all_arr = feature_set_all_arr.reshape((400*92, 3))
#
# # # # 训练高斯混合模型
# EAD_gem = GEM(K=2)
# EAD_gem.fit(feature_set_all_arr)
# # #
# # # 保存类文件
# gmm_save = open('saves/' + 'EAD_validate' + '.p', 'wb')
# pickle.dump(EAD_gem, gmm_save)
# gmm_save.close()


# # 加载类文件
gmm_open = open('saves/' + 'EAD_train' + '.p', 'rb')
EAD_load_train = pickle.load(gmm_open)

gmm_open = open('saves/' + 'EAD_validate' + '.p', 'rb')
EAD_load_validate = pickle.load(gmm_open)


# 探索每个文件的识别率
for i in range(1):
    # i = i+7
    path_name = r'C:\Users\盼哥哥\Desktop\个人经验管理体系\1-项目课题\3.5-基于肌电唤醒技术的多场景应用-腕关节断离患者的智能助手\数据采集\手势识别-1'
    all_data = pd.read_csv(path_name + '\\' + file_name[i] + '.csv', header=None, skiprows=1)

    for k in range(1):
        # 特征提取
        # feature_set = EAD_feat_ext(prepro_data[0, :])
        feature_set = EAD_feat_ext(all_data.values[k*8])

        # gem = GEM(K=2)
        # gem.fit(feature_set)
        # print(EAD_gem_load.alpha, '\n', EAD_gem_load.sigma, '\n', EAD_gem_load.mu)

        # 活动段检测
        res = []

        for j in range(len(feature_set)):
            res.append(EAD_load_validate.predict(feature_set[j, :]))
            # res_ada.append(EAD_gem_load.predict_thr(feature_set[j, :]))
        res = np.array(res)
        res_ori = res[:, 0]
        res_thr = res[:, 1]
        # res_ada = np.array(res_ada)

        # 5帧后处理 - 众数
        res_thr_mode = []
        for j in range(len(feature_set) - 4):
            res_thr_mode.append(stats.mode(res_thr[j:j + 5])[0])
        res_thr_mode = np.array(res_thr_mode)

        # 5帧后处理 - 4/5
        res_thr_mode_plus = []
        for j in range(len(feature_set) - 4):
            if stats.mode(res_thr[j:j + 5])[0] == 0:
                res_thr_mode_plus.append(0)
            elif stats.mode(res_thr[j:j + 5])[0] == 1:
                if stats.mode(res_thr[j:j + 5])[1] > 3:
                    res_thr_mode_plus.append(1)
                else:
                    res_thr_mode_plus.append(0)


        res_thr_mode_plus = np.array(res_thr_mode_plus)

        plt.figure()
        plt.subplot(311)
        plt.plot(res_thr)
        # plt.plot(res_ori)

        plt.subplot(312)
        plt.plot(res_thr_mode)

        plt.subplot(313)
        plt.plot(res_thr_mode_plus)
        plt.show()
