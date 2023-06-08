'''
字符识别优化流程
    1. 建立字符串搜索树
        1. 每个字符取前五个概率输出构建搜索树 v1
        2. 混淆矩阵 v2
        3. 概率 + 2元语言模型 v3

    2. 计算字符串概率
        1. 字符语言模型
        2. 字符语言模型 + 词语语言模型
'''



from keras.models import *
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
import csv

n_gram_dic = np.load('npy/n_gram_dic.npy')
n_gram_freq = np.load('npy/n_gram_freq.npy')
corpus_one = np.load('npy/corpus_one.npy')
corpus_two = np.load('npy/corpus_two.npy')
corpus_three = np.load('npy/corpus_three.npy')
corpus_four = np.load('npy/corpus_four.npy')
corpus_five = np.load('npy/corpus_five.npy')
corpus_plus_ori = np.load('npy/corpus_plus_ori.npy')
corpus_plus_cut = np.load('npy/corpus_plus_cut.npy')
corpus_plus_num = np.load('npy/corpus_plus_num.npy')


# 模型测试
# 概率转字符串列表


# 一个字母
def prob_to_list_one(single_tree):
    # 构建字符串列表
    word_list = list()
    for i in single_tree[0]:
            word_inner = lab2cha_dic[i]
            word_list.append(word_inner)
    # print(word_list)
    # 寻找概率最大的字符串作为输出
    # 判断列表在不在单词库
    # 如果在，则输出第一个存在的单词
    # 若不在，则还是输出第一个，因为这两个都是概率最大的两个
    for z in word_list:
        if z in corpus_one:
            return z
    return word_list[0]


# 两个字母
def prob_to_list_two(single_tree):
    # 构建字符串列表
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            word_inner = lab2cha_dic[i] + lab2cha_dic[j]
            word_list.append(word_inner)
    # print(word_list)
    # 寻找概率最大的字符串作为输出
    # 判断列表在不在单词库
    # 如果在，则输出第一个存在的单词
    # 若不在，则还是输出第一个，因为这两个都是概率最大的两个
    for z in word_list:
        if z in corpus_two:
            return z
    return word_list[0]


# 三个字母
def prob_to_list_three(single_tree):
    # 构建字符串列表
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k]
                word_list.append(word_inner)
    # print(word_list)
    # 寻找概率最大的字符串作为输出
    # 判断列表在不在单词库
    # 如果在，则输出第一个存在的单词
    # 若不在，则还是输出第一个，因为这两个都是概率最大的两个
    for z in word_list:
        if z in corpus_three:
            return z
    return word_list[0]


# 四个字母
def prob_to_list_four(single_tree):
    # 构建字符串列表
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                for m in single_tree[3]:
                    word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k] + lab2cha_dic[m]
                    word_list.append(word_inner)
    # print(word_list)
    # 寻找概率最大的字符串作为输出
    # 判断列表在不在单词库
    # 如果在，则输出第一个存在的单词
    # 若不在，则还是输出第一个，因为这两个都是概率最大的两个
    for z in word_list:
        if z in corpus_four:
            return z
    return word_list[0]


# 五个字母
def prob_to_list_five(single_tree):
    # 构建字符串列表
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                for m in single_tree[3]:
                    for n in single_tree[4]:
                        word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k] + lab2cha_dic[m] + lab2cha_dic[n]
                        word_list.append(word_inner)
    # print(word_list)
    # 寻找概率最大的字符串作为输出
    # 判断列表在不在单词库
    # 如果在，则输出第一个存在的单词
    # 若不在，则还是输出第一个，因为这两个都是概率最大的两个
    for z in word_list:
        if z in corpus_five:
            return z
    return word_list[0]


# 超过5个字母
def prob_to_list_five_plus(single_tree):
    # 构建字符串列表
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                for m in single_tree[3]:
                    for n in single_tree[4]:
                        for q in single_tree[5]:
                            word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k] + lab2cha_dic[m] + lab2cha_dic[n] + lab2cha_dic[q]
                            word_list.append(word_inner)
    # print(word_list)
    # 寻找概率最大的字符串作为输出
    # 判断列表在不在单词库
    # 如果在，则输出第一个存在的单词
    # 若不在，则还是输出第一个，因为这两个都是概率最大的两个
    for z in word_list:
        if z in corpus_plus_cut:
            index = np.where(corpus_plus_cut == z)[0]
            length = corpus_plus_num[index]
            fin_index = np.where(length == len(single_tree))[0]
            if len(fin_index) > 0:
                # print('***')
                return corpus_plus_ori[index[fin_index]][0]

    pre_word = list()
    for i in range(len(single_tree)):
        pre_word.append(lab2cha_dic[single_tree[i][0]])
    # print('$$$')
    return "".join(pre_word)


# 肌电模型+语言模型
def em_to_lm(single_tree_prob):
    # 判断字符串长度
    num = len(single_tree_prob)
    if num == 1:
        search_word = prob_to_list_one(single_tree_prob)
        # print(search_word)
    elif num == 2:
        search_word = prob_to_list_two(single_tree_prob)
        # print(search_word)
    elif num == 3:
        search_word = prob_to_list_three(single_tree_prob)
        # print(search_word)
    elif num == 4:
        search_word = prob_to_list_four(single_tree_prob)
        # print(search_word)
    elif num == 5:
        search_word = prob_to_list_five(single_tree_prob)
        # print(search_word)
    else:
        search_word = prob_to_list_five_plus(single_tree_prob)
        # print(search_word)
    return search_word
    # return search_word


# 肌电模型输出单词
def em_to_wrd(single_tree_prob):
    word_list = list()
    num = len(single_tree_prob)
    for w in range(num):
        word_list.append(lab2cha_dic[single_tree_prob[w][0]])
    return "".join(word_list)


# 字符转标签
cha2lab_dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
               'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
               'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
               '&': 27, '~': 28, '!': 29, '%': 30}

lab2cha_dic = dict([val, key] for key, val in cha2lab_dic.items())

# 文件名构建
with open('txt/test-text.txt', encoding='utf-8') as file:
    c_text = file.read()
cont_text = c_text.split('\n')


model_load = load_model('saves/hdw_sub3_em_sepcnn' + '.h5')

# x_test_1 = np.load('npy/hdw_tf_feaset_d9.npy')
# y_test_1 = np.load('npy/hdw_tf_lablel_d9.npy')
# x_test_2 = np.load('npy/hdw_tf_feaset_d10.npy')
# y_test_2 = np.load('npy/hdw_tf_lablel_d10.npy')
# x_test_3 = np.load('npy/hdw_tf_feaset_d11.npy')
# y_test_3 = np.load('npy/hdw_tf_lablel_d11.npy')
# # x_test = np.concatenate((x_test_1, x_test_2, x_test_3), axis=0)
# # y_test = np.concatenate((y_test_1, y_test_2, y_test_3), axis=0)
# x_test = x_test_3
# y_test = y_test_3

x_test = np.load('npy/hdw_sub3_feat_s5.npy')
y_test = np.load('npy/hdw_sub3_label_s5.npy')

for i in range(x_test.shape[0]):
    x_test[i, :, :, :] = (x_test[i, :, :, :] - np.mean(x_test[i, :, :, :]))/np.std(x_test[i, :, :, :])

y_test_one_hot = to_categorical(y_test-1)
pre_onehot = model_load.predict(x_test)

# 组建字符串搜索树
# 取概率最大的前五个
pre_search_tree = [np.argsort(-x)[0:5] + 1 for x in pre_onehot]
pre_search_tree = np.array(pre_search_tree)

pre_fin = [np.argmax(x)+1 for x in pre_onehot]
pre_fin = np.array(pre_fin).reshape((len(pre_fin), 1))

pre_text = list()
m = 0
em_all_cha = list()
em_wrd_sep = list()
em_wrd_full = list()
correct_cha = list()
pre_cha = list()
correct_word = list()
pre_word = list()
single_tree_prob = list()
# 预测标签转文本
for z in range(len(cont_text)):
    item = cont_text[z]
    item_text = item.split(' ')
    print(item)
    # list 转 矩阵
    a = 0
    for k, cha in enumerate(item):
        if cha == ' ':
            em_word = em_to_wrd(single_tree_prob)
            print(em_word)
            lm_word = em_to_lm(single_tree_prob)
            print(lm_word)

            # print(single_tree)
            correct_word.append(item_text[a])
            pre_word.append(lm_word)
            em_wrd_full.append("".join(em_wrd_sep))
            # print(em_wrd_sep)
            a = a + 1
            for z in lm_word:
                pre_cha.append(z)
            single_tree_prob = list()
            em_wrd_sep = list()
        else:
            correct_cha.append(cha)
            single_tree_prob.append(pre_search_tree[m, :])
            em_all_cha.append(lab2cha_dic[pre_fin[m, 0]])
            em_wrd_sep.append(lab2cha_dic[pre_fin[m, 0]])
            m = m + 1

# 计算字符ACC
# 肌电模型
em_acc_cha = (np.array(correct_cha) == np.array(em_all_cha)).mean()
em_num_cha = (np.array(correct_cha) == np.array(em_all_cha)).sum()
print(em_num_cha)
print(len(correct_cha))

# 肌电模型 + 语言模型
lm_acc_cha = (np.array(correct_cha) == np.array(pre_cha)).mean()
lm_num_cha = (np.array(correct_cha) == np.array(pre_cha)).sum()
print(lm_num_cha)
print(len(correct_cha))

# 计算单词ACC
# 肌电模型191
# 302
em_acc_wrd = (np.array(correct_word) == np.array(em_wrd_full)).mean()
em_num_wrd = (np.array(correct_word) == np.array(em_wrd_full)).sum()
print(em_num_wrd)
print(len(correct_word))

# 肌电模型 + 语言模型
lm_acc_wrd = (np.array(correct_word) == np.array(pre_word)).mean()
lm_num_wrd = (np.array(correct_word) == np.array(pre_word)).sum()
print(lm_num_wrd)
print(len(correct_word))





