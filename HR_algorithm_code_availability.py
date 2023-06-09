# handwriting recognition for subject A2

from keras.models import *
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# import the corpus
corpus_one = np.load('npy/corpus_one.npy')
corpus_two = np.load('npy/corpus_two.npy')
corpus_three = np.load('npy/corpus_three.npy')
corpus_four = np.load('npy/corpus_four.npy')
corpus_five = np.load('npy/corpus_five.npy')
corpus_plus_ori = np.load('npy/corpus_plus_ori.npy')
corpus_plus_cut = np.load('npy/corpus_plus_cut.npy')
corpus_plus_num = np.load('npy/corpus_plus_num.npy')


# one-letter word
def prob_to_list_one(single_tree):
    word_list = list()
    for i in single_tree[0]:
            word_inner = lab2cha_dic[i]
            word_list.append(word_inner)
    for z in word_list:
        if z in corpus_one:
            return z
    return word_list[0]


# two-letter word
def prob_to_list_two(single_tree):
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            word_inner = lab2cha_dic[i] + lab2cha_dic[j]
            word_list.append(word_inner)
    for z in word_list:
        if z in corpus_two:
            return z
    return word_list[0]


# three-letter word
def prob_to_list_three(single_tree):
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k]
                word_list.append(word_inner)
    for z in word_list:
        if z in corpus_three:
            return z
    return word_list[0]


# four-letter word
def prob_to_list_four(single_tree):
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                for m in single_tree[3]:
                    word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k] + lab2cha_dic[m]
                    word_list.append(word_inner)
    for z in word_list:
        if z in corpus_four:
            return z
    return word_list[0]


# five-letter word
def prob_to_list_five(single_tree):
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                for m in single_tree[3]:
                    for n in single_tree[4]:
                        word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k] + lab2cha_dic[m] + lab2cha_dic[n]
                        word_list.append(word_inner)
    for z in word_list:
        if z in corpus_five:
            return z
    return word_list[0]


# Words with more than five letters
def prob_to_list_five_plus(single_tree):
    word_list = list()
    for i in single_tree[0]:
        for j in single_tree[1]:
            for k in single_tree[2]:
                for m in single_tree[3]:
                    for n in single_tree[4]:
                        for q in single_tree[5]:
                            word_inner = lab2cha_dic[i] + lab2cha_dic[j] + lab2cha_dic[k] + lab2cha_dic[m] + lab2cha_dic[n] + lab2cha_dic[q]
                            word_list.append(word_inner)
    for z in word_list:
        if z in corpus_plus_cut:
            index = np.where(corpus_plus_cut == z)[0]
            length = corpus_plus_num[index]
            fin_index = np.where(length == len(single_tree))[0]
            if len(fin_index) > 0:
                return corpus_plus_ori[index[fin_index]][0]
    pre_word = list()
    for i in range(len(single_tree)):
        pre_word.append(lab2cha_dic[single_tree[i][0]])
    return "".join(pre_word)


# language model
def em_to_lm(single_tree_prob):
    # judge word length
    num = len(single_tree_prob)
    if num == 1:
        search_word = prob_to_list_one(single_tree_prob)
    elif num == 2:
        search_word = prob_to_list_two(single_tree_prob)
    elif num == 3:
        search_word = prob_to_list_three(single_tree_prob)
    elif num == 4:
        search_word = prob_to_list_four(single_tree_prob)
    elif num == 5:
        search_word = prob_to_list_five(single_tree_prob)
    else:
        search_word = prob_to_list_five_plus(single_tree_prob)
    return search_word


# EMG model
def em_to_wrd(single_tree_prob):
    word_list = list()
    num = len(single_tree_prob)
    for w in range(num):
        word_list.append(lab2cha_dic[single_tree_prob[w][0]])
    return "".join(word_list)


# label
cha2lab_dic = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9,
               'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17,
               'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26,
               '&': 27, '~': 28, '!': 29, '%': 30}

lab2cha_dic = dict([val, key] for key, val in cha2lab_dic.items())

# import the testing handwriting content
with open('txt/test-text.txt', encoding='utf-8') as file:
    c_text = file.read()
cont_text = c_text.split('\n')

# import the EMG model
model_load = load_model('saves/hdw_sub2_EM' + '.h5')

# import the testing set
x_test_aug = np.load('npy/hdw_sub2_feat_testing_aug.npy')
y_test = np.load('npy/hdw_sub2_label_testing_aug.npy')

# normalization
for i in range(x_test_aug.shape[0]):
    x_test_aug[i, :, :, :] = (x_test_aug[i, :, :, :] - np.mean(x_test_aug[i, :, :, :]))/np.std(x_test_aug[i, :, :, :])

y_test_one_hot = to_categorical(y_test-1)
pre_one_hot_aug = model_load.predict(x_test_aug)

pre_non_onehot_aug = [np.argmax(x)+1 for x in pre_one_hot_aug]
pre_non_onehot_aug = np.array(pre_non_onehot_aug).reshape((len(pre_non_onehot_aug), 1))


# perfomr data augmentation on testing set
pre_fin_mode = list()
pre_mode_res = list()
for i in range(302):
    temp2 = np.zeros((1, 30))
    for k in range(12):
         temp2 = temp2 + pre_one_hot_aug[i*12 + k, :]
    pre_mode_res.append(temp2)

pre_mode_res = np.array(pre_mode_res).reshape((302, 30))
pre_fin_mode = np.array([np.argmax(x)+1 for x in pre_mode_res])
pre_fin_mode = pre_fin_mode.reshape((302, 1))

# build the String Search Tree
pre_search_tree = [np.argsort(-x)[0:5] + 1 for x in pre_mode_res]
pre_search_tree = np.array(pre_search_tree)

pre_fin = np.array(pre_fin_mode)
acc_sin_mode = (pre_fin_mode == y_test).mean()
print(acc_sin_mode)

# testing set with no data augmentation
# pre_search_tree = [np.argsort(-x)[0:5] + 1 for x in pre_one_hot_aug]
# pre_search_tree = np.array(pre_search_tree)
#
# pre_fin = [np.argmax(x)+1 for x in pre_one_hot_aug]
# pre_fin = np.array(pre_fin).reshape((len(pre_fin), 1))

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

# testing
for z in range(len(cont_text)):
    item = cont_text[z]
    item_text = item.split(' ')
    print(item)
    a = 0
    for k, cha in enumerate(item):
        if cha == ' ':
            em_word = em_to_wrd(single_tree_prob)
            print(em_word)
            lm_word = em_to_lm(single_tree_prob)
            print(lm_word)
            correct_word.append(item_text[a])
            pre_word.append(lm_word)
            em_wrd_full.append("".join(em_wrd_sep))
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

# character accuracy
# EMG model results
em_acc_cha = (np.array(correct_cha) == np.array(em_all_cha)).mean()
em_num_cha = (np.array(correct_cha) == np.array(em_all_cha)).sum()
print(em_num_cha)
print(len(correct_cha))

# language model results
lm_acc_cha = (np.array(correct_cha) == np.array(pre_cha)).mean()
lm_num_cha = (np.array(correct_cha) == np.array(pre_cha)).sum()
print(lm_num_cha)
print(len(correct_cha))

# confusion matrix
cm_label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', '&', '~', '!', '%']

cm_label_plot = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
            'w', 'x', 'y', 'z', ',', '.', '?', '%']
C = confusion_matrix(correct_cha, pre_cha, labels=cm_label)

plt.matshow(C, cmap=plt.cm.Blues)

for i in range(len(C)):
    for j in range(len(C)):
        plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center', weight='normal')

plt.xticks(range(0, 30), labels=cm_label_plot)
plt.yticks(range(0, 30), labels=cm_label_plot)
plt.show()

