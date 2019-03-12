# -*- coding: utf-8 -*
import numpy as np
import random

alpha = 0.1
beta = 0.1
iter_times = 500
n_components = 20
n_top_words = 10


def get_key(dict, value):
    for k, v in dict.items():
        if v == value:
            return k


def lda():
    # 每篇文档随机分配主题
    for i in range(doc_num):
        dw_sum[i] = doc_length[i]
        for j in range(doc_length[i]):
            topic = random.randint(0, n_components-1)
            Z[i][j] = topic
            tw[doc_words[i][j]][topic] += 1
            dw[i][topic] += 1
            tw_sum[topic] += 1

    # 迭代过程
    for x in range(iter_times):
        for i in range(doc_num):
            for j in range(doc_length[i]):
                topic = gibbs_sampling(i, j)
                Z[i][j] = topic

    # 文档中主题的分布
    for i in range(doc_num):
        theta[i] = (dw[i] + alpha) / (dw_sum[i] + n_components * alpha)

    # 主题中词的分布
    for i in range(n_components):
        phi[i] = (tw.T[i] + beta) / (tw_sum[i] + words_num * beta)

    f = open('results_20.txt', 'w')

    for x in range(n_components):
        f.write("Topic: #%d" % x + '\n')
        idx = np.argsort(phi[x])
        
        for i in range(len(idx) - 1, len(idx) - 1 - n_top_words, -1):
            k = get_key(word2id, idx[i])
            f.write("\t" + k + ", " + str(phi[x][idx[i]]) + '\n')


# 吉布斯采样过程
def gibbs_sampling(i, j):
    topic = Z[i][j]
    word = doc_words[i][j]
    tw[word][topic] -= 1
    dw[i][topic] -= 1
    tw_sum[topic] -= 1
    dw_sum[i] -= 1

    # 每个词对应主题的Gibbs采样
    p1 = (tw[word] + beta) / (tw_sum + words_num * beta)
    p2 = (dw[i] + alpha) / (dw_sum[i] + n_components * alpha)
    p = p1 * p2

    p_sq = np.squeeze(np.asarray(p / np.sum(p)))
    topic = np.argmax(np.random.multinomial(1, p_sq))

    tw[word][topic] += 1
    tw_sum[topic] += 1
    dw[i][topic] += 1
    dw_sum[i] += 1
    return topic


if __name__ == '__main__':
    words_num = 0
    doc_num = 0
    doc_length = []
    doc_words = []
    word2id = {}

    # 加载停用词
    swfile = open('./ENstopwords.txt', 'r')
    sws = swfile.read()
    sws = sws.splitlines()
    stop_words = []
    for word in sws:
        stop_words.append(word)

    # 加载文档并去除停用词，初始化doc_num文档个数, word_num词汇个数, doc_length每个文档中词的个数
    news_file = open('./news.txt', 'r')
    lines = news_file.readlines()
    doc_num = len(lines)
    item_index = 0
    for line in lines:
        if line != "":
            words = []
            arr = line.strip().split()
            for w in arr:
                if w.isalpha():
                    w = w.lower()
                    if w not in stop_words:
                        if w in word2id.keys():
                            words.append(word2id[w])
                        else:
                            word2id[w] = item_index
                            words.append(item_index)
                            item_index += 1
            doc_length.append(len(words))
            doc_words.append(words)

    words_num = len(word2id)
    
    p = np.zeros(n_components)
    # tw: 每个主题中的词
    tw = np.zeros((words_num, n_components), dtype='int')
    # tw_sum: 每个主题中词的个数
    tw_sum = np.zeros(n_components, dtype='int')
    # dw: 每个文档中每个主题的次的个数
    dw = np.zeros((doc_num, n_components), dtype='int')
    # dw_sum： 每个文档中词的个数
    dw_sum = np.zeros(doc_num, dtype='int')

    # 每个文档中的每个词的主题，初始化
    Z = []
    for a in range(doc_num):
        temp_Z = []
        for b in range(doc_length[a]):
            temp_Z.append(0)
        Z.append(temp_Z)
    Z = np.array(Z)

    # 每个文档中的每个主题的theta
    theta = []
    for i in range(doc_num):
        temp = []
        for j in range(n_components):
            temp.append(0.0)
        theta.append(theta)

    # 每个主题的每个词的phi
    phi = []
    for i in range(n_components):
        temp = []
        for j in range(words_num):
            temp.append(0.0)
        phi.append(temp)

    lda()
