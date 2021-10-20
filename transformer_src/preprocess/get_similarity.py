import numpy as np
import pandas as pd
import string
import nltk
import redis
from sklearn.utils import shuffle
from scipy import sparse
from os.path import expanduser
from nltk.corpus import wordnet

redis_data = redis.StrictRedis(host='localhost', port=6379, db=0)


def get_similarity(tokens, max_seq_length=512):
    wordnet = nltk.corpus.wordnet
    tokens_length = len(tokens)
    rows = []
    cols = []
    sim_matrix = sparse.dok_matrix((max_seq_length, max_seq_length), dtype=np.float32)
    not_zero_num = 0
    for i in range(tokens_length):
        word = tokens[i]
        for j in range(i, tokens_length):
            targ = tokens[j]
            tag_words = '$1_{}_$2_{}'.format(word, targ)
            get_tag_score = redis_data.get(tag_words)
            if get_tag_score != None:
                sim_matrix[i, j] = float(get_tag_score)
                get_tag_score = float(get_tag_score)
                if get_tag_score != 0:
                    not_zero_num += 1
                continue
            left_syn = get_synsets(word)
            left = wordnet.synsets(word)
            right_syn = get_synsets(targ)
            right = wordnet.synsets(targ)
            if word != 'oov' and targ != 'oov':
                if left != [] and right != []:
                    if targ in left_syn or word in right_syn:
                        get_tag_score = 1.0
                    else:
                        count1, count2= 0, 0
                        ScoreList1, ScoreList2 = 0, 0
                        for word1 in left:
                            for word2 in right:
                                try:
                                    score1 = word1.wup_similarity(word2)
                                except:
                                    score1 = 0.0
                                try:
                                    score2 = word2.wup_similarity(word1)
                                except:
                                    score2 = 0.0
                                #score1 = word1.stop_similarity(word2)
                                if score1 is not None:
                                    ScoreList1 += score1
                                    count1 += 1
                                if score2 is not None:
                                    ScoreList2 += score2
                                    count2 += 1

                        if count1 + count2 != 0:
                            similarity = (ScoreList1 + ScoreList2)/(count1 + count2)
                            get_tag_score = similarity
                            # tmp.append(similarity)
                        else:
                            if word == targ:
                                get_tag_score = 1.0
                                # tmp.append(1)
                            else:
                                get_tag_score = 0.0
                                # tmp.append(0)
                else:
                    if word == targ:
                        get_tag_score = 1.0
                        # tmp.append(1)
                    else:
                        get_tag_score = 0.0
                        # tmp.append(0)
            else:
                get_tag_score = 0.0
            redis_data.set(tag_words, get_tag_score)
            sim_matrix[i, j] = get_tag_score
            if get_tag_score != 0:
                not_zero_num += 1
    return sim_matrix


def get_synsets(input_lemma):
    wordnet = nltk.corpus.wordnet
    synsets = []
    for syn in wordnet.synsets(input_lemma):
        for lemma in syn.lemmas():
            synsets.append(lemma.name())

    return synsets


if __name__ == '__main__':
    tmp = get_similarity("hello carry [SEP]", 3)
    print(tmp)
