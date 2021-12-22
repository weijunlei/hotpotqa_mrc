import numpy as np
import pandas as pd
import string
import nltk
from sklearn.utils import shuffle
from os.path import expanduser
from nltk.corpus import wordnet


def get_similarity(text_a, text_b):
    wordnet = nltk.corpus.wordnet
    left_lsent = text_a.lower().translate(str.maketrans('', '', string.punctuation)).split()
    right_lsent = text_b.lower().translate(str.maketrans('', '', string.punctuation)).split()
    sim = []
    for i in range(len(left_lsent)):
        word = left_lsent[i]
        tmp = []
        for j in range(len(right_lsent)):
            targ = right_lsent[j]
            left_syn = get_synsets(word)
            right_syn = get_synsets(targ)
            left = wordnet.synsets(word)
            right = wordnet.synsets(targ)
            if word != 'oov' and targ != 'oov':
                if left != [] and right != []:
                    if targ in left_syn or word in right_syn:
                        tmp.append(1.0)
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
                            tmp.append(similarity)
                        else:
                            if word == targ:
                                tmp.append(1)
                            else:
                                tmp.append(0)
                else:
                    if word == targ:
                        tmp.append(1)
                    else:
                        tmp.append(0)
            else:
                tmp.append(0)

        sim.append(tmp)

    return sim


def get_synsets(input_lemma):
    wordnet = nltk.corpus.wordnet
    synsets = []
    for syn in wordnet.synsets(input_lemma):
        for lemma in syn.lemmas():
            synsets.append(lemma.name())

    return synsets


if __name__ == '__main__':
    tmp = get_similarity("hello carry", "hello carry")
    print(tmp)
