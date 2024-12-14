# import nltk
# nltk.download('brown')
import copy
import re
from collections import defaultdict, Counter

import numpy as np
import sklearn.metrics
from nltk.corpus import brown
from sklearn.model_selection import train_test_split

from constants import Constants
from mleTagBaseline import MleTagBaseline
from bigramHMM import BigramHMM
from pseudoWords import PseudoWords


def clean_corpus(corpus):
    for i, sentence in enumerate(corpus):
        for j, (word, tag) in enumerate(sentence):
            corpus[i][j] = (word, clean_tag(tag))
    return corpus


def clean_tag(tag):
    # Use regex to match the prefix before the first '+', '-', '*', or '$'
    match = re.match(r'[^+\-*\$]*', tag)
    prefix = match.group()
    if prefix:
        return prefix
    return tag


def load_data():
    # Load the tagged sentences for the "news" category
    news = brown.tagged_sents(categories='news')
    news = clean_corpus(news)
    # Divide the corpus into training set and test set (last 10% as the test set)
    train_set, test_set = train_test_split(news, test_size=0.1, train_size=0.9, shuffle=False)
    train_set, test_set = clean_corpus(train_set), clean_corpus(test_set)
    return train_set, test_set


def count_occurrences(train_set, test_set):
    pos_counter = Counter()
    word_pos_counter = defaultdict(Counter)
    seen_words = set()
    total_words = set()

    for sent in train_set:
        pos_counter[Constants.START_POS] += 1
        for word, pos in sent:
            pos_counter[pos] += 1
            word_pos_counter[word][pos] += 1
            seen_words.add(word)
            total_words.add(word)
    for sent in test_set:
        for word, pos in sent:
            total_words.add(word)

    return pos_counter, word_pos_counter, seen_words, total_words


def accuracy(test_set, pos_pred):
    seen_words, unseen_words, correct_pred_seen, correct_pred_unseen = 0, 0, 0, 0
    for i, sent in enumerate(test_set):
        for j, pair in enumerate(sent):
            word, pos = pair
            # seen word
            if word in words_in_train_set:
                seen_words += 1
                if pos_pred[i][j] == pos:
                    correct_pred_seen += 1
            # unseen word
            else:
                unseen_words += 1
                if pos_pred[i][j] == pos:
                    correct_pred_unseen += 1

    seen_accuracy = correct_pred_seen / seen_words
    unseen_accuracy = correct_pred_unseen / unseen_words
    total_accuracy = (correct_pred_seen + correct_pred_unseen) / (seen_words + unseen_words)

    return seen_accuracy, unseen_accuracy, total_accuracy


if __name__ == '__main__':
    # Q(a)
    train_set, test_set = load_data()
    pos_counter, word_pos_counter, words_in_train_set, total_words = count_occurrences(train_set, test_set)
    # Q(b)
    mle_classifier = MleTagBaseline()
    mle_classifier.train(pos_counter, word_pos_counter)
    seen_accuracy, unseen_accuracy, total_accuracy = accuracy(test_set, mle_classifier.predict(test_set))
    print("Qb - ii")
    print("MLE tag classifier:")
    print(f"The error rate of seen words is: {1 - seen_accuracy}")
    print(f"The error rate of unseen words is: {1 - unseen_accuracy}")
    print(f"The error rate of all words is: {1 - total_accuracy}\n")

    # # Q (c)
    bigram_hmm_classifier = BigramHMM(train_set, test_set, pos_counter, word_pos_counter, words_in_train_set,
                                      total_words)
    bigram_hmm_classifier.train()
    bigram_hmm_pos_pred = bigram_hmm_classifier.predict()
    seen_accuracy, unseen_accuracy, total_accuracy = accuracy(test_set, bigram_hmm_pos_pred)
    print("Qc - iii")
    print("Bigram HMM:")
    print(f"The error rate of seen words is: {1 - seen_accuracy}")
    print(f"The error rate of unseen words is: {1 - unseen_accuracy}")
    print(f"The error rate of all words is: {1 - total_accuracy}\n")

    # Q(d)
    bigram_hmm_with_smoothing = BigramHMM(train_set, test_set, pos_counter, word_pos_counter, words_in_train_set,
                                          total_words,
                                          True)
    bigram_hmm_with_smoothing.train()
    seen_accuracy, unseen_accuracy, total_accuracy = accuracy(test_set, bigram_hmm_with_smoothing.predict())
    print("Qd - iii")
    print("Bigram HMM with Add-1 smoothing:")
    print(f"The error rate of seen words is: {1 - seen_accuracy}")
    print(f"The error rate of unseen words is: {1 - unseen_accuracy}")
    print(f"The error rate of all words is: {1 - total_accuracy}\n")

    # Q(e)
    pseudoWords = PseudoWords(copy.deepcopy(train_set), copy.deepcopy(test_set), word_pos_counter, total_words,
                              words_in_train_set)
    pos_counter_pseudo, word_pos_counter_pseudo, words_in_train_set_pseudo, total_words_pseudo = \
        count_occurrences(pseudoWords.train_set, pseudoWords.test_set)

    bigram_hmm_with_pseudo = BigramHMM(pseudoWords.train_set, pseudoWords.test_set, pos_counter_pseudo,
                                       word_pos_counter_pseudo,
                                       words_in_train_set_pseudo, total_words_pseudo)

    bigram_hmm_with_pseudo.train()
    seen_accuracy, unseen_accuracy, total_accuracy = accuracy(test_set, bigram_hmm_with_pseudo.predict())
    print("Qe - ii")
    print("Bigram HMM with Pseudo-words:")
    print(f"The error rate of seen words is: {1 - seen_accuracy}")
    print(f"The error rate of unseen words is: {1 - unseen_accuracy}")
    print(f"The error rate of all words is: {1 - total_accuracy}\n")

    bigram_hmm_with_pseudo_and_smoothing = BigramHMM(pseudoWords.train_set, pseudoWords.test_set, pos_counter_pseudo,
                                                     word_pos_counter_pseudo,
                                                     words_in_train_set_pseudo, total_words_pseudo,
                                                     True)

    bigram_hmm_with_pseudo_and_smoothing.train()
    pred_pseudo_smoothing = bigram_hmm_with_pseudo_and_smoothing.predict()
    seen_accuracy, unseen_accuracy, total_accuracy = accuracy(test_set, pred_pseudo_smoothing)
    print("Qe - iii")
    print("Bigram HMM with Add-1 smoothing and Pseudo-words:")
    print(f"The error rate of seen words is: {1 - seen_accuracy}")
    print(f"The error rate of unseen words is: {1 - unseen_accuracy}")
    print(f"The error rate of all words is: {1 - total_accuracy}\n")

    y_true = [pos for sent in test_set for _, pos in sent]
    y_pred = [pos for sent in pred_pseudo_smoothing for pos in sent]
    tags = np.unique(y_true)
    confusion_mat = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=tags)
    print("Confusion matrix investigation:")
    np.fill_diagonal(confusion_mat, -1)
    max_err_tags = confusion_mat.argmax(1)
    for i, tag in enumerate(tags):
        print(f" for the true POS {tag}, the most frequent POS mistakenly predicted is {tags[max_err_tags[i]]}")

