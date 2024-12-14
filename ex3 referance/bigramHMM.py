import numpy as np
from collections import defaultdict

import constants
from constants import Constants


class BigramHMM:

    def __init__(self, train_set, test_set, pos_counter, word_pos_counter, words_in_train_set, total_words,
                 add_one_smoothing=False):
        self.train_set = train_set
        self.test_set = test_set
        self.pos_counter = pos_counter
        self.word_pos_counter = word_pos_counter
        self.words_in_train_set = words_in_train_set
        self.total_words = total_words
        self.add_one_smoothing = add_one_smoothing
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))

    def calculate_bigram_pos(self):
        bigram_pos_counter = defaultdict(lambda: defaultdict(int))

        for sent in self.train_set:
            prev = Constants.START_POS
            for word, pos in sent:
                bigram_pos_counter[prev][pos] += 1
                prev = pos
            bigram_pos_counter[prev][Constants.STOP_POS] += 1

        return bigram_pos_counter

    def train(self):
        bigram_pos_counter = self.calculate_bigram_pos()

        # Calculate transitions
        for prev_pos in bigram_pos_counter.keys():
            for curr_pos, count in bigram_pos_counter[prev_pos].items():
                self.transitions[prev_pos][curr_pos] = count / self.pos_counter[prev_pos]

        # Calculate emissions
        if self.add_one_smoothing:
            for pos in self.pos_counter.keys():
                for word in self.total_words:
                    self.emissions[pos][word] = self.word_pos_counter[word][pos] + 1

        else:
            for word in self.word_pos_counter.keys():
                for pos, count in self.word_pos_counter[word].items():
                    self.emissions[pos][word] = count

        self.emissions = self.normalize(self.emissions)

    def normalize(self, d):
        for tag in d.keys():
            sum = 0
            for word, count in d[tag].items():
                sum += count
            for word in d[tag].keys():
                d[tag][word] /= sum

        return d

    def viterbi_forward_pass(self, sentence, tag_to_index, tags, dp, backpointers):
        for i in range(len(sentence)):
            word = sentence[i]
            for tag in tags:
                if word not in self.words_in_train_set and not self.add_one_smoothing:
                    self.emissions[Constants.DEFAULT_POS][word] = 1
                if i == 0:
                    dp[i][tag_to_index[tag]] = self.transitions[Constants.START_POS][tag] * self.emissions[tag][word]
                    continue
                best_prev_tag_index = 0
                best_prob = 0
                for j, prev_tag in enumerate(tags):
                    cur_prob = dp[i - 1][j] * self.transitions[prev_tag][tag] * self.emissions[tag][word]
                    if cur_prob > best_prob:
                        best_prev_tag_index = j
                        best_prob = cur_prob
                dp[i][tag_to_index[tag]] = best_prob
                backpointers[i][tag_to_index[tag]] = best_prev_tag_index

        dp[-1] = [dp[-1][i] * self.transitions[tag][Constants.STOP_POS] for i, tag in enumerate(tags)]
        return dp, backpointers

    def viterbi_backward_pass(self, sentence, tag_to_index, tags, dp, backpointers):
        best_last_tag_index = np.argmax(dp[len(sentence) - 1])
        best_tagging = [tags[best_last_tag_index]]

        for i in range(len(sentence) - 1, 0, -1):
            best_last_tag_index = tag_to_index[best_tagging[-1]]
            prev_tag = backpointers[i][best_last_tag_index]
            best_tagging.append(tags[prev_tag])

        return best_tagging[::-1]

    def viterbi_bigram(self, sentence):
        tags = list(self.emissions.keys())
        num_tags = len(tags)
        dp = [[0.0 for _ in range(num_tags)] for _ in range(len(sentence))]
        backpointers = [[0 for _ in range(num_tags)] for _ in range(len(sentence))]
        tag_to_index = {tag: i for i, tag in enumerate(tags)}

        dp, backpointers = self.viterbi_forward_pass(sentence, tag_to_index, tags, dp, backpointers)

        return self.viterbi_backward_pass(sentence, tag_to_index, tags, dp, backpointers)

    def predict(self):
        pos_pred = []
        for sent in self.test_set:
            sent_only_words = [word for word, tag in sent]
            pos_pred.append(self.viterbi_bigram(sent_only_words))

        return pos_pred
