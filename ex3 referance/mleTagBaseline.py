from collections import defaultdict
from constants import Constants


class MleTagBaseline:
    def __init__(self):
        self.best_pos_per_word = defaultdict(str)

    def train(self, pos_counter, word_pos_counter):

        corpus_size = sum(pos_counter.values())

        # Iterate through each unique word in the training set
        for word in word_pos_counter.keys():
            cur_best_prob = 0

            # Iterate through each POS tag associated with the current word
            for pos in word_pos_counter[word]:
                # Calculate the probability of the current POS tag in the entire corpus
                pos_prob = pos_counter[pos] / corpus_size

                # Calculate the conditional probability of the word given the current POS tag
                word_given_pos_prob = word_pos_counter[word][pos] / pos_counter[pos]

                # Calculate the overall probability for the current word and POS tag
                cur_prob = word_given_pos_prob * pos_prob

                # Update the best POS tag for the current word if the current probability is higher
                if cur_best_prob < cur_prob:
                    cur_best_prob = cur_prob
                    self.best_pos_per_word[word] = pos

    def predict(self, test_set):
        pos_pred = []
        for sent in test_set:
            sent_pos_pred = []
            for word, pos in sent:
                if self.best_pos_per_word[word]:
                    sent_pos_pred.append(self.best_pos_per_word[word])
                else:
                    sent_pos_pred.append(Constants.DEFAULT_POS)
            pos_pred.append(sent_pos_pred)

        return pos_pred
