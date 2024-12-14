class PseudoWords:
    def __init__(self, train_set, test_set, word_pos_counter, total_words, words_in_train_set):
        self.train_set = train_set
        self.test_set = test_set
        self.word_pos_counter = word_pos_counter
        self.total_words = total_words
        self.words_in_train_set = words_in_train_set
        self.words_to_replace = self.total_words - self.words_in_train_set
        self.add_low_frequent_words()
        self.words_to_pseudo = {}
        self.categorize_pseudo_words()
        self.replace_pseudo_words(self.train_set)
        self.replace_pseudo_words(self.test_set)

    def replace_pseudo_words(self, data_set):
        for i, sent in enumerate(data_set):
            for j in range(len(sent)):
                word, pos = sent[j]
                if word in self.words_to_pseudo:
                    data_set[i][j] = (self.words_to_pseudo[word], pos)


    def add_low_frequent_words(self):
        for word in self.word_pos_counter.keys():
            if sum(self.word_pos_counter[word].values()) < 5:
                self.words_to_replace.add(word)

    def categorize_pseudo_words(self):
        for word in self.words_to_replace:
            if word.endswith('ing'):
                self.words_to_pseudo[word] = 'ingSuff'
            elif word.isnumeric() and len(word) == 4:
                self.words_to_pseudo[word] = "4DigitNum"
            elif word.isnumeric():
                self.words_to_pseudo[word] = "digitNum"
            elif word.endswith('ed'):
                self.words_to_pseudo[word] = 'edSuff'
            elif word.startswith('$'):
                self.words_to_pseudo[word] = '$Pref'
            elif word.endswith('th'):
                self.words_to_pseudo[word] = 'thSuff'
            elif word.endswith("'s"):
                self.words_to_pseudo[word] = "'sSuff"
            elif word.endswith('ly'):
                self.words_to_pseudo[word] = "lySuff"
            elif word.endswith('ion'):
                self.words_to_pseudo[word] = "ionSuff"
            elif word.isupper():
                self.words_to_pseudo[word] = "upperCase"
            elif len(word) > 1 and word[0].isupper():
                self.words_to_pseudo[word] = "initUpperCase"
            elif word.endswith('er'):
                self.words_to_pseudo[word] = 'erSuff'
            elif '-' in word:
                self.words_to_pseudo[word] = '-InWord'
            elif '.' in word:
                self.words_to_pseudo[word] = '.InWord'
            elif ':' in word:
                self.words_to_pseudo[word] = ':InWord'
            elif ',' in word:
                self.words_to_pseudo[word] = ',InWord'
            else:
                self.words_to_pseudo[word] = 'otherWord'