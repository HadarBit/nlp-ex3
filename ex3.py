from collections import defaultdict, Counter
import re

import nltk
import numpy as np
from nltk.corpus import brown
import pandas as pd

DEFAULT_TAG = "NN"
START_TAG = "<s>"
END_TAG = "</s>"


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
    # Ensure that the Brown corpus is downloaded
    nltk.download('brown')

    # Load the tagged sentences from the 'news' category
    news_sentences = list(brown.tagged_sents(categories='news'))

    # Calculate the split index for the last 10% of the sentences
    split_index = int(len(news_sentences) * 0.9)

    # Divide the sentences into training and test sets
    train_set = news_sentences[:split_index]
    test_set = news_sentences[split_index:]
    
    train_set = clean_corpus(train_set)
    test_set = clean_corpus(test_set)

    # Optionally, you can print the sizes of the train and test sets
    print(f"Training set size: {len(train_set)} sentences")
    print(f"Test set size: {len(test_set)} sentences")
    return train_set, test_set


from collections import defaultdict


def get_tranistion_probs(transition_counts):
    transition_probabilities = {
        prev_tag: {tag: count / sum(next_tags.values())
            for  tag, count in next_tags.items()
        }for prev_tag, next_tags in transition_counts.items()}
    return transition_probabilities 

def get_emission_probs(unique_tags,unique_words,emission_counts,smoothing_param,tag_counts):
  if smoothing_param > 0:
    emission_probabilities = {}
    vocab_size = len(unique_words)  # For smoothing calculations
    for tag in unique_tags:
        total = tag_counts[tag]
        tag_total_with_smoothing = total + smoothing_param * vocab_size

        emission_probabilities[tag] = {
            word: (emission_counts[tag][word] + smoothing_param) / tag_total_with_smoothing
            for word in unique_words
        }
  else:
    emission_probabilities = {}
    for tag, words in emission_counts.items():
        total = tag_counts[tag]
        emission_probabilities[tag] = {word: count / total  if count > 0 else 0 for word, count in words.items()}
  return emission_probabilities




def compute_transition_emission_probabilities(train_set, test_set, smoothing_param=0):
    # Transition and emission counts
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)

    # Collect unique words and tags from training set (no need for test set here)
    unique_words,unique_tags = set(),set()
    for sentence in train_set:
        for word, tag in sentence:
            unique_words.add(word)
            unique_tags.add(tag)

    # Process training set for transition and emission counts
    for sentence in train_set:
        previous_tag = START_TAG
        for word, tag in sentence:
            # Update counts
            transition_counts[previous_tag][tag] += 1
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1
            previous_tag = tag

        # Handle end-of-sentence transition
        transition_counts[previous_tag][END_TAG] += 1

    # Compute transition probabilities
    transition_probabilities =get_tranistion_probs(transition_counts)
    # Compute emission probabilities
    emission_probabilities =get_emission_probs(unique_tags,unique_words,emission_counts,smoothing_param,tag_counts)
    return transition_probabilities, emission_probabilities



def viterbi_algorithm(sentence, transition_probabilities, emission_probabilities, all_tags,
                      train_words, smoothing_param):
    unknown = 0
    # unknown = 1e-6
    dp = [[0.0 for _ in range(len(all_tags))] for _ in range(len(sentence))]
    backpointers = [[0 for _ in range(len(all_tags))] for _ in range(len(sentence))]
    tag_to_index = {tag: i for i, tag in enumerate(all_tags)}

    # Recursion for subsequent words
    for k in range(len(sentence)):
        cur_word = sentence[k]
        for tag in all_tags:
            if cur_word not in train_words and smoothing_param == 0 :
                emission_probabilities[DEFAULT_TAG][cur_word] = 1
            if k == 0:  # start of a sentence
                dp[k][tag_to_index[tag]] = transition_probabilities.get(START_TAG, {}).get(tag,
                                                                                           unknown) * \
                                           emission_probabilities.get(tag, {}).get(cur_word,
                                                                                   unknown)
                continue
            idx_of_best_prev_tag = 0
            max_prob = 0
            for j, prev_tag in enumerate(all_tags):
                cur_prob = (dp[k - 1][j] * transition_probabilities.get(prev_tag, {}).get(tag,
                                                                                         unknown)
                            * emission_probabilities.get(tag, {}).get(cur_word, unknown))
                if cur_prob > max_prob:
                    idx_of_best_prev_tag = j
                    max_prob = cur_prob
            dp[k][tag_to_index[tag]] = max_prob
            backpointers[k][tag_to_index[tag]] = idx_of_best_prev_tag

    # updating the prob with the probabilties for sentence end
    dp[-1] = [dp[-1][i] * transition_probabilities.get(tag, {}).get(END_TAG, unknown) for i, tag in
              enumerate(all_tags)]

    # Backtracking to find the best sequence
    best_last_tag_indx = np.argmax(dp[-1])
    all_tags = list(all_tags)
    best_sequence = [all_tags[best_last_tag_indx]]

    for k in range(len(sentence) - 1, 0, -1):
        best_last_tag_indx = tag_to_index[best_sequence[-1]]
        best_prev_tag = backpointers[k][best_last_tag_indx]
        best_sequence.append(all_tags[best_prev_tag])

    best_sequence.reverse()  # we started from the end

    return best_sequence


def run_viterbi_on_test_set(test_set1, transition_probabilities, emission_probabilities, all_tags,
                            train_words1, smoothing_param=0,known_words = None):
    # Initialize counters for error rates
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    total_correct = 0
    total_words = 0
    predict_output = {"word":[],"tag":[],"pred":[]}
    if known_words is None:
      known_words = train_words1

    for sentence in test_set1:
        # Separate words and true tags
        words = [word for word, _ in sentence]
        true_tags = [tag for _, tag in sentence]

        # Run Viterbi algorithm to get predicted tags
        predicted_tags = viterbi_algorithm(words, transition_probabilities, emission_probabilities,
                                           all_tags, train_words1, smoothing_param)

        # Compare true tags with predicted tags
        for true_tag, predicted_tag, word in zip(true_tags, predicted_tags, words):
            predict_output["word"].append(word)
            predict_output["tag"].append(true_tag)
            predict_output["pred"].append(predicted_tag)
            total_words += 1
            if word in known_words:  # Known word
                known_total += 1
                if true_tag == predicted_tag:
                    known_correct += 1
            else:  # Unknown word
                unknown_total += 1
                if true_tag == predicted_tag:
                    unknown_correct += 1
            if true_tag == predicted_tag:
                total_correct += 1

    # Calculate error rates
    known_accuracy = known_correct / known_total if known_total > 0 else 0
    unknown_accuracy = unknown_correct / unknown_total if unknown_total > 0 else 0
    overall_accuracy = total_correct / total_words

    known_error_rate = 1 - known_accuracy
    unknown_error_rate = 1 - unknown_accuracy
    overall_error_rate = 1 - overall_accuracy

    # Print results
    print("Bigram HMM Error Rates (Viterbi on Test Set):")
    print(f"Known words error rate: {known_error_rate:.4f}")
    print(f"Unknown words error rate: {unknown_error_rate:.4f}")
    print(f"Overall error rate: {overall_error_rate:.4f}")
    return predict_output


def assign_pseudo_word(word):
    # Numbers and special characters
    if re.match(r'^\d+$', word):
        return "<NUMBER>"
    if re.search(r'\d', word):
        return "<ALPHANUMERIC>"

    # Capitalization
    if word[0].isupper():
        return "<PROPER-NOUN>"

    # Suffix-based categorization
    if word.endswith("ing"):
        return "<VERB-ING>"
    if word.endswith("ed"):
        return "<VERB-PAST>"
    if word.endswith("s") and len(word) > 1:
        return "<PLURAL-NOUN>"
    if word.endswith("tion"):
        return "<NOMINALIZATION>"

    # Word length
    if len(word) <= 2:
        return "<SHORT-WORD>"
    if len(word) > 10:
        return "<LONG-WORD>"

    # Default: Unknown word
    return "<UNKNOWN>"



def get_unknown_and_low_freq_words(df_train, df_test, freq_threshold=5):
    unknown_words = df_test[~df_test["name"].isin(df_train["name"])]["name"].tolist()
    low_freq_words = df_test.groupby("name").value_counts().to_frame("cnt").query(
        "cnt < @freq_threshold").reset_index()["name"].tolist()
    return unknown_words + low_freq_words


# Example usage

def __update_data_set_with_pseudo_words(train_set, test_set, words_for_pseudo_tagging,
                                        pseudo_words):
    new_train = []
    for sentence in train_set:
        new_sentence = []
        for word, tag in sentence:

            if word in words_for_pseudo_tagging:
                # Replace word with its pseudo-word and retain the tag
                new_sentence.append((word, pseudo_words[word]))
            else:
                # Keep the original word and tag
                new_sentence.append((word, tag))
        new_train.append(new_sentence)

    new_test = []
    for i, sentence in enumerate(test_set):
        new_sentence = []
        for word, tag in sentence:
            if word in words_for_pseudo_tagging:
                # Replace word with its pseudo-word and retain the tag
                new_sentence.append((word, pseudo_words[word]))
                # If the word is unknown (from test set) and not in the training set,
                # add it to training data
                if word not in train_set:
                    new_train.append([(word, pseudo_words[word])])
            else:
                # Keep the original word and tag
                new_sentence.append((word, tag))
        new_test.append(new_sentence)

    return new_train, new_test


def create_pseudo_words(train_set, test_set, freq_threshold=5):
    # Count word frequencies
    df_train, df_test = turn_set_to_data_frame(train_set), turn_set_to_data_frame(test_set)
    words_for_pseudo_tagging = get_unknown_and_low_freq_words(df_train, df_test, freq_threshold)
    # Assign pseudo-words to each word
    pseudo_words = {word: assign_pseudo_word(word) for word in words_for_pseudo_tagging}
    new_train, new_test = __update_data_set_with_pseudo_words(train_set, test_set,
                                                              words_for_pseudo_tagging,
                                                              pseudo_words)
    return new_train, new_test


def make_sorted_dict(df_train1):
    count_per_tag = df_train1.groupby(["name", "tag"]).size().reset_index(name='appearances')
    count_per_word = df_train1.groupby(["name"]).size().reset_index(name='total_word_appearances')
    count_per_tag = count_per_tag.merge(count_per_word, on="name", how="left")
    count_per_tag["rate"] = count_per_tag["appearances"] / count_per_tag["total_word_appearances"]

    count_dict = {}

    for _, row in count_per_tag.iterrows():
        word = row['name']
        tag = row['tag']
        appearances = row['rate']

        if word not in count_dict:
            count_dict[word] = []

        # Append the (tag, appearances) tuple
        count_dict[word].append((tag, appearances))

    # Sort each list of tags by appearances in descending order
    for word in count_dict:
        count_dict[word] = sorted(count_dict[word], key=lambda x: x[1], reverse=True)

    return count_dict


def calculate_error_rates(df_train1, df_test1):
    count_dict = make_sorted_dict(df_train1)

    # Add a column for predicted tags
    df_test1["predicted_tag"] = df_test1["name"].apply(lambda word: get_prob(word, count_dict))

    # Determine if a word is known or unknown
    known_words = set(df_train1["name"].unique())
    df_test1["is_known"] = df_test1["name"].apply(lambda word: word in known_words)

    # Calculate accuracy for known and unknown words
    known_df = df_test1[df_test1["is_known"]]
    unknown_df = df_test1[~df_test1["is_known"]]

    known_accuracy = len(known_df[known_df["tag"] == known_df["predicted_tag"]]) / len(
        known_df) if len(known_df) > 0 else 0
    unknown_accuracy = len(unknown_df[unknown_df["tag"] == unknown_df["predicted_tag"]]) / len(
        unknown_df) if len(unknown_df) > 0 else 0

    overall_accuracy = len(df_test1[df_test1["tag"] == df_test1["predicted_tag"]]) / len(df_test1)

    known_error_rate1 = 1 - known_accuracy
    unknown_error_rate1 = 1 - unknown_accuracy
    overall_error_rate1 = 1 - overall_accuracy

    return known_error_rate1, unknown_error_rate1, overall_error_rate1


def get_prob(word, count_dict):
    if count_dict.get(word):
        return count_dict[word][0][0]
    else:
        return "NN"


def turn_set_to_data_frame(data):
    list_of_pd = []
    for i in range(len(data)):
        list_of_pd.append(pd.DataFrame(data[i]))
    df = pd.concat(list_of_pd)
    df.columns = ["name", "tag"]
    return df


def get_all_tags(train_set1):
    tags = set()
    for sentence in train_set1:
        for _, tag in sentence:
            tags.add(tag)
    return tags

def plot_heatplot(confusion_mat):
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming `confusion_mat` is a DataFrame
    pivot_table = pd.pivot(confusion_mat, index="tag", columns="pred", values="word").fillna(0)

    # Plot the heatmap
    plt.figure(figsize=(18, 12))
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".0f", 
        cmap="Reds",  # Red gradient from white (low) to red (high)
        cbar=True,
        linewidths=0.5  # Optional: adds small lines between cells for clarity
    )

    # Add labels and title
    plt.title("Confusion Matrix Heatmap", fontsize=16)
    plt.xlabel("Predicted Tags", fontsize=12)
    plt.ylabel("True Tags", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Load the data
    train_set, test_set = load_data()

    # Convert to DataFrames
    df_train = turn_set_to_data_frame(train_set)
    df_test = turn_set_to_data_frame(test_set)

    # Calculate error rates
    known_error_rate, unknown_error_rate, overall_error_rate = calculate_error_rates(df_train,
                                                                                     df_test)

    # Print error rates
    print("\nQ3b (ii)")
    print(f"Known words error rate: {known_error_rate:.4f}")
    print(f"Unknown words error rate: {unknown_error_rate:.4f}")
    print(f"Overall error rate: {overall_error_rate:.4f}")

    print("\nQ3c (iii) viterbi algorithm")
    
    transition_probabilities, emission_probabilities = compute_transition_emission_probabilities(
        train_set, test_set)
    all_tags = get_all_tags(train_set)
    train_words = set(word for sentence in train_set for word, _ in sentence)
    preidcted_output = run_viterbi_on_test_set(test_set, transition_probabilities, emission_probabilities, all_tags,
                            train_words, smoothing_param=0)

    # Question d
    print("\nQ3d (ii) viterbi algorithm with add-1 smoothing")
    transition_probabilities2, emission_probabilities2 = compute_transition_emission_probabilities(
        train_set,test_set, smoothing_param=1)
    q4 = run_viterbi_on_test_set(test_set, transition_probabilities2, emission_probabilities2,
                            all_tags, train_words, smoothing_param=1)

    print("\nQ3e (ii) viterbi algorithm with pseudo words")
    pseudo_train, pseudo_test = create_pseudo_words(train_set, test_set, 5)
    transition_probabilities_pseudo, emission_probabilities_pseudo = (
        compute_transition_emission_probabilities(
        pseudo_train, pseudo_test))
    all_tags_pseudo = get_all_tags(pseudo_train)
    train_words_pseudo = set(word for sentence in pseudo_train for word, _ in sentence)
    q5_dict = run_viterbi_on_test_set(pseudo_test, transition_probabilities_pseudo,
                            emission_probabilities_pseudo, all_tags_pseudo,
                            train_words_pseudo,known_words=train_words)

    
    print("\nQ3e (iii)  viterbi algorithm with pseudo words and add-1 smoothing")
    transition_probabilities_pseudo, emission_probabilities_pseudo = (
        compute_transition_emission_probabilities(
            pseudo_train,pseudo_test , smoothing_param=1))
    all_tags_pseudo = get_all_tags(pseudo_train)
    q6_dict = run_viterbi_on_test_set(pseudo_test, transition_probabilities_pseudo,
                            emission_probabilities_pseudo, all_tags_pseudo,
                            train_words_pseudo, smoothing_param=1,known_words = train_words)
    q6 = pd.DataFrame(q6_dict)
    confusion_mat = q6.groupby(["tag","pred"])["word"].nunique().reset_index()
    confusion_mat = confusion_mat[confusion_mat["word"]>0]
    plot_heatplot(confusion_mat)
    

