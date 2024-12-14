from collections import defaultdict

import nltk
from nltk.corpus import brown
import pandas as pd


def load_data():
    # Ensure that the Brown corpus is downloaded
    nltk.download('brown')

    # Load the tagged sentences from the 'news' category
    news_sentences = brown.tagged_sents(categories='news')

    # Calculate the split index for the last 10% of the sentences
    split_index = int(len(news_sentences) * 0.9)

    # Divide the sentences into training and test sets
    train_set = news_sentences[:split_index]
    test_set = news_sentences[split_index:]

    # Optionally, you can print the sizes of the train and test sets
    print(f"Training set size: {len(train_set)} sentences")
    print(f"Test set size: {len(test_set)} sentences")
    return train_set, test_set


def compute_transition_emission_probabilities(train_set, smoothing_param=0):
    # Transition and emission counts
    transition_counts = defaultdict(lambda: defaultdict(int))
    emission_counts = defaultdict(lambda: defaultdict(int))
    tag_counts = defaultdict(int)
    word_set = set()

    # Process each sentence
    for sentence in train_set:
        previous_tag = '<s>'  # Start-of-sentence tag
        for word, tag in sentence:
            # Update counts
            transition_counts[previous_tag][tag] += 1
            emission_counts[tag][word] += 1
            tag_counts[tag] += 1
            previous_tag = tag

            if smoothing_param:
                word_set.add(word)

        # Handle end-of-sentence transition
        transition_counts[previous_tag]['</s>'] += 1

    # Compute probabilities
    transition_probabilities = {}
    for prev_tag, next_tags in transition_counts.items():
        total = sum(next_tags.values())
        transition_probabilities[prev_tag] = {tag: count / total for tag, count in
                                              next_tags.items()}

    emission_probabilities = {}
    vocab_size = len(word_set)
    for tag, words in emission_counts.items():
        total = tag_counts[tag]
        emission_probabilities[tag] = {
            word: (count + smoothing_param) / (total + smoothing_param * vocab_size)
            # Add smoothing to the denominator
            for word, count in words.items()
        }

    return transition_probabilities, emission_probabilities


def viterbi_algorithm(sentence, transition_probabilities, emission_probabilities, all_tags):
    n = len(sentence)
    dp = [{} for _ in range(n)]  # dp[k][tag] = max probability of the k-th word ending with `tag`
    backpointer = [{} for _ in range(n)]

    # Initialization for the first word
    for tag in all_tags:
        transition_prob = transition_probabilities.get('<s>', {}).get(tag, 1e-6)
        emission_prob = emission_probabilities.get(tag, {}).get(sentence[0], 1e-6)
        dp[0][tag] = transition_prob * emission_prob
        backpointer[0][tag] = None

    # Recursion for subsequent words
    for k in range(1, n):
        for tag in all_tags:
            max_prob, best_prev_tag = 0, None
            for prev_tag in all_tags:
                prob = dp[k - 1].get(prev_tag, 1e-6) * transition_probabilities.get(prev_tag,
                                                                                    {}).get(tag,
                                                                                            1e-6)
                if prob > max_prob:
                    max_prob = prob
                    best_prev_tag = prev_tag
            dp[k][tag] = max_prob * emission_probabilities.get(tag, {}).get(sentence[k], 1e-6)
            backpointer[k][tag] = best_prev_tag

    # Termination: Find the best path
    max_final_prob = 0
    best_final_tag = None
    for tag in all_tags:
        prob = dp[n - 1].get(tag, 1e-6) * transition_probabilities.get(tag, {}).get('</s>', 1e-6)
        if prob > max_final_prob:
            max_final_prob = prob
            best_final_tag = tag

    # Backtracking to find the best sequence
    best_sequence = [best_final_tag]
    for k in range(n - 1, 0, -1):
        best_sequence.append(backpointer[k][best_sequence[-1]])
    best_sequence.reverse()

    return best_sequence


def run_viterbi_on_test_set(test_set, transition_probabilities, emission_probabilities, all_tags,
                            train_words):
    # Initialize counters for error rates
    known_correct = 0
    known_total = 0
    unknown_correct = 0
    unknown_total = 0
    total_correct = 0
    total_words = 0

    for sentence in test_set:
        # Separate words and true tags
        words = [word for word, _ in sentence]
        true_tags = [tag for _, tag in sentence]

        # Run Viterbi algorithm to get predicted tags
        predicted_tags = viterbi_algorithm(words, transition_probabilities, emission_probabilities,
                                           all_tags)

        # Compare true tags with predicted tags
        for true_tag, predicted_tag, word in zip(true_tags, predicted_tags, words):
            total_words += 1
            if word in train_words:  # Known word
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


def make_sorted_dict(df_train):
    count_per_tag = df_train.groupby(["name", "tag"]).size().reset_index(name='appearances')
    count_per_word = df_train.groupby(["name"]).size().reset_index(name='total_word_appearances')
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


def calculate_error_rates(df_train, df_test):
    count_dict = make_sorted_dict(df_train)

    # Add a column for predicted tags
    df_test["predicted_tag"] = df_test["name"].apply(lambda word: get_prob(word, count_dict))

    # Determine if a word is known or unknown
    known_words = set(df_train["name"].unique())
    df_test["is_known"] = df_test["name"].apply(lambda word: word in known_words)

    # Calculate accuracy for known and unknown words
    known_df = df_test[df_test["is_known"]]
    unknown_df = df_test[~df_test["is_known"]]

    known_accuracy = len(known_df[known_df["tag"] == known_df["predicted_tag"]]) / len(
        known_df) if len(known_df) > 0 else 0
    unknown_accuracy = len(unknown_df[unknown_df["tag"] == unknown_df["predicted_tag"]]) / len(
        unknown_df) if len(unknown_df) > 0 else 0

    overall_accuracy = len(df_test[df_test["tag"] == df_test["predicted_tag"]]) / len(df_test)

    known_error_rate = 1 - known_accuracy
    unknown_error_rate = 1 - unknown_accuracy
    overall_error_rate = 1 - overall_accuracy

    return known_error_rate, unknown_error_rate, overall_error_rate


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


def get_all_tags(train_set):
    tags = set()
    for sentence in train_set:
        for _, tag in sentence:
            tags.add(tag)
    return tags


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

    # print("\nQ3c (iii)")
    # transition_probabilities, emission_probabilities = compute_transition_emission_probabilities(
    #     train_set)
    all_tags = get_all_tags(train_set)
    train_words = set(word for sentence in train_set for word, _ in sentence)
    # run_viterbi_on_test_set(test_set, transition_probabilities, emission_probabilities, all_tags,
    #                         train_words)


    # Question d
    print("\nQ3d (ii)")
    transition_probabilities2, emission_probabilities2 = compute_transition_emission_probabilities(
        train_set, smoothing_param=1)
    run_viterbi_on_test_set(test_set, transition_probabilities2, emission_probabilities2, all_tags,
                            train_words)
