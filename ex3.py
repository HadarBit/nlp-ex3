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


def make_sorted_dict(df_train):
    count_per_tag = df_train.groupby(["name", "tag"]).size().reset_index(name='appearances')
    count_per_word = df_train.groupby(["name"]).size().reset_index(name='total_word_appearances')
    count_per_tag = count_per_tag.merge(count_per_word, on="name", how="left")
    count_per_tag["rate"] = count_per_tag["appearances"] / count_per_tag["total_word_appearances"]

    # put in different funs
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


def error_rate_using_likelyhood(df_train, df_test):
    count_dict = make_sorted_dict(df_train)
    df_test["predicted tag"] = df_test.apply(lambda x: get_prob(x["name"], count_dict), axis=1)
    accuracy = len(df_test[df_test["tag"] == df_test["predicted tag"]]) / len(df_test)
    error_rate = 1 - accuracy
    return error_rate


def get_prob(word, count_dict):
    if count_dict.get(word):
        return count_dict[word][0][0]
    else:
        return "NN"


def turn_set_to_data_frame(data):
    list_of_pd = []
    for i in range(len(train_set)):
        list_of_pd.append(pd.DataFrame(train_set[i]))
    df = pd.concat(list_of_pd)
    df.columns = ["name", "tag"]
    return df


if __name__ == "__main__":
    # question a
    train_set, test_set = load_data()

    # question b
    df_train, df_test = turn_set_to_data_frame(train_set), turn_set_to_data_frame(test_set)
    error_rate = error_rate_using_likelyhood(df_train, df_test)