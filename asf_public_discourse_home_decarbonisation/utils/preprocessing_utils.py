import pandas as pd
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams
from collections import Counter
import re
from typing import List
from nltk.probability import FreqDist


######### 3. Word cloud of frequently used words in posts ############
def preprocess_text(dataframe, custom_stopwords) -> List:
    """
    Preprocesses text data for word cloud generation.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing text data in a column named 'text'.
        custom_stopwords (list): A list of additional stopwords to be removed from the text.

    Returns:
        dict: A dictionary of word frequencies filtered based on the defined threshold.
    """
    # Combine and lowercase text data
    text_data = " ".join(dataframe["text"].astype(str)).lower()

    # Update stop words
    stop_words = set(stopwords.words("english"))
    stop_words.update(custom_stopwords)

    # Tokenize the text into words and remove stopwords
    tokens = text_data.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]

    return filtered_tokens


######## 4.Generate bigram and trigram frequency distributions ########


def process_ngrams(filtered_tokens, n):
    """
    Processes n-grams (bigrams or trigrams) from the given tokens and filters them based on a threshold.

    Args:
        filtered_tokens (list): A list of tokens from which to form n-grams.
        n (int): The size of the n-grams (2 for bigrams, 3 for trigrams).

    Returns:
        tuple: A tuple containing two dictionaries, the raw and filtered n-gram frequency distributions.
    """
    if n == 2:
        raw_ngram_freq_dist = FreqDist(bigrams(filtered_tokens))
        threshold_multiplier = 0.0002
    elif n == 3:
        raw_ngram_freq_dist = FreqDist(trigrams(filtered_tokens))
        threshold_multiplier = 0.00005
    else:
        raise ValueError("n must be either 2 (for bigrams) or 3 (for trigrams)")

    total_ngrams = sum(raw_ngram_freq_dist.values())
    ngram_threshold = round(max(3, total_ngrams * threshold_multiplier))

    ngram_freq_dist = {
        ngram: freq
        for ngram, freq in raw_ngram_freq_dist.items()
        if freq >= ngram_threshold
    }

    return raw_ngram_freq_dist, ngram_freq_dist, ngram_threshold


######## 5.Generate bigram and trigram word clouds ########
def wordcloud_preprocess_ngrams(ngram_freq_dists):
    """
    Converts n-gram frequency distributions into string frequency distributions.

    Args:
        ngram_freq_dists (list): A list of n-gram frequency distributions where each distribution
                                 is a dictionary with n-gram tuples as keys and their frequencies as values.

    Returns:
        list: A list of string frequency distributions corresponding to the input n-gram frequency distributions.
    """
    string_freq_dists = []

    for ngram_freq_dist in ngram_freq_dists:
        string_freq_dist = FreqDist()

        for ngram, freq in ngram_freq_dist.items():
            ngram_string = " ".join(ngram)
            string_freq_dist[ngram_string] += freq

        string_freq_dists.append(string_freq_dist)

    return string_freq_dists


############ 7. Frequency of selected keywords in posts using the dictionary ###########
# Function to update keyword frequencies
def update_keyword_frequencies(dataframe, text_column, ruleset) -> Counter:
    """
    Updates and returns the frequency of keywords based on a specified ruleset.

    This function scans through each text entry in a specified column of the DataFrame,
    searches for occurrences of each keyword as defined in the ruleset, and counts
    the total number of occurrences of each keyword.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing the text data.
        text_column (str): The name of the column in the DataFrame that contains the text data.
        ruleset (list of dicts): A list of dictionaries, where each dictionary has 'value' as the
                                 keyword (regular expression) and 'tag' as the associated tag
                                 for the keyword.

    Returns:
        collections.Counter: A Counter object mapping each tag to its frequency across all texts.

    The function uses regular expressions to identify the presence of keywords and is case-insensitive.
    Each occurrence of a keyword increments the count for its corresponding tag.
    """
    custom_keyword_counter = Counter()
    for text in dataframe[text_column]:
        for rule in ruleset:
            matches = re.findall(rule["value"], str(text), re.IGNORECASE)
            custom_keyword_counter[rule["tag"]] += len(matches)
    return custom_keyword_counter


# Function to prepare DataFrame for plotting
def prepare_keyword_dataframe(
    keyword_counter, total_rows, min_frequency_threshold=0.0001
) -> (pd.DataFrame, int):
    """
    Prepares a DataFrame for plotting based on keyword frequencies and applies a frequency threshold.

    This function converts a Counter object containing keyword frequencies into a DataFrame.
    It then filters the DataFrame based on a minimum frequency threshold, which is either a specified
    percentage of the total number of rows or a fixed minimum count, whichever is higher.

    Args:
        keyword_counter (collections.Counter): A Counter object with keywords as keys and their
                                               frequencies as values.
        total_rows (int): The total number of rows in the original dataset.
        min_frequency_threshold (float, optional): The minimum frequency threshold as a percentage
                                                   of total rows. Defaults to 0.0001 (0.01%).

    Returns:
        tuple:
            - A pandas.DataFrame containing tags and their frequencies, filtered by the calculated threshold.
            - An integer representing the frequency threshold used for filtering.

    The DataFrame has columns 'Tag' and 'Frequency'. Tags with frequencies below the threshold are excluded.
    """
    keyword_df = pd.DataFrame.from_dict(
        keyword_counter, orient="index", columns=["Frequency"]
    ).reset_index()
    keyword_df.columns = ["Tag", "Frequency"]
    keyword_df["Tag"] = keyword_df["Tag"].str.replace("_", " ")
    keyword_df = keyword_df.sort_values(by="Frequency", ascending=False)
    print(keyword_df)
    df_threshold = round(max(5, total_rows * min_frequency_threshold))
    return keyword_df[keyword_df["Frequency"] > df_threshold], df_threshold
