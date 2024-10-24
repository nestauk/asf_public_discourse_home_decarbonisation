import pandas as pd
from nltk.corpus import stopwords

from nltk.util import bigrams, trigrams, ngrams
from collections import Counter
import re
from typing import List, Tuple, Dict
from nltk.probability import FreqDist
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_abbreviations,
)


######### 3. Word cloud of frequently used words in posts ############
def preprocess_text(dataframe: pd.DataFrame, custom_stopwords: List[str]) -> List[str]:
    """
    Preprocesses text data before text analysis by:
        - applying .lower() to text
        - removing stopwords
        - tokenizing text
    Outputs the resulting list of tokens.
    Args:
        dataframe (pandas.DataFrame): The DataFrame containing text data in a column named 'text'.
        custom_stopwords (list): A list of additional stopwords to be removed from the text.

    Returns:
        List[str]: A list of filtered tokens.
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


def process_ngrams(raw_ngram_freq_dist: FreqDist, ngram_threshold: float) -> FreqDist:
    """
    Filters n-grams based on a specified threshold frequency.
    Args:
        raw_ngram_freq_dist (FreqDist): A frequency distribution of n-grams where each n-gram is a tuple (before filtering).
        ngram_threshold (float): The threshold frequency for n-grams.
    Returns:
        ngram_freq_dist (FreqDist): A frequency distribution of n-grams where each n-gram is a tuple (after filtering).
    """
    ngram_freq_dist = {
        ngram: freq
        for ngram, freq in raw_ngram_freq_dist.items()
        if freq >= ngram_threshold
    }

    return ngram_freq_dist


######## 5.Generate bigram and trigram word clouds ########
def wordcloud_preprocess_ngrams(
    ngram_freq_dists: List[FreqDist],
) -> (List[FreqDist], List[FreqDist]):
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
def update_keyword_frequencies(
    dataframe: pd.DataFrame, text_column: str, ruleset: List[Dict[str, str]]
) -> Counter:
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
    keyword_counter: Counter,
    df_threshold: float,
) -> Tuple[pd.DataFrame, int]:
    """
    Prepares a DataFrame for plotting based on keyword frequencies and applies a frequency threshold.

    This function converts a Counter object containing keyword frequencies into a DataFrame.
    It then filters the DataFrame based on a minimum frequency threshold, which is either a specified
    percentage of the total number of rows or a fixed minimum count, whichever is higher.

    Args:
        keyword_counter (collections.Counter): A Counter object with keywords as keys and their
                                               frequencies as values.
        df_threshold (float): The minimum frequency threshold.

    Returns:
        - A pandas.DataFrame containing tags and their frequencies, filtered by the calculated threshold.

    The DataFrame has columns 'Tag' and 'Frequency'. Tags with frequencies below the threshold are excluded.
    """
    keyword_df = pd.DataFrame.from_dict(
        keyword_counter, orient="index", columns=["Frequency"]
    ).reset_index()
    keyword_df.columns = ["Tag", "Frequency"]
    keyword_df["Tag"] = keyword_df["Tag"].str.replace("_", " ")
    keyword_df = keyword_df.sort_values(by="Frequency", ascending=False)
    print(keyword_df)
    return keyword_df[keyword_df["Frequency"] > df_threshold]


def preprocess_data_for_linechart_over_time(
    df: pd.DataFrame,
    start_date: str = "2018-01-01",
    key_terms: List[str] = ["heat pump", "boiler"],
) -> pd.DataFrame:
    """
    Preprocess the data for analysis.

    Parameters:
    df (pd.DataFrame): The input dataframe with 'date' and 'text' columns.
    start_date (str): The start date for filtering the data.
    key_terms (List[str]): List of key terms to search for in the text.

    Returns:
    pd.DataFrame: The preprocessed dataframe.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= start_date]
    df["text"] = df["text"].fillna("")
    df["text"] = df["text"].apply(process_abbreviations)
    df.set_index("date", inplace=True)

    # Create new columns for each key term
    for term in key_terms:
        column_name = f"mentions_{term.replace(' ', '_')}"
        df[column_name] = df["text"].str.contains(term, case=False).astype(int)

    return df


def resample_and_calculate_averages_for_linechart(
    df: pd.DataFrame,
    key_terms: List[str] = ["heat pump", "boiler"],
    cadence_of_aggregation="M",
    window: int = 12,
) -> pd.DataFrame:
    """
    Resample the data into months and calculate rolling averages.

    Parameters:
    df (pd.DataFrame): The input dataframe with mentions columns.
    key_terms (List[str]): List of key terms for which rolling averages will be calculated.
    window (int): The window size for calculating the rolling average.

    Returns:
    pd.DataFrame: The resampled and averaged dataframe.
    """
    df_monthly = df.resample(cadence_of_aggregation).sum()

    # Calculate rolling averages for each key term
    for term in key_terms:
        column_name = f"mentions_{term.replace(' ', '_')}"
        avg_column_name = f"{column_name}_avg"
        df_monthly[avg_column_name] = (
            df_monthly[column_name].rolling(window=window).mean()
        )

    return df_monthly


def calculate_ngram_threshold(tokens: List[str], n: int, freq_multiplier: float) -> int:
    """
    Calculates and returns the frequency threshold for n-grams.

    Args:
        tokens (List[str]): A list of tokens from which n-grams are generated.
        n (int): The 'n' in n-grams, representing the number of elements in each gram.
        freq_multiplier (float): The multiplier to calculate the frequency threshold.

    Returns:
        int: The calculated threshold for n-grams.
    """
    # Calculate initial frequency distribution for n-grams
    raw_ngram_freq_dist = FreqDist(ngrams(tokens, n))

    # Calculate total count and threshold for n-grams
    total_ngrams = sum(raw_ngram_freq_dist.values())
    ngram_threshold = round(max(3, total_ngrams * freq_multiplier))

    return raw_ngram_freq_dist, ngram_threshold
