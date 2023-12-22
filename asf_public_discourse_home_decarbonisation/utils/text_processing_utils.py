"""
Utils for dealing with n-grams.
"""

from nltk import ngrams
import pandas as pd
from collections import Counter


def create_ngram_from_ordered_tokens(tokens: list, n: int) -> list:
    """
    Creates a list of n-grams from a list of ordered tokens.
    Args:
        tokens: a list of tokens
        n: number of tokens considered for n-gram
    Returns:
        list: list of n-grams
    """
    # Generate n-grams
    n_grams = ngrams(tokens, n)

    # Join the n-grams into strings
    n_gram_strings = [" ".join(gram) for gram in n_grams]

    return n_gram_strings


def frequency_ngrams(data: pd.DataFrame, ngrams_col: str) -> Counter:
    """
    Computes the frequency of ngrams.

    Args:
        data (pd.DataFrame): Dataframe containing the ngram data
        ngrams_col (str): Name of column containing the ngrams

    Returns:
        Counter: The counter containing the frequency of ngrams
    """
    ngrams = [ng for sublist in data[ngrams_col].tolist() for ng in sublist]
    frequency_ngrams = Counter(ngrams)

    return frequency_ngrams


def identify_n_gram_type(frequency_dict: dict) -> str:
    """
    Identifies the type of n-gram based on the size of the n-gram.

    Args:
        frequency_dict (dict): Dictionary containing the frequency of n-grams
    Returns:
        str: The type of n-gram e.g. "tokens", "bigrams", "trigrams", "{n}-grams" when n >=4
    """
    size_ngram = len(next(iter(frequency_dict)).split(" "))
    if size_ngram == 1:
        n_gram_type = "tokens"
    elif size_ngram == 2:
        n_gram_type = "bigrams"
    elif size_ngram == 3:
        n_gram_type = "trigrams"
    else:
        n_gram_type = f"{size_ngram}-grams"

    return n_gram_type
