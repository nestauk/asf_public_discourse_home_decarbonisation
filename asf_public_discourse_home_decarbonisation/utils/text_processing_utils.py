"""
Utils for processing text data.
"""

import re
from nltk.corpus import stopwords
from nltk import ngrams
import nltk

nltk.download("stopwords")
from gensim.parsing.preprocessing import STOPWORDS
import re
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag

from nltk import pos_tag
from nltk.corpus import wordnet

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
import pandas as pd
from collections import Counter


def remove_urls(text: str) -> str:
    """
    Removes URLs from text.
    Args:
        text: a string, tipically one or multiple sentences long
    Returns:
        text without URLs
    """
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    cleaned_text = re.sub(pattern, " ", text)
    return cleaned_text


def replace_punctuation_with_space(text: str) -> str:
    """
    Replaces punctuation with space.
    Args:
        text: a string, tipically one or multiple sentences long
    Returns:
        text wihtout punctuation
    """
    pattern = r"[{}]".format(re.escape(string.punctuation))
    replaced_text = re.sub(pattern, " ", text)
    replaced_text = re.sub(r"\s+", " ", replaced_text)  # Remove consecutive spaces
    return replaced_text.strip()  # Remove leading/trailing spaces


def remove_text_after_patterns(text: str) -> str:
    """
    Removes pattern of the form "xxx writes: ".

    Args:
        text (str): text to be cleaned

    Returns:
        str: cleaned text
    """
    # We use re.sub() to replace the pattern with an empty string
    result = re.sub(r"\w+ wrote »", " ", text)
    return result


def identify_part_of_spech(pos_tag: str) -> str:
    """
    Convert the tag given by nltk.pos_tag to the tag used by wordnet.

    Args:
        pos_tag (str): NLTk part of speach tag

    Returns:
        str: WordNet part os speach tag
    """
    if pos_tag.startswith("N"):
        return wordnet.NOUN
    elif pos_tag.startswith("V"):
        return wordnet.VERB
    elif pos_tag.startswith("R"):
        return wordnet.ADV
    elif pos_tag.startswith("J"):
        return wordnet.ADJ
    else:
        return wordnet.NOUN  # Default to NOUN if the part of speech is not recognized


def lemmatize_sentence(tokens: list):
    """
    Lemmatizes tokens using WordNetLemmatizer.

    Args:
        tokens (list): a list of tokens

    Returns:
        list: a list of lemmatised tokens
    """
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    lemmatized_words_dict = {
        word: lemmatizer.lemmatize(word, identify_part_of_spech(pos_tag))
        for word, pos_tag in pos_tags
    }
    return lemmatized_words_dict


def process_text(text: str) -> str:
    """
    Preprocesses text by:
        - replacing &amp with "and"
        - removing URLs
        - puts text to lower case
        - replaces punctuation with space
        - applies lemmatisation

    Args:
        text:
    Returns:
        Preprocessed text.
    """
    text = re.sub("&amp;", " and ", text)
    text = remove_urls(text)

    text = text.lower()

    text = remove_text_after_patterns(text)

    text = replace_punctuation_with_space(text)

    return text


def create_ngram_frequencies(tokens: list, n: int) -> list:
    """
    Computes frequencies from n-grams.
    Args:
        tokens: a list of tokens
        n: number of tokens considered for n-gram
    Returns:
        a list
    """
    # Generate n-grams (in this example)
    n_grams = ngrams(tokens, n)

    # Join the n-grams into strings
    n_gram_strings = [" ".join(gram) for gram in n_grams]

    return n_gram_strings


def english_stopwords_definition() -> list:
    """
    Defines English stopwords by putting together NLTK and gensim stopwords.

    Returns:
        A list of English stopwords.
    """
    sw_nltk = stopwords.words("english")
    sw_gensim = [s for s in STOPWORDS if s not in sw_nltk]

    stopwords_list = sw_nltk + sw_gensim

    return stopwords_list


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
        str: The type of n-gram e.g. "tokens", "bigrams", "trigrams", "{n}-gram" when n >=4
    """
    size_ngram = len(next(iter(frequency_dict)).split(" "))
    if size_ngram == 1:
        n_gram_type = "tokens"
    elif size_ngram == 2:
        n_gram_type = "bigrams"
    elif size_ngram == 3:
        n_gram_type = "trigrams"
    else:
        n_gram_type = f"{size_ngram}-gram"

    return n_gram_type
