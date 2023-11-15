"""
Utils for processing text.
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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


# def lemmatize_sentence(tokens: list) -> list:
#     """
#     Lemmatizes tokens according to appropriate Part of Speech.
#     Args:
#         tokens: a list of tokens
#     Returns:
#         A list of lemmatized tokens.
#     """
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_tokens = []
#     for word, tag in pos_tag(tokens):
#         if tag.startswith("NN"):
#             pos = "n"
#         elif tag.startswith("VB"):
#             pos = "v"
#         else:
#             pos = "a"
#         lemmatized_tokens.append(lemmatizer.lemmatize(word, pos))
#     return lemmatized_tokens


def penn_to_wordnet(pos_tag):
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


def lemmatize_sentence_2(tokens):
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    lemmatized_words_dict = {
        word: lemmatizer.lemmatize(word, penn_to_wordnet(pos_tag))
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
