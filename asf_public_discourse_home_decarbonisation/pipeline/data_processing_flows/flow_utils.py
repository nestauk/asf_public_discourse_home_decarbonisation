"""
Util functions.
"""

import pandas as pd
import boto3
import sys
import logging
from fnmatch import fnmatch
import json
import pickle
import boto3
from typing import List, Union, Dict
import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")
import re
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
from gensim.parsing.preprocessing import STOPWORDS

logger = logging.getLogger(__name__)

S3_BUCKET = "asf-public-discourse-home-decarbonisation"


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def load_s3_data(bucket_name: str, file_path: str) -> Union[pd.DataFrame, Dict]:
    """
    Load data from S3 location.

    Args:
        bucket_name (str) : The S3 bucket name
        file_path (str): S3 key to load
    Returns:
        Union[pd.DataFrame, Dict]: the loaded data
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, file_path)
    if fnmatch(file_path, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_path, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_path)
    elif fnmatch(file_path, "*.parquet"):
        return pd.read_parquet("s3://" + bucket_name + "/" + file_path)
    elif fnmatch(file_path, "*.pkl") or fnmatch(file_path, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.json", "*.pkl" or "*.parquet"'
        )


def get_mse_category_data(category: str, collection_date: str) -> pd.DataFrame:
    """
    Gets data from a specific Money Saving Expert category, collected on a certain date.

    Args:
        category (str): An MSE category
        collection_date (str): A date in the format "YYYY_MM_DD"
    Returns:
        pd.DataFramedataframe with most up to date category/sub-forum data
    """
    if category == "energy":
        return load_s3_data(
            bucket_name=S3_BUCKET,
            file_path=f"data/mse/outputs/mse_data_category_energy.parquet",
        )

    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path=f"data/mse/outputs/mse_data_category_{category}_{collection_date}.parquet",
    )


def get_all_mse_data(collection_date: str) -> pd.DataFrame:
    """
    Gets data from all Money Saving Expert categories.

    Args:
        collection_date (str): A date in the format "YYYY_MM_DD"
    Returns:
        pd.DataFrame: a dataframe with data from all categories
    """
    categories = [
        "green-ethical-moneysaving",
        "lpg-heating-oil-solid-other-fuels",
        "energy",
        "is-this-quote-fair",
    ]
    all_mse_data = pd.DataFrame()
    for cat in categories:
        if (
            cat != "energy"
        ):  # this is a temporary fix, while we sort the collection for this category
            aux = load_s3_data(
                bucket_name=S3_BUCKET,
                file_path=f"data/mse/outputs/mse_data_category_{cat}_{collection_date}.parquet",
            )
        else:
            aux = load_s3_data(
                bucket_name=S3_BUCKET,
                file_path=f"data/mse/outputs/mse_data_category_energy.parquet",
            )
        all_mse_data = pd.concat([all_mse_data, aux])

    return all_mse_data.drop_duplicates().reset_index(drop=True)


def get_mse_data(category: str, collection_datetime: str) -> pd.DataFrame:
    """
    Get MSE data for a specific category and collection date.
    Existing categories are: "green-ethical-moneysaving", "lpg-heating-oil-solid-other-fuels", "energy", "is-this-quote-fair"
    Additionally, you can also use "sample" (to get the initial sample collected) or "all" (to get all the MSE data collected).

    Args:
        category (str): An MSE category, "sample" or "all"
        collection_datetime (str): A date in the format "YYYY_MM_DD"

    Returns:
        pd.DataDrame: a dataframe with the MSE data
    """
    mse_categories = [
        "green-ethical-moneysaving",
        "lpg-heating-oil-solid-other-fuels",
        "energy",
        "is-this-quote-fair",
    ]

    accepted_categories = mse_categories + ["all"]
    try:
        # Check if the category is in the list if one of the accepted categories
        category_index = accepted_categories.index(category)
        if category in [
            "green-ethical-moneysaving",
            "lpg-heating-oil-solid-other-fuels",
            "is-this-quote-fair",
            "energy",
        ]:
            mse_data = get_mse_category_data(category, collection_datetime)
        else:  # category == "all"
            mse_data = get_all_mse_data(collection_datetime)
        logger.info(f"Data for category '{category}' imported successfully from S3.")
    except ValueError:
        logger.error(f"{category} is not a valid category!")
        sys.exit(-1)

    return mse_data


def remove_urls(text: str) -> str:
    """
    Removes URLs from text.
    Args:
        text (str): a string, tipically one or multiple sentences long
    Returns:
        str: text without URLs
    """
    pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    cleaned_text = re.sub(pattern, " ", text)
    return cleaned_text


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


def remove_items_in_list(mse_data: pd.DataFrame, list_of_items: list) -> pd.DataFrame:
    """
    Removes items from text and titles.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        list_of_items (list): list of items to be removed
    Returns:
        pd.DataFrame: MSE data with items removed from text and titles
    """
    mse_data["tokens_title"] = mse_data["tokens_title"].apply(
        lambda x: [token for token in x if token not in list_of_items]
    )
    mse_data["tokens_text"] = mse_data["tokens_text"].apply(
        lambda x: [token for token in x if token not in list_of_items]
    )
    return mse_data


def english_stopwords_definition() -> list:
    """
    Defines English stopwords by putting together NLTK and gensim stopwords.

    Returns:
        list: a list of English stopwords.
    """
    sw_nltk = stopwords.words("english")
    sw_gensim = [s for s in STOPWORDS if s not in sw_nltk]

    stopwords_list = sw_nltk + sw_gensim

    return stopwords_list


def process_text(text: str) -> str:
    """
    Preprocesses text by:
        - replacing &amp with "and"
        - removing URLs
        - puts text to lower case
        - removing username patterns

    Args:
        text:
    Returns:
        Preprocessed text.
    """
    text = re.sub("&amp;", " and ", text)
    text = remove_urls(text)

    text = text.lower()

    text = remove_text_after_patterns(text)

    return text


def lemmatise(text: str) -> list:
    """
    Applies lemmatisation sentence by sentence.
    It then removes punctuations, stopwords and tokens tagged as space (e.g. "\n")

    Args:
        text (str): any text

    Returns:
        list: list of lemmatised token without stopwords or punctuation
    """
    doc = nlp(text)
    # apply sentence by sentence lemmatisation
    processed_sentences = []
    for sentence in doc.sents:
        lemmatized_tokens = [
            token.lemma_
            for token in sentence
            if not (token.is_punct or token.is_stop or token.pos_ == "SPACE")
        ]
        processed_sentences.extend(lemmatized_tokens)
    return processed_sentences
