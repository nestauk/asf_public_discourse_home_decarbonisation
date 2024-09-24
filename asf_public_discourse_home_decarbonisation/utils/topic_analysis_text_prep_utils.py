"""
Utility functions for preparing/cleaning text data for topic analysis
"""

import pandas as pd
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from datetime import datetime

from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    remove_urls,
    remove_username_pattern,
    replace_username_mentions,
    remove_introduction_patterns,
    process_abbreviations,
    ends_with_punctuation,
)


def cleaning_and_enhancing_forum_data(forum_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and enhances forum data by:
    - Removing URLs
    - Removing username patterns
    - Replacing username mentions
    - Removing introduction patterns
    - Adding space after punctuation ("?", ".", "!"), so that sentences are correctly split
    - Processing abbreviations such as "ashp" to "air source heat pump"
    - Adding a column with the whole text (title + text)
    - Adding columns with the datetime, date, and year

    Args:
        forum_data (pd.DataFrame): forum data

    Returns:
        pd.DataFrame: enhanced forum data
    """
    # making text and title as strings
    forum_data["text"] = forum_data["text"].astype(str)
    forum_data["title"] = forum_data["title"].astype(str)

    # cleaning the text data
    forum_data["text"] = forum_data["text"].apply(remove_urls)
    forum_data["text"] = forum_data["text"].apply(remove_username_pattern)
    forum_data["text"] = forum_data["text"].apply(replace_username_mentions)
    forum_data["text"] = forum_data["text"].apply(remove_introduction_patterns)

    # title is only used for the original post
    forum_data["title"] = forum_data.apply(
        lambda x: "" if x["is_original_post"] == 0 else x["title"], axis=1
    )

    # Adding a column with the whole text (title + text)
    forum_data["whole_text"] = forum_data.apply(
        lambda x: (
            x["title"] + " " + x["text"]
            if (ends_with_punctuation(x["title"]) or x["is_original_post"] == 0)
            else x["title"] + ". " + x["text"]
        ),
        axis=1,
    )

    # Processing abbreviations such as "ashp" to "air source heat pump"
    forum_data["whole_text"] = forum_data["whole_text"].apply(process_abbreviations)

    # creating date/time variables
    forum_data["datetime"] = pd.to_datetime(forum_data["datetime"])
    forum_data["date"] = forum_data["datetime"].dt.date
    forum_data["year"] = forum_data["datetime"].dt.year

    # Adding space after punctuation ("?", ".", "!"), so that sentences are correctly split
    forum_data["whole_text"] = forum_data["whole_text"].apply(
        lambda t: re.sub(r"([.!?])([A-Za-z])", r"\1 \2", t)
    )

    return forum_data


def create_sentence_df(forum_data: pd.DataFrame) -> pd.DataFrame:
    """
    Breaks down the whole text into sentences and creates a dataframe with the sentences.

    Args:
        forum_data (pd.DataFrame): forum data

    Returns:
        pd.DataFrame: sentences data
    """
    forum_data["sentences"] = forum_data["whole_text"].apply(sent_tokenize)

    sentences_data = forum_data.explode("sentences")
    sentences_data["sentences"] = sentences_data["sentences"].astype(str)
    sentences_data["sentences"] = sentences_data["sentences"].str.strip()

    return sentences_data


def remove_small_sentences(
    sentences_data: pd.DataFrame, min_n_tokens: int = 5
) -> pd.DataFrame:
    """
    Removes small sentences with less than `min_n_tokens` tokens.

    Args:
        sentences_data (pd.DataFrame): sentences data
        min_n_tokens (int): minimum number of tokens
    Returns:
        pd.DataFrame: filtered sentences data
    """
    sentences_data["tokens"] = sentences_data["sentences"].apply(word_tokenize)
    sentences_data["non_punctuation_tokens"] = sentences_data["tokens"].apply(
        lambda x: [token for token in x if token not in string.punctuation]
    )

    sentences_data["n_tokens"] = sentences_data["non_punctuation_tokens"].apply(len)

    sentences_data = sentences_data[sentences_data["n_tokens"] > min_n_tokens]

    return sentences_data


def prepping_data_for_topic_analysis(
    forum_data: pd.DataFrame,
    filter_by_expression: str,
    start_date: int,
    end_date: int,
    phrases_to_remove: list = ["thank", "happy to help", "kind wishes", "kind regards"],
    only_keep_sentences_with_expression: bool = False,
) -> pd.DataFrame:
    """
    Prepares the data for topic analysis by:
    - Cleaning and enhancing the forum data
    - Filtering by expression
    - Transforming text into sentences
    - Removing small sentences
    - Removing sentences thanking people

    Args:
        forum_data (pd.DataFrame): dataframe with forum data
        filter_by_expression (str): expression to filter data by e.g. "heat pump". If None, all data is kept.
        start_date (int): start date
        end_date (int): end date
        phrases_to_remove (list): list of phrases to remove from the sentences. Defaults to ["thank", "happy to help", "kind wishes", "kind regards"
        only_keep_sentences_with_expression (bool): whether to only keep sentences containing the expression after creating the sentences df. Defaults to False.
    Returns:
        pd.DataFrame: dataframe with sentences data
    """
    # Data cleaning
    forum_data = cleaning_and_enhancing_forum_data(forum_data)

    if start_date is not None:
        forum_data = forum_data[
            forum_data["date"] >= datetime.strptime(start_date, "%Y-%m-%d").date()
        ]
    if end_date is not None:
        forum_data = forum_data[
            forum_data["date"] <= datetime.strptime(end_date, "%Y-%m-%d").date()
        ]

    # Focusing on conversations mentioning a certain expression e.g. "heat pump"
    if filter_by_expression is not None:
        if filter_by_expression == "solar panel":
            ids_to_keep = forum_data[
                (
                    forum_data["whole_text"].str.contains(
                        "|".join(
                            ["solar panel", "solar pv", "solar photovoltaic", " pv "]
                        ),
                        case=False,
                    )
                )
                & (forum_data["is_original_post"] == 1)
            ]["id"].unique()
        else:
            ids_to_keep = forum_data[
                (
                    forum_data["whole_text"].str.contains(
                        filter_by_expression, case=False
                    )
                )
                & (forum_data["is_original_post"] == 1)
            ]["id"].unique()

        forum_data = forum_data[forum_data["id"].isin(ids_to_keep)]

    # Breaking down text into sentences and striping white spaces
    sentences_data = create_sentence_df(forum_data)

    if only_keep_sentences_with_expression:
        # filtering again after breaking into sentences
        if filter_by_expression == "solar panel":
            sentences_data = sentences_data[
                (
                    sentences_data["sentences"].str.contains(
                        "|".join(
                            ["solar panel", "solar pv", "solar photovoltaic", " pv "]
                        ),
                        case=False,
                    )
                )
            ]
        else:
            sentences_data = sentences_data[
                (
                    sentences_data["sentences"].str.contains(
                        filter_by_expression, case=False
                    )
                )
            ]

    # Remove small sentences
    sentences_data = remove_small_sentences(sentences_data, min_n_tokens=5)

    # Removing sentences thanking people
    sentences_data = sentences_data[
        ~sentences_data["sentences"].str.contains(
            "|".join(phrases_to_remove), case=False
        )
    ]

    sentences_data.reset_index(drop=True, inplace=True)

    return sentences_data
