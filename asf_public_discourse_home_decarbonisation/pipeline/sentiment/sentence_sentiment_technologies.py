"""
Apply a sentiment model to sentences and save scores.
Collect and save sentiment for sentences by running:

MSE:
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source "mse" --filter_by_expression "heat pump" --relevant_clusters "1,2,3,4,5,7,8,9,10,11,18,19,20,21,22,25,31,32,40,51,52,54"
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source "mse" --filter_by_expression "heat pump" --irrelevant_clusters "0,6"

Buildhub:
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source "buildhub" --filter_by_expression "heat pump" --relevant_clusters "1,2,3,5,6,9,10,11,12,16,27,29,31,32,56,64"

For other uses, the class can be used with:

from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment import SentenceBasedSentiment
sentiment_model = SentenceBasedSentiment(process_data=False)
texts = ["This is a really great sentence", "This sentence is awful", "Cat"]
sentiment_scores = sentiment_model.get_sentence_sentiment(texts)
>> [("This is a really great sentence", 'positive', 0.97741115), ("This sentence is awful", 'negative', 0.9255473), ("Cat", 'neutral', 0.6470574)]
"""

# Package imports
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import os
from argparse import ArgumentParser
from typing import Union
import pandas as pd
import re
from datetime import datetime
from nltk.tokenize import sent_tokenize, word_tokenize
import string

# Local imports
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    save_to_s3,
    load_s3_data,
)
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment import (
    SentenceBasedSentiment,
    list_chunks,
)
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    remove_urls,
    remove_username_pattern,
    remove_introduction_patterns,
    process_abbreviations,
    ends_with_punctuation,
    replace_username_mentions,
)


def cleaning_and_enhancing_forum_data(forum_data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Args:
        forum_data (pd.DataFrame): forum data
    Returns:
        pd.DataFrame: enhanced forum data
    """
    forum_data["text"] = forum_data["text"].astype(str)
    forum_data["title"] = forum_data["title"].astype(str)

    forum_data["text"] = forum_data["text"].apply(remove_urls)
    forum_data["text"] = forum_data["text"].apply(remove_username_pattern)
    forum_data["text"] = forum_data["text"].apply(replace_username_mentions)
    forum_data["text"] = forum_data["text"].apply(remove_introduction_patterns)

    forum_data["title"] = forum_data.apply(
        lambda x: "" if x["is_original_post"] == 0 else x["title"], axis=1
    )
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

    forum_data["datetime"] = pd.to_datetime(forum_data["datetime"])
    forum_data["date"] = forum_data["datetime"].dt.date
    forum_data["year"] = forum_data["datetime"].dt.year

    # Adding space after punctuation ("?", ".", "!"), so that sentences are correctly split
    forum_data["whole_text"] = forum_data["whole_text"].apply(
        lambda t: re.sub(r"([.!?])([A-Za-z])", r"\1 \2", t)
    )

    return forum_data


def create_sentence_df(forum_data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Args:
        forum_data (pd.DataFrame): _description_
    Returns:
        pd.DataFrame: _description_
    """
    forum_data["sentences"] = forum_data["whole_text"].apply(sent_tokenize)

    sentences_data = forum_data.explode("sentences")
    sentences_data["sentences"] = sentences_data["sentences"].astype(str)
    sentences_data["sentences"] = sentences_data["sentences"].str.strip()

    return sentences_data


def remove_small_sentences(sentences_data: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Args:
        sentences_data (pd.DataFrame): _description_
    Returns:
        pd.DataFrame: _description_
    """
    sentences_data["tokens"] = sentences_data["sentences"].apply(word_tokenize)
    sentences_data["non_punctuation_tokens"] = sentences_data["tokens"].apply(
        lambda x: [token for token in x if token not in string.punctuation]
    )

    sentences_data["n_tokens"] = sentences_data["non_punctuation_tokens"].apply(len)

    sentences_data = sentences_data[sentences_data["n_tokens"] > 5]

    return sentences_data


def prepping_data_for_topic_analysis(
    forum_data: pd.DataFrame, filter_by_expression: str, start_date: int, end_date: int
) -> pd.DataFrame:
    """_summary_
    Args:
        forum_data (pd.DataFrame): _description_
        filter_by_expression (str): _description_
        start_year (int): _description_
        end_year (int): _description_
    Returns:
        pd.DataFrame: _description_
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

    # ransforming text into sentences and striping white sapces
    sentences_data = create_sentence_df(forum_data)

    if filter_by_expression == "solar panel":
        sentences_data = sentences_data[
            (
                sentences_data["sentences"].str.contains(
                    "|".join(["solar panel", "solar pv", "solar photovoltaic", " pv "]),
                    case=False,
                )
            )
        ]
    else:
        sentences_data = sentences_data[
            (sentences_data["sentences"].str.contains(filter_by_expression, case=False))
        ]
    # Remove small sentences
    sentences_data = remove_small_sentences(sentences_data)

    # Removing sentences thanking people
    sentences_data = sentences_data[
        ~(
            sentences_data["sentences"].str.contains("thank", case=False)
            | sentences_data["sentences"].str.contains("happy to help", case=False)
            | sentences_data["sentences"].str.contains("kind wishes", case=False)
            | sentences_data["sentences"].str.contains("kind regards", case=False)
        )
    ]

    sentences_data.reset_index(drop=True, inplace=True)

    return sentences_data


if __name__ == "__main__":

    chunk_size = 100

    mse_data = get_mse_data(category="all", collection_date="2024_06_03")

    hp_sentences_data = prepping_data_for_topic_analysis(
        mse_data, "heat pump", "2018-01-01", "2024-05-22"
    )

    solar_sentences_data = prepping_data_for_topic_analysis(
        mse_data, "solar panel", "2018-01-01", "2024-05-22"
    )

    boilers_sentences_data = prepping_data_for_topic_analysis(
        mse_data, "boiler", "2018-01-01", "2024-05-22"
    )

    hp_sentences_data = list(
        hp_sentences_data.drop_duplicates("sentences")["sentences"]
    )
    solar_sentences_data = list(
        solar_sentences_data.drop_duplicates("sentences")["sentences"]
    )
    boilers_sentences_data = list(
        boilers_sentences_data.drop_duplicates("sentences")["sentences"]
    )

    sentiment_model = SentenceBasedSentiment()

    # BOILERS

    all_sentiment = []
    for text in tqdm(list_chunks(boilers_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    output_name = f"data/mse/outputs/sentiment/desnz/mse_boiler_sentences_sentiment.csv"

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )

    # SOLAR PANELS
    all_sentiment = []
    for text in tqdm(list_chunks(solar_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    output_name = (
        f"data/mse/outputs/sentiment/desnz/mse_solar_panel_sentences_sentiment.csv"
    )

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )

    # HPS
    all_sentiment = []
    for text in tqdm(list_chunks(hp_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    output_name = (
        f"data/mse/outputs/sentiment/desnz/mse_heat_pump_sentences_sentiment.csv"
    )

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )
