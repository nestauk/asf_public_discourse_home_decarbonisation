"""
Script for identifying topics of conversation in forum sentences.

The pipeline:
- Gets forum data (including original posts and replies)
- Cleans and enhances the forum data
- Breaks the forum text data into sentences
- Applies topic analysis to unique sentences to identify topics of conversation
- Outputs are saved to S3 including information about topics the sentences data

To run this script:
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source SOURCE --start_date START_DATE --end_date END_DATE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --filter_by_expression FILTER_BY_EXPRESSION --min_topic_size MIN_TOPIC_SIZE

where:
- SOURCE is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] FILTER_BY_EXPRESSION is the expression to filter by. Defaults to 'heat pump'
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 100.

Examples for MSE:
2018-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_date "2018-01-01" --end_date "2024-05-22"
2016-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_date "2016-01-01" --end_date "2024-05-23"

Examples for Buildhub:
2018-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "buildhub" --start_date "2018-01-01" --end_date "2024-05-22"
2016-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "buildhub" --start_date "2016-01-01" --end_date "2024-05-23"
"""

# Package imports
import argparse
import pandas as pd
import string
import re
from datetime import datetime
from bertopic import BERTopic
from umap import UMAP
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import logging
from bertopic.representation import OpenAI
from asf_public_discourse_home_decarbonisation import config

logger = logging.getLogger(__name__)

# Local imports
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
)
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    remove_urls,
    remove_username_pattern,
    replace_username_mentions,
    remove_introduction_patterns,
    process_abbreviations,
    ends_with_punctuation,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        help="`mse` or `buildhub`",
        required=True,
    )
    parser.add_argument(
        "--reduce_outliers_to_zero",
        help="True to reduce outliers to zero. Defaults to False",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--filter_by_expression",
        help="Expression to filter by. Defaults to 'heat pump'",
        default="heat pump",
    )
    parser.add_argument(
        "--start_date",
        help="Analysis start date in the format YYYY-MM-DD. Default to None (all data)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--end_date",
        help="Analysis end date in the format YYYY-MM-DD. Defaults to None (all data)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--min_topic_size",
        help="Minimum topic size. Defaults to 100",
        default=100,
        type=int,
    )
    return parser.parse_args()


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
        phrases_to_remove (list): list of phrases to remove from the sentences. Defaults to ["thank", "happy to help", "kind wishes", "kind regards"]
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
        ids_to_keep = forum_data[
            (forum_data["whole_text"].str.contains(filter_by_expression, case=False))
            & (forum_data["is_original_post"] == 1)
        ]["id"].unique()

        forum_data = forum_data[forum_data["id"].isin(ids_to_keep)]

    # Breaking down text into sentences and striping white spaces
    sentences_data = create_sentence_df(forum_data)

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


def update_topics_with_duplicates(
    topics_info: pd.DataFrame, doc_info: pd.DataFrame, sentences_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Updates the topics information with the number of duplicates.

    Args:
        topics_info (pd.DataFrame): dataframe with the topics information
        doc_info (pd.DataFrame): dataframe with the document information
        sentences_data (pd.DataFrame): dataframe with the sentences data

    Returns:
        pd.DataFrame: updated topics information
    """
    updated_topics_info = sentences_data[
        ["sentences", "datetime", "date", "year", "id"]
    ]

    updated_topics_info = updated_topics_info.merge(
        doc_info, left_on="sentences", right_on="Document"
    )

    updated_topics_info = (
        updated_topics_info.groupby("Topic", as_index=False)[["id"]]
        .count()
        .rename(columns={"id": "updated_count"})
        .merge(topics_info, on="Topic")
    )

    updated_topics_info["updated_%"] = (
        updated_topics_info["updated_count"]
        / sum(updated_topics_info["updated_count"])
        * 100
    )

    updated_topics_info = updated_topics_info.sort_values(
        "updated_count", ascending=False
    ).reset_index(drop=True)

    return updated_topics_info


def update_docs_with_duplicates(
    doc_info: pd.DataFrame, sentences_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Updates the document information with the number of duplicates.

    Args:
        doc_info (pd.DataFrame): dataframe with the document information
        sentences_data (pd.DataFrame): dataframe with the sentences data

    Returns:
        pd.DataFrame: dataframe with the updated document information
    """

    sentence_counts = (
        sentences_data.groupby("sentences", as_index=False)[["id"]]
        .count()
        .rename(columns={"id": "count"})
    )

    updated_doc_info = doc_info.merge(
        sentence_counts, left_on="Document", right_on="sentences"
    )

    return updated_doc_info


def topic_model_definition(min_topic_size: int, representation_model: OpenAI = None):
    """
    Defines the topic model according to a set of parameters.

    Args:
        min_topic_size (int): minimum topic size
        representation_model (OpenAI): representation model
    Returns:
        BERTopic: BERTopic model
    """
    vectorizer_model = CountVectorizer(stop_words="english")
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    if representation_model is not None:
        topic_model = BERTopic(
            umap_model=umap_model,
            min_topic_size=min_topic_size,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            representation_model=representation_model,
        )
    else:
        topic_model = BERTopic(
            umap_model=umap_model,
            min_topic_size=min_topic_size,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
        )

    return topic_model


if __name__ == "__main__":
    # Dealing with user defined arguments
    args = parse_arguments()
    source = args.source
    reduce_outliers_to_zero = args.reduce_outliers_to_zero
    filter_by_expression = args.filter_by_expression
    start_date = args.start_date
    end_date = args.end_date
    min_topic_size = args.min_topic_size

    # Reading data
    if source == "mse":
        forum_data = get_mse_data(
            category="all",
            collection_date=config["latest_data_collection_date"]["mse"],
            processing_level="raw",
        )
    elif source == "buildhub":
        forum_data = get_bh_data(
            category="all",
            collection_date=config["latest_data_collection_date"]["buildhub"],
        )
        forum_data.rename(columns={"url": "id", "date": "datetime"}, inplace=True)
    else:
        raise ValueError("Invalid source")

    # Creating dataset of sentences and preparing inputs for topic analysis
    sentences_data = prepping_data_for_topic_analysis(
        forum_data,
        filter_by_expression,
        start_date,
        end_date,
        phrases_to_remove=["thank", "happy to help", "kind wishes", "kind regards"],
    )

    docs = list(sentences_data.drop_duplicates("sentences")["sentences"])
    dates = list(sentences_data.drop_duplicates("sentences")["date"])

    # Topic analysis
    topic_model = topic_model_definition(min_topic_size)
    topics, probs = topic_model.fit_transform(docs)
    topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

    # Logging relevant information
    logger.info(f"Number of topics: {len(topics_info) - 1}")
    logger.info(
        f"% of outliers: {topics_info[topics_info['Topic'] == -1]['%'].values[0]}"
    )

    # Reducing outliers to zero where relevant
    if reduce_outliers_to_zero:
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
        topic_model.update_topics(docs, topics=new_topics)
        topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)
        topics_info.sort_values("Count", ascending=False)

    # Updating topics and docs information with duplicates (as only unique sentences are used for topic analysis)
    topics_info = update_topics_with_duplicates(topics_info, doc_info, sentences_data)
    doc_info = update_docs_with_duplicates(doc_info, sentences_data)

    path_to_save_prefix = f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_{start_date}_{end_date}"
    # Saving outputs to S3
    topics_info.to_csv(
        f"{path_to_save_prefix}_sentence_topics_info.csv",
        index=False,
    )
    doc_info.to_csv(
        f"{path_to_save_prefix}_sentence_docs_info.csv",
        index=False,
    )
    sentences_data.to_csv(
        f"{path_to_save_prefix}_sentences_data.csv",
        index=False,
    )
