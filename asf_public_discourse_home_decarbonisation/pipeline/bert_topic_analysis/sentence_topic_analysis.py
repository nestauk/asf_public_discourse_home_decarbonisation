"""
Identifying topics of conversation from sentences.

python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_year 2018

"""

# Package imports
import argparse
import pandas as pd
import string
import re
from bertopic import BERTopic
from umap import UMAP
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import logging
import openai

logger = logging.getLogger(__name__)

# Local imports
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
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
        "--start_year",
        help="Analysis start year. Default to None (all data)",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--end_year",
        help="Analysis end year. Defaults to None (all data)",
        default=None,
        type=int,
    )
    return parser.parse_args()


def cleaning_and_enhancing_forum_data(forum_data: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        forum_data (pd.DataFrame): forum data

    Returns:
        pd.DataFrame: enhanced forum data
    """
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
    forum_data: pd.DataFrame, filter_by_expression: str, start_year: int, end_year: int
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

    if start_year is not None:
        forum_data = forum_data[forum_data["year"] >= start_year]
    if end_year is not None:
        forum_data = forum_data[forum_data["year"] <= end_year]

    # Focusing on conversations mentioning a certain expression e.g. "heat pump"
    if filter_by_expression is not None:
        ids_to_keep = forum_data[
            (forum_data["whole_text"].str.contains(filter_by_expression, case=False))
            & (forum_data["is_original_post"] == 1)
        ]["id"].unique()

        forum_data = forum_data[forum_data["id"].isin(ids_to_keep)]

    # ransforming text into sentences and striping white sapces
    sentences_data = create_sentence_df(forum_data)

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


def update_topics_with_duplicates(
    topics_info: pd.DataFrame, doc_info: pd.DataFrame, sentences_data: pd.DataFrame
):
    """_summary_

    Args:
        topics_info (pd.DataFrame): _description_
        doc_info (pd.DataFrame): _description_
        sentences_data (pd.DataFrame): _description_

    Returns:
        _type_: _description_
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


def topic_model_definition(
    min_topic_size: int, reduce_outliers_to_zero: bool, representation_model: openai
):
    """_summary_

    Args:
        min_topic_size (int): _description_
        reduce_outliers_to_zero (bool): _description_
        representation_model (openai): _description_

    Returns:
        _type_: _description_
    """
    vectorizer_model = CountVectorizer(stop_words="english")
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    if reduce_outliers_to_zero:
        topic_model = BERTopic(
            umap_model=umap_model,
            min_topic_size=min_topic_size,
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
        )
    elif representation_model is not None:
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
        )
    return topic_model


if __name__ == "__main__":
    args = parse_arguments()
    source = args.source
    reduce_outliers_to_zero = args.reduce_outliers_to_zero
    filter_by_expression = args.filter_by_expression

    if source == "mse":
        forum_data = get_mse_data(
            category="all", collection_date="2023_11_15", processing_level="raw"
        )

    sentences_data = prepping_data_for_topic_analysis(
        forum_data, filter_by_expression, args.start_year, args.end_year
    )

    docs = sentences_data.drop_duplicates("sentences")["sentences"]
    dates = list(sentences_data.drop_duplicates("sentences")["date"])

    if source == "mse":
        min_topic_size = 100
    else:
        min_topic_size = 50

    topic_model = topic_model_definition(min_topic_size, reduce_outliers_to_zero)
    topics, probs = topic_model.fit_transform(docs)
    topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

    logger.info(f"Number of topics: {len(topics_info) - 1}")
    logger.info(
        f"% of outliers: {topics_info[topics_info['Topic'] == -1]['%'].values[0]}"
    )

    if reduce_outliers_to_zero:
        new_topics = topic_model.reduce_outliers(
            docs, topics, probabilities=probs, strategy="probabilities"
        )
        topic_model.update_topics(docs, topics=new_topics)
        topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)
        topics_info.sort_values("Count", ascending=False)

    topics_info = update_topics_with_duplicates(topics_info, doc_info, sentences_data)

    topics_info.to_csv(
        f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_sentence_topics_info.csv",
        index=False,
    )
    doc_info.to_csv(
        f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_sentence_docs_info.csv",
        index=False,
    )
