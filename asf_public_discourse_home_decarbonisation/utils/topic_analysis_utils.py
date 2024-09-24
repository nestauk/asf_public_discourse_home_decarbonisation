"""
Utility functions for topic analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import OpenAI

from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    NESTA_COLOURS,
    set_plotting_styles,
)

set_plotting_styles()


def create_bar_plot_most_common_topics(
    topics_info: pd.DataFrame,
    title: str = "Most common topics",
    top_n_topics: int = 16,
    variable="titles",
):
    """
    Plots the most common topics in a horizontal bar chart.

    Args:
        topics_info (pd.DataFrame): Dataframe with the topics' name and percentage of docs.
        title (str, optional): plot title. Defaults to "Most common topics".
        top_n_topics (int, optional): number of topics to show. Defaults to 16.
        variable (str, optional): variable used for the topic analysis e.g. titles, text, questions, etc. Defaults to "titles".
    """
    plt.figure(figsize=(8, 6))
    # Remove the first topic as it is the "outliers" topic
    topics_to_plot = topics_info[1:top_n_topics]
    plt.barh(topics_to_plot["Name"], topics_to_plot["%"], color=NESTA_COLOURS[0])
    plt.title(title)
    plt.yticks(size=11)
    plt.xticks(size=14)
    plt.xlabel(f"% of {variable}")


def get_outputs_from_topic_model(topic_model, docs: list) -> pd.DataFrame:
    """
    Get the topics, topics_info and doc_info from a topic model.

    Args:
        topic_model (object): topic model object
        docs (list): documents used for topic analysis

    Returns:
        tuple: topics, topics_info, doc_info
    """
    topics = topic_model.get_topics()

    topics_info = topic_model.get_topic_info()
    topics_info["%"] = topics_info["Count"] / len(docs) * 100

    doc_info = topic_model.get_document_info(docs)
    return topics, topics_info, doc_info


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
