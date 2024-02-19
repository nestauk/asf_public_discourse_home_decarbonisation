"""
Utility functions for topic analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
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
