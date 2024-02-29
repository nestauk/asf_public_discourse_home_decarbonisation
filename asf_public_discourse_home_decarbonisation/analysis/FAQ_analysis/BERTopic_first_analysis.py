"""
python BERTopic_first_analysis.py

This script clusters questions together to identify groups of similar questions. To do that we apply BERTopic topic model on a set of questions.

The process includes the following steps:
1. Load the 'extracted questions' data from a CSV file, extracting the 'Question' column.
2. Create a BERTopic model and fit it to the questions.
3. Visualise the topics identified by the model in various ways, including a general topic visualization, a bar chart of the top topics, and a hierarchy of the topics.
4. Plot the distribution of topics.
"""

import pandas as pd
from bertopic import BERTopic
import matplotlib.pyplot as plt
import argparse
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    finding_path_to_font,
)
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
from typing import List, Tuple
import os

set_plotting_styles()
font_path_ttf = finding_path_to_font("Averta-Regular")


def create_argparser() -> argparse.ArgumentParser:
    """
    Creates an argument parser that can receive the following arguments:
    - category: category or sub-forum (defaults to "119_air_source_heat_pumps_ashp")
    - forum: forum (i.e. mse or bh) (defaults to "bh")
    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        help="Category or sub-forum",
        default="119_air_source_heat_pumps_ashp",
        type=str,
    )

    parser.add_argument(
        "--forum",
        help="forum (i.e. mse or bh)",
        default="bh",
        type=str,
    )
    parser.add_argument(
        "--post_type",
        help="post type (i.e. all, original or replies)",
        default="all",
        type=str,
    )

    args = parser.parse_args()
    return args


def load_data(file_path: str) -> List[str]:
    """
    Loads extracted questions from a CSV file into a list.

    Args:
        file_path (str): The path to the CSV file containing the questions.

    Returns:
        List[str]: A list of questions.
    """
    return pd.read_csv(file_path)["Question"].tolist()


def create_topic_model(questions: List[str]) -> Tuple[BERTopic, List[int], List[float]]:
    """
    Creates and fits a BERTopic model to the given list of questions.

    Args:
        questions (List[str]): A list of questions to model.

    Returns:
        tuple: Contains the fitted BERTopic model, topics, and their probabilities.
    """
    topic_model = BERTopic()
    topics, probabilities = topic_model.fit_transform(questions)
    return topic_model, topics, probabilities


def visualise_topics(topic_model: BERTopic, figure_file_path: str) -> None:
    """
    Generates and saves a visualisation of topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
    """
    fig = topic_model.visualize_topics()
    fig.write_image(figure_file_path + "topic_visualisation.png")


def visualise_barchart(topic_model: BERTopic, figure_file_path: str) -> None:
    """
    Generates and saves a barchart visualisation of the top n topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
    """
    fig_barchart = topic_model.visualize_barchart(top_n_topics=16, n_words=10)
    fig_barchart.write_image(figure_file_path + "topic_visualisation_barchart.png")


def visualise_hierarchy(topic_model: BERTopic, figure_file_path: str) -> None:
    """
    Generates and saves a hierarchical visualisation of topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
    """
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_image(figure_file_path + "topic_visualisation_hierarchy.png")


def plot_topic_distribution(topic_model: BERTopic, figure_file_path: str) -> None:
    """
    Plots and saves the distribution of the top topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
    """
    topic_counts = topic_model.get_topic_info()["Count"][1:17]
    topic_labels = topic_model.get_topic_info()["Name"][1:17].str.replace("_", " ")
    plt.figure(figsize=(14, 8))
    plt.barh(topic_labels, topic_counts, color=NESTA_COLOURS[0])
    plt.ylabel("Topics")
    plt.xlabel("Count")
    plt.title("Topic Distribution")
    plt.tight_layout()
    plt.savefig(
        figure_file_path + "topic_distribution.png", dpi=300, bbox_inches="tight"
    )


def save_topic_info(topic_model, questions, doc_topics_info_path):
    """
    doc_info and topics_info will tell you
    - which cluster a specific question belongs to;
    - if a question is representative of the cluster or not;

    Parameters:
    topic_model (BERTopic): The BERTopic model.
    questions (list): The list of questions.
    doc_topics_info_path (str): The path to save the CSV files.
    """
    # Get document info and save to CSV
    doc_info = topic_model.get_document_info(questions)
    doc_info.to_csv(
        os.path.join(doc_topics_info_path, "document_info.csv"),
        index=False,
        encoding="utf-8",
    )

    # Get topic info, calculate percentage, and save to CSV
    topics_info = topic_model.get_topic_info()
    topics_info["%"] = topics_info["Count"] / len(questions) * 100
    topics_info.to_csv(
        os.path.join(doc_topics_info_path, "topics_info.csv"),
        index=False,
        encoding="utf-8",
    )


def main():
    """
    Main function to execute the topic modeling workflow.

    This function sets up the plotting styles, loads the data, creates a topic model from the questions,
    and generates visualisations for topics, barchart, hierarchy, and topic distribution.
    """
    set_plotting_styles()
    args = create_argparser()
    category = args.category
    forum = args.forum
    post_type = args.post_type
    input_data = os.path.join(
        PROJECT_DIR,
        f"outputs/data/extracted_questions/{forum}/forum_{category}/extracted_questions_{category}_{post_type}.csv",
    )
    figure_path = os.path.join(
        PROJECT_DIR, f"outputs/figures/extracted_questions/{forum}/forum_{category}/"
    )
    doc_topics_info_path = os.path.join(
        PROJECT_DIR, f"outputs/outputs/BERTopic_csv_files/{forum}/forum_{category}/"
    )
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(doc_info_path, exist_ok=True)
    questions = load_data(input_data)
    topic_model, topics, probabilities = create_topic_model(questions)
    visualise_topics(topic_model, figure_path)
    visualise_barchart(topic_model, figure_path)
    visualise_hierarchy(topic_model, figure_path)
    plot_topic_distribution(topic_model, figure_path)
    save_topic_info(topic_model, questions, doc_topics_info_path)


if __name__ == "__main__":
    main()
