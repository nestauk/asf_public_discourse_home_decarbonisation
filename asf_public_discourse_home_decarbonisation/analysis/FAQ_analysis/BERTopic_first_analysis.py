"""
python BERTopic_first_analysis.py

This script clusters questions together to identify groups of similar questions. To do that we apply BERTopic topic model on a set of questions, with the integration of OpenAI's GPT-3 model for improved topic representation.

The process includes the following steps:
1. Load the 'extracted questions' data from a CSV file, extracting the 'Question' column.
2. Create an OpenAI client using the provided API key.
3. Create a BERTopic model with the OpenAI client as the representation model and fit it to the questions.
4. Visualise the topics identified by the model in various ways, including a general topic visualization, a bar chart of the top topics, and a hierarchy of the topics.
5. Plot the distribution of topics.
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
from umap import UMAP
import re
import openai
from bertopic.representation import OpenAI

# Sets the plotting styles and finds the path to the specified font for later use in plots.
set_plotting_styles()
font_path_ttf = finding_path_to_font("Averta-Regular")

# An instance of the OpenAI client using the API key retrieved from the environment variables. This client is used to interact with the OpenAI API.
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
"""
The below 'prompt' is a template for a prompt used in the BERTopic model for topic modeling. The prompt is designed to be used with the OpenAI GPT-3 model to generate a concise topic label based on provided documents and keywords.

The prompt contains placeholders for documents and keywords which are to be replaced with actual data during runtime.

Structure of the prompt:

- 'Documents': This is a placeholder where the actual documents related to a particular topic are to be inserted. These documents are used by the model to understand the context of the topic.

- 'Keywords': This is a placeholder where the actual keywords related to the topic are to be inserted. These keywords help the model in generating a more accurate and relevant topic label.

- 'Precise topic label': This is the instruction for the model to generate a concise topic label that is less than 10 words. The generated label is based on the provided documents and keywords.

Usage:

The prompt is used in the following way:

1. Replace '[DOCUMENTS]' and '[KEYWORDS]' with actual data.
2. Pass the updated prompt to the OpenAI GPT-3 model.
3. The model generates a concise topic label based on the provided documents and keywords.
"""
prompt = """Create a concise topic label using the provided documents and keywords. The label title should be less than 10 words:

Documents:
- [DOCUMENTS]

Keywords:
- [KEYWORDS]

Precise topic label:
"""

# Creates an instance of the OpenAI representation model with specified parameters for topic modeling.
representation_model = OpenAI(
    client,
    model="gpt-3.5-turbo",
    chat=True,
    prompt=prompt,
    nr_docs=30,
    delay_in_seconds=3,
)


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


def deduplicate_and_load_csv(
    file_path: str, output_path: str, column_name: str = "title_and_text_questions"
):
    """
    Deduplicates a CSV file based on a specified column ('Question' as default) and saves the result to a new CSV file. We want this to be done before we start the topic modelling to get diverse representative questions.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path where the deduplicated CSV file will be saved.
        column_name (str): The name of the column to check for duplicates. Default is 'Question'.

    Returns:
        List[str]: A list of deduplicated questions.

    Example usage:
        deduplicate_csv('path_to_your_input_file.csv', 'path_to_your_output_file.csv', 'Question')
    """
    # Step 1: Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Step 2: Remove duplicates based on the specified column
    deduplicated_df = df.drop_duplicates(subset=[column_name])
    # Step 3: Replace non-breaking space characters with regular spaces
    deduplicated_df[column_name] = deduplicated_df[column_name].str.replace("\xa0", " ")

    # Step 3: Convert the questions to lower case
    deduplicated_df[column_name] = deduplicated_df[column_name].str.lower()

    # Step 4: Replace acronyms with their full forms
    deduplicated_df[column_name] = deduplicated_df[column_name].apply(
        lambda x: x.replace("ashps", "air source heat pumps")
        .replace("ashp", "air source heat pump")
        .replace("gshps", "ground source heat pumps")
        .replace("gshp", "ground source heat pump")
        .replace("hps", "heat pumps")
        .replace("hp", "heat pump")
        .replace("ufh", "under floor heating")
        .replace("temps", "temperatures")
        .replace("rhi", "renewable heat incentive")
        .replace("mcs", "microgeneration certification scheme")
        .replace("dhw", "domestic hot water system")
        .replace("a2a", "air to air")
        .replace(" ir ", " infrared ")
        .replace("uvcs", "unvented cylinders")
        .replace("uvc", "unvented cylinder")
    )
    deduplicated_df[column_name] = deduplicated_df[column_name].apply(
        lambda x: re.sub(r"\btemp\b", "temperature", x)
    )
    # Add spaces before "air" where necessary
    deduplicated_df[column_name] = deduplicated_df[column_name].apply(
        lambda x: re.sub(r"(an|of|the)(air)", r"\1 \2", x)
    )

    # Step 4: Save the deduplicated DataFrame to a new CSV file
    deduplicated_df.to_csv(output_path, index=False)

    # Step 5: Return the deduplicated questions as a list
    return deduplicated_df[column_name].tolist()


def create_topic_model(questions: List[str]) -> Tuple[BERTopic, List[int], List[float]]:
    """
    Creates and fits a BERTopic model to the given list of questions.

    Args:
        questions (List[str]): A list of questions to model.

    Returns:
        tuple: Contains the fitted BERTopic model, topics, and their probabilities.
    """
    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
    )
    topic_model = BERTopic(
        umap_model=umap_model, representation_model=representation_model
    )
    topics, probabilities = topic_model.fit_transform(questions)
    return topic_model, topics, probabilities


def visualise_topics(topic_model: BERTopic, figure_file_path: str):
    """
    Generates and saves a visualisation of topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
        figure_file_path (str): The path where the generated figure will be saved.
    """
    fig = topic_model.visualize_topics()
    fig.write_image(figure_file_path + "topic_visualisation.png")


def visualise_barchart(topic_model: BERTopic, figure_file_path: str):
    """
    Generates and saves a barchart visualisation of the top n topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
        figure_file_path (str): The path where the generated figure will be saved.
    """
    fig_barchart = topic_model.visualize_barchart(top_n_topics=16, n_words=10)
    fig_barchart.write_image(figure_file_path + "topic_visualisation_barchart.png")


def visualise_hierarchy(topic_model: BERTopic, figure_file_path: str):
    """
    Generates and saves a hierarchical visualisation of topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
        figure_file_path (str): The path where the generated figure will be saved.
    """
    fig_hierarchy = topic_model.visualize_hierarchy()
    fig_hierarchy.write_image(figure_file_path + "topic_visualisation_hierarchy.png")


def plot_topic_distribution(topic_model: BERTopic, figure_file_path: str):
    """
    Plots and saves the distribution of the top topics identified by the BERTopic model.

    Args:
        topic_model (BERTopic): The BERTopic model after fitting to data.
        figure_file_path (str): The path where the generated figure will be saved.
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
        f"outputs/data/extracted_questions/{forum}/forum_{category}/extracted_title_and_text_questions_{category}_{post_type}.csv",
    )
    deduplicated_data_path = os.path.join(
        PROJECT_DIR,
        f"outputs/data/extracted_questions/{forum}/forum_{category}/deduplicated_title_and_text_questions_{category}_{post_type}.csv",
    )
    figure_path = os.path.join(
        PROJECT_DIR, f"outputs/figures/extracted_questions/{forum}/forum_{category}/"
    )
    doc_topics_info_path = os.path.join(
        PROJECT_DIR, f"outputs/outputs/BERTopic_csv_files/{forum}/forum_{category}/"
    )
    os.makedirs(figure_path, exist_ok=True)
    os.makedirs(doc_topics_info_path, exist_ok=True)
    questions = deduplicate_and_load_csv(input_data, deduplicated_data_path)
    topic_model, topics, probabilities = create_topic_model(questions)
    visualise_topics(topic_model, figure_path)
    visualise_barchart(topic_model, figure_path)
    visualise_hierarchy(topic_model, figure_path)
    plot_topic_distribution(topic_model, figure_path)
    save_topic_info(topic_model, questions, doc_topics_info_path)


if __name__ == "__main__":
    main()
