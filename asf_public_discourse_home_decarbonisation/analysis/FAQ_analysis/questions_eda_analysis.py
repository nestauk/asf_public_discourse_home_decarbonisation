"""
This script performs exploratory data analysis (EDA) on a set of questions and can be run with following command:
    'python questions_eda_analysis.py'

The process includes the following steps:
1. Load the data from a CSV file, extracting the 'Question' column.
2. Count the frequency of each question.
3. Wrap the text of each question and add an ellipsis if it exceeds a certain length.
4. Plot the counts of the most frequent questions.

The script is designed to be modular, with separate functions for each step of the process. This makes the code easier to read and maintain, and allows for parts of the code to be reused elsewhere if needed.
"""

from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    finding_path_to_font,
)
import os
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from matplotlib.ticker import MaxNLocator
import argparse
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation import PROJECT_DIR

set_plotting_styles()
font_path_ttf = finding_path_to_font("Averta-Regular")


def create_argparser() -> argparse.ArgumentParser:
    """
    Creates an argument parser that can receive the following arguments:
    - category: category or sub-forum (defaults to "119_air_source_heat_pumps_ashp")
    - forum: forum (i.e. mse or bh) (defaults to "bh")
    - post_type: post type (i.e. all, original or replies) (defaults to "all")
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


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from the extracted questions CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    return pd.read_csv(file_path)


def get_question_counts(df: pd.DataFrame, column_name: str = "Question") -> pd.Series:
    """
    Calculates the frequency of each unique question in a specified column.

    Args:
        df (pd.DataFrame): The DataFrame containing the questions.
        column_name (str): The name of the column from which to count questions.

    Returns:
        pd.Series: A Series containing question counts, indexed by the question text.
    """
    return df[column_name].value_counts()


def wrap_text_with_ellipsis(text: str, line_width: int, max_lines: int) -> str:
    """
    Wraps text to a specified line width and maximum number of lines, adding an ellipsis if truncated.

    Args:
        text (str): The text to wrap.
        line_width (int): The maximum width of each line.
        max_lines (int): The maximum number of lines.

    Returns:
        str: The wrapped (and possibly truncated) text with an ellipsis if necessary.
    """
    wrapped_text = textwrap.wrap(text, width=line_width)
    if len(wrapped_text) > max_lines:
        return "\n".join(wrapped_text[:max_lines]) + "..."
    else:
        return "\n".join(wrapped_text)


def plot_question_counts(
    question_counts: pd.Series, top_n: int = 5, figure_path: str = None
) -> None:
    """
    Plots a horizontal bar chart of the most frequent questions.
    Args:
        question_counts (pd.Series): A Series containing question counts, indexed by question text.
        top_n (int): The number of top questions to display.
    """
    top_questions = question_counts.head(top_n)
    wrapped_questions = [wrap_text_with_ellipsis(q, 40, 2) for q in top_questions.index]

    plt.figure(figsize=(12, 8))
    plt.barh(wrapped_questions, top_questions.values, color=NESTA_COLOURS[0])
    plt.gcf().subplots_adjust(left=0.5)
    plt.xlabel("Frequency")
    plt.ylabel("Questions")
    plt.title(f"Top {top_n} Most Frequent Questions")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(
        figure_path + "most_frequent_questions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


"""
def plot_question_counts(
    question_counts: pd.Series, min_frequency: int = 2, figure_path: str = None
) -> None:

    Plots a horizontal bar chart of the questions with a frequency above a certain value.

    Args:
        question_counts (pd.Series): A Series containing question counts, indexed by question text.
        min_frequency (int): The minimum frequency to display.

    top_questions = question_counts[question_counts >= min_frequency]
    wrapped_questions = [wrap_text_with_ellipsis(q, 40, 2) for q in top_questions.index]

    plt.figure(figsize=(12, 8))
    plt.barh(wrapped_questions, top_questions.values, color=NESTA_COLOURS[0])
    plt.gcf().subplots_adjust(left=0.5)
    plt.xlabel("Frequency")
    plt.ylabel("Questions")
    plt.title(f"Questions with Frequency Above {min_frequency}")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(
        figure_path + "most_frequent_questions.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
"""


def main():
    """
    Main function to perform data loading, question counting, and plotting the most frequent questions.
    """
    args = create_argparser()
    category = args.category
    forum = args.forum
    post_type = args.post_type

    input_data = os.path.join(
        PROJECT_DIR,
        f"outputs/data/extracted_questions/{forum}/forum_{category}/extracted_questions_{category}_{post_type}.csv",
    )
    output_figures_path = os.path.join(
        PROJECT_DIR, f"outputs/figures/extracted_questions/{forum}/forum_{category}/"
    )
    os.makedirs(output_figures_path, exist_ok=True)
    extracted_questions_df = load_data(input_data)
    question_counts = get_question_counts(extracted_questions_df)
    plot_question_counts(question_counts, figure_path=output_figures_path)
    dont_knows_path = os.path.join(
        PROJECT_DIR,
        f"outputs/data/extracted_questions/{forum}/forum_{category}/idk_phrases_{category}_{post_type}.csv",
    )
    dont_knows_df = load_data(dont_knows_path)
    dont_knows_df["sentences_without_inclusion"] = dont_knows_df[
        "sentences_without_inclusion"
    ].str.lower()
    dk_counts = get_question_counts(
        dont_knows_df, column_name="sentences_without_inclusion"
    )
    dk_counts = dk_counts[dk_counts > 1]
    if len(dk_counts) > 0:
        print(dk_counts)
    else:
        print('No frequent "do not know" expressions found')


if __name__ == "__main__":
    main()
