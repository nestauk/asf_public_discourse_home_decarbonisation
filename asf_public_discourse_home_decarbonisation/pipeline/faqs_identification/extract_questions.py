"""
This script is designed to extract questions and specific patterns (that could've been written as questions) from a given forum text dataset.
Key Functionalities:
    - Extract Questions: Identifies and extracts questions from sentences based on criteria such finishing with a question mark or starting with a question token (e.g. what, why). Additionally, we also have filtering criteria such as length of sentence and number of tokens.
    - Extract Specific Patterns: Targets and extracts specific patterns or phrases with phrases such as "don't know", tailored by the `extract_idk` function.
    - Configurable: Allows runtime configuration through command-line arguments for flexible usage in different scenarios.
To run this script:
    'python extract_questions.py --forum <forum> --category <category> --collection_date_time <collection_date_time> --post_type <post_type> --num_of_words <num_of_words>'
where:
    <forum>: The forum that we are looking: `bh` for Buildhub and `mse` for Money Saving Expert.
    <category>: The category within the forum (e.g. `energy`) or `all` for all data for 'mse' and 'combined_data' for all data for 'bh'.
    <collection_date_time>: The date time format, this differs between 'YYYY_MM_DD' for MSE and 'YYMMDD' for bh.
    <post_type>: The type of post to filter by ('original', 'reply', or 'all').
    <num_of_words>: The minimum number of words a question should have to be included.
Example Usage:
    For MSE:
        'python extract_questions.py --forum mse --category green-ethical-moneysaving --post_type original --collection_date_time 2023_11_15 --num_of_words 5'
    For buildhub:
        'python extract_questions.py --forum bh --category 120_ground_source_heat_pumps_gshp --post_type reply --collection_date_time 231120 --num_of_words 5'
    Note: In these run commands note the different format in date-time format for the different forums.
"""

import pandas as pd
from pandas import DataFrame
import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import argparse
from asf_public_discourse_home_decarbonisation.getters.bh_getters import (
    get_bh_category_data,
)
from asf_public_discourse_home_decarbonisation.getters.mse_getters import (
    get_mse_data,
)
import os
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
import logging

logger = logging.getLogger(__name__)
nltk.download("punkt")


def create_argparser() -> argparse.ArgumentParser:
    """
    Creates an argument parser that can receive the following arguments:
    - forum: forum that we are looking at (defaults to "bh" but can be "mse")
    - category: category or sub-forum (defaults to "119_air_source_heat_pumps_ashp")
    - collection_date_time: collection date/time (defaults to "231120")
    - num_of_words: minimum number of words a question should have to be included (defaults to 5)
    - post_type: type of post ('original', 'reply', or 'all') (defaults to "all")
    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forum",
        help="Forum that we are looking at",
        default="bh",
        type=str,
    )

    parser.add_argument(
        "--category",
        help="Category or sub-forum",
        default="119_air_source_heat_pumps_ashp",
        type=str,
    )

    parser.add_argument(
        "--collection_date_time",
        help="Collection date/time",
        default="231120",
        type=str,
    )
    parser.add_argument(
        "--num_of_words",
        help="Minimum number of words a question should have to be included",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--post_type",
        help="Type of post ('original', 'reply', or 'all')",
        default="all",
        choices=["original", "reply", "all"],
        type=str,
    )

    args = parser.parse_args()
    return args


def get_category_data(
    forum: str, category: str, collection_date_time: str
) -> DataFrame:
    """
    Retrieves category-specific data from a forum based on the provided forum name, category, and collection date/time.

    Args:
        forum (str): The name of the forum ("bh" or "mse").
        category (str): The category of data to retrieve.
        collection_date_time (str): The date and time of data collection.


    Returns:
        DataFrame: The category-specific data retrieved from the forum.

    Raises:
        SystemExit: If an invalid forum name is provided.

    Note:
        - The function calls the respective data retrieval function based on the forum name.
        - If an invalid forum name is provided, the function logs an information message and exits the program.
    """
    if forum == "bh":
        category_dataframe = get_bh_category_data(category, collection_date_time)
    elif forum == "mse":
        category_dataframe = get_mse_data(category, collection_date_time)
    else:
        logging.info("Please enter a valid forum name (bh or mse)")
        sys.exit(-1)
    return category_dataframe


def process_category_data(
    category_dataframe: DataFrame,
    inclusion_phrases: List[str],
    num_of_words: int,
    post_type: str,
) -> DataFrame:
    """
    Processes the category-specific data in the provided DataFrame by applying various functions.

    Args:
        category_dataframe (DataFrame): The category-specific data to be processed.
        inclusion_phrases (List[str]): A list of inclusion phrases to use when extracting the "don't know" phrases.
        num_of_words (int): minimum number of words a question should have to be included (defaults to 5)
        post_type (str): The type of post to filter by ('original', 'reply', or 'all') (defaults to "all")
    Returns:
        DataFrame: The processed category-specific data.
    """
    # Filter the data by post type
    category_dataframe = filter_data_by_post_type(category_dataframe, post_type)
    # Convert all text to strings to avoid type errors
    category_dataframe["text"] = category_dataframe["text"].astype(str)
    category_dataframe["sentences"] = category_dataframe["text"].apply(sent_tokenize)

    # Apply the function to extract questions
    category_dataframe["questions"] = category_dataframe["sentences"].apply(
        lambda sentences: extract_questions(sentences)
    )
    # Apply the function to filter out short questions
    category_dataframe["questions"] = category_dataframe["questions"].apply(
        lambda questions: filter_out_short_questions(questions, num_of_words)
    )
    # Apply the function to extract "don't know" phrases
    category_dataframe["idk_phrases"] = category_dataframe["sentences"].apply(
        lambda sentences: extract_idk(sentences, inclusion_phrases)
    )
    # Apply the function to extract X from the "don't know X" phrases
    category_dataframe["sentences_without_inclusion"] = category_dataframe[
        "idk_phrases"
    ].apply(lambda idk_phrases: extract_X_from_idk(idk_phrases, inclusion_phrases))
    return category_dataframe


def filter_data_by_post_type(dataframe: DataFrame, post_type: str) -> DataFrame:
    """
    Filters the forum dataframe based on the specified post type.

    Args:
        dataframe (DataFrame): The forum dataframe to filter.
        post_type (str): The post type to include ("original", "reply", or "all").

    Returns:
        The filtered dataframe (DataFrame)
    """
    if post_type == "original":
        filtered_dataframe = dataframe.loc[dataframe["is_original_post"] == 1].copy()
    elif post_type == "reply":
        filtered_dataframe = dataframe.loc[dataframe["is_original_post"] == 0].copy()
    else:
        filtered_dataframe = dataframe
    return filtered_dataframe


def extract_questions(sentences: List[str]) -> List[str]:
    """
    Extracts and returns a curated list of questions from a list of sentences.

    The function utilises a combination of punctuation analysis (identifying sentences ending with '?').

    Args:
        sentences (List[str]): A list of sentences from which questions are to be extracted.

    Returns:
        List[str]: A list of strings, each a question identified from the input sentences. Questions are filtered to exclude those that are URLs.

    """
    # Regular expression to match sentences that either end with a "?"
    question_pattern = r"[^.?!]*\?"
    all_questions = []  # Initialise an empty list to hold all extracted questions
    # Iterate over each sentence in the input list
    for sentence in sentences:
        # Skip sentences that likely represent URLs
        if "/" in sentence and sentence.count("/") > 2:
            continue
        # Find all matches in the sentence
        potential_questions = re.findall(
            question_pattern, sentence, flags=re.IGNORECASE
        )
        all_questions.extend(potential_questions)
    return all_questions


def filter_out_short_questions(
    questions: List[str], num_of_words: int = 5
) -> List[str]:
    """
    Filters out short questions from a list of questions based on the number of words in the question.

    Args:
        questions (List[str]): The list of questions to filter.
        num_of_words (int): The minimum number of words a question should have to be included.

    Returns:
        List[str]: The filtered list of questions.
    """
    filtered_questions = []
    # Skip questions that are 'num_of_words' words or less and contain a question word
    for question in questions:
        words = nltk.word_tokenize(question)
        if len(words) <= num_of_words:
            continue
        # Add sentence to all_questions if it's not just question marks with optional spaces.
        if not re.fullmatch(r"\?+\s*", question):
            filtered_questions.append(question)
    return filtered_questions


def extract_idk(sentences: List[str], inclusion_phrases: List[str]) -> List[str]:
    """
    Extracts and returns a list of sentences mentioning at least one of the expressions in inclusion_phrases.

    The function searches for sentences that contain any of a predefined set of phrases indicating uncertainty or lack of knowledge (e.g., "don’t know", "do not know", "do not know how to"). It then puts these into a list.

    This can be particularly useful for analysing text where it's important to both identify expressions of uncertainty and examine the context without these expressions.

    Args:
        sentences (List[str]): A list of sentences to be evaluated. Each item in the list should be a string representing a single sentence.
        inclusion_phrases (List[str]): A list of inclusion phrases. Each item in the list should be a string representing a phrase to search for in the sentences.

    Returns:
        idk_phrases (List[str]):
            - This list contains sentences that include at least one of the expressions in inclusion_phrases.

    Note:
        - The function is case-insensitive when searching for the predefined phrases.
    """
    idk_phrases = []
    for sentence in sentences:
        if any(phrase in sentence.lower() for phrase in inclusion_phrases):
            idk_phrases.append(sentence)
    return idk_phrases


def extract_X_from_idk(
    idk_phrases: List[str], inclusion_phrases: List[str]
) -> List[str]:
    """
    Extracts and returns a list of expressions coming after the inclusion_phrases in sentences.
    Example:
         - idk_phrases: ["honestly I don't know what a heat pump is", "i have asked multiple people, but still don't know how what to do to install a heat pump in my home"
         - inclusion_phrases: ["don't know how to", "don't know"]
         - Returns: ["what a heat pump is", "what to do to install a heat pump in my home"]

    The function takes a list of inclusion_phrases and sentences mentioning at least one of the expressions in inclusion_phrases. It extracts the text following the inclusion phrase and returns a list of these extracted sentences.
    Args:
        idk_phrases (List[str]): A list of sentences mentioning at least one of the inclusion_phrases. Each item in the list should be a string representing a single sentence.
        inclusion_phrases (List[str]): A list of inclusion phrases. Each item in the list should be a string representing an expression (e.g. "don't know")
    Returns:
        sentences_without_inclusion (List[str]):
            - This list contains the text following the inclusion phrase.

    Note:
        - The function is case-insensitive when searching for the inclusion phrases.
    """
    sentences_without_inclusion = []

    idk_phrases = [phrase.lower() for phrase in idk_phrases]
    inclusion_phrases = [phrase.lower() for phrase in inclusion_phrases]

    for sentence in idk_phrases:
        for expression in inclusion_phrases:
            if expression in sentence:
                start_index = sentence.index(expression) + len(expression)
                extracted_text = sentence[start_index:].strip()
                sentences_without_inclusion.append(extracted_text)
                break
    return sentences_without_inclusion


def logger_statistics(category_dataframe: DataFrame):
    """
    Calculates and logs basic statistics about the category-specific data:
          - total number of sentences;
          - total number of phrases mentioning "do not know" or "don't know";
          - total number of questions.
    Args:
        category_dataframe (DataFrame): The category-specific data.
    """
    total_number_of_idk_phrases = category_dataframe["idk_phrases"].apply(len).sum()
    total_number_of_sentences = category_dataframe["text"].apply(len).sum()
    total_questions = category_dataframe["questions"].apply(len).sum()

    # Print some basic statistics
    logging.info(f"Total number of idk phrases: {total_number_of_idk_phrases}")
    logging.info(f"Total number of questions: {total_questions}")
    logging.info(f"Total number of sentences: {total_number_of_sentences}")


if __name__ == "__main__":
    # Read the CSV file
    args = create_argparser()
    category = args.category
    collection_date_time = args.collection_date_time
    num_of_words = args.num_of_words
    post_type = args.post_type
    # uses different getter function depending on the forum
    category_dataframe = get_category_data(args.forum, category, collection_date_time)
    FORUM_FAQ_PATH = os.path.join(
        PROJECT_DIR, f"outputs/data/extracted_questions/{args.forum}/forum_{category}/"
    )
    # Ensure the output directory exists
    os.makedirs(FORUM_FAQ_PATH, exist_ok=True)
    # List of phrases to include
    inclusion_phrases = [
        "don’t know ",
        "do not know ",
    ]
    # Process the category-specific data
    category_dataframe = process_category_data(
        category_dataframe,
        inclusion_phrases,
        num_of_words,
        post_type,
    )
    # Log some basic statistics
    logger_statistics(category_dataframe)
    # Create a list of all questions
    all_questions_in_category = sum(category_dataframe["questions"], [])
    all_idk_phrases_in_category = sum(category_dataframe["idk_phrases"], [])
    all_idk_phrases_without_inclusion_in_category = sum(
        category_dataframe["sentences_without_inclusion"], []
    )
    # Create a new DataFrame with just the questions
    logger.info("CSV files are located in: " + FORUM_FAQ_PATH + "\n")
    questions_category_dataframe = pd.DataFrame(
        all_questions_in_category, columns=["Question"]
    )
    # Create a new DataFrame with just the questions
    # output the filtered questions to a csv file.
    questions_category_dataframe.to_csv(
        FORUM_FAQ_PATH + "extracted_questions_" + category + "_" + post_type + ".csv",
        index=False,
    )
    # output the "don't know" and "don't know X" phrases to a dataframe.
    idk_phrases_category_dataframe = pd.DataFrame(
        {
            "idk_phrases": all_idk_phrases_in_category,
            "sentences_without_inclusion": all_idk_phrases_without_inclusion_in_category,
        }
    )
    idk_phrases_category_dataframe.to_csv(
        FORUM_FAQ_PATH + "idk_phrases_" + category + "_" + post_type + ".csv",
        index=False,
    )
