"""
This script is designed to extract frequently asked questions (FAQs) and specific patterns from a given forum text dataset.
Key Functionalities:
    - Extract Questions: Identifies and extracts questions from text based on criteria such as the number of words and minimum sentence length.
    - Extract Specific Patterns: Targets and extracts specific patterns or phrases, tailored by the `extract_idk` function.
    - Configurable: Allows runtime configuration through command-line arguments for flexible usage in different scenarios.
To run this script:
    'python extract_faqs.py --forum <forum> --category <category> --collection_date_time <collection_date_time>'
where:
    <forum>: The forum that we are looking at (e.g. bh or MSE).
    <category>: The category within the forum (e.g. energy).
    <collection_date_time>: The date time format, this differs between 'YYYY_MM_DD' for MSE and 'YYMMDD' for bh.
Example Usage:
    For MSE:
        'python extract_faqs.py --forum mse --category energy --collection_date_time 2023_11_15'
    For buildhub:
        'python extract_faqs.py --forum bh --category combined_data --collection_date_time 231120'
    Note: In these run commands note the different format in date-time format for the different forums.
"""
import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import argparse
from asf_public_discourse_home_decarbonisation.getters.bh_getters import (
    get_bh_category_data,
)
from asf_public_discourse_home_decarbonisation.getters.mse_getters import (
    get_mse_category_data,
)
import os
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
import logging

logger = logging.getLogger(__name__)


def create_argparser() -> argparse.ArgumentParser:
    """
    Creates an argument parser that can receive the following arguments:
    - category: category or sub-forum (defaults to "119_air_source_heat_pumps_ashp")
    - collection_date_time: collection date/time (defaults to "231120")
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

    args = parser.parse_args()
    return args


def extract_questions(
    sentences: List[str], num_of_words: int = 5, min_length: int = 5
) -> List[str]:
    """
    Extracts and returns a curated list of questions from a list of sentences, specifically filtering out certain types of questions. This function is designed to identify and include questions that are more substantial and exclude shorter, less informative questions such as "any thoughts?" or "why?".

    The function utilises a combination of punctuation analysis (identifying sentences ending with '?') and the presence of certain question words at the beginning. However, it also employs a set of criteria to exclude questions that are deemed too brief or lacking in content, even if they technically qualify as questions.

    Args:
        sentences (List[str]): A list of sentences from which questions are to be extracted.
        num_of_words (int, optional): The maximum number of words allowed for a question to be considered valid. Defaults to 5.
        min_length (int, optional): The minimum length of a question to be considered valid. Defaults to 5.

    Returns:
        List[str]: A list of strings, each a question identified from the input sentences. Questions are filtered to exclude those that are brief or deemed less informative.

    Note:
        - The function uses predefined question words to identify potential questions but excludes questions that are brief or deemed less informative, such as "why?" or "any thoughts?".
        - This exclusion is based on the specific requirements for the type of questions the function is intended to extract, focusing on more substantial and content-rich questions.
    """
    question_words = [
        "what",
        "how",
        "where",
        "when",
        "who",
        "why",
        "which",
        "really",
        "any",
        "what's",
        "how's",
        "really",
        "thoughts",
        "is",
    ]
    # Regular expression to match sentences that either end with a "?" or start with a specified question word
    question_pattern = (
        r"\b(?:" + "|".join(question_words) + r")\b[^.?!]*|[^.?!]*\?[ ]*(?=[A-Z]|$)"
    )
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
        # Filter and add valid questions to the all_questions list
        for question in potential_questions:
            # Skip questions that are four words or less and contain a question word
            words = question.split()
            if len(words) <= num_of_words and any(
                word.lower() in question_words for word in words
            ):
                continue
            # Add sentence to all_questions if it's at least min_length long and not just question marks with optional spaces.
            if len(question) >= min_length and not re.fullmatch(r"\?+\s*", question):
                all_questions.append(question)

    return all_questions


def extract_idk(sentences: List[str]) -> Tuple[List[str], List[str]]:
    """
    Extracts and returns two lists based on a list of input sentences: one containing sentences with specific "I don't know" phrases, and another with these phrases removed.

    The function searches for sentences that contain any of a predefined set of phrases indicating uncertainty or lack of knowledge (e.g., "I don’t know", "I do not know", "I do not know how to"). It then segregates these sentences into two lists: one preserving the original sentences, and another with the specific phrases removed.

    This can be particularly useful for analysing text where it's important to both identify expressions of uncertainty and examine the context without these expressions.

    Args:
        sentences (List[str]): A list of sentences to be evaluated. Each item in the list should be a string representing a single sentence.

    Returns:
        Tuple[List[str], List[str]]:
            - The first list contains sentences that include any of the predefined "I don't know" phrases.
            - The second list contains the same sentences but with the "I don't know" phrases removed, allowing for analysis of the remaining sentence content.

    Note:
        - The function is case-insensitive when searching for the predefined phrases.
        - The removal of phrases is done in a straightforward manner, which may not account for complex sentence structures or punctuation.
    """
    idk_phrases = []
    sentences_without_inclusion = []
    inclusion_phrases = [
        "i don’t know ",
        "i do not know ",
        "i do not know how to ",
    ]  # List of phrases to include
    for sentence in sentences:
        if any(phrase in sentence.lower() for phrase in inclusion_phrases):
            idk_phrases.append(sentence)
            for phrase in inclusion_phrases:
                sentence = sentence.lower().replace(phrase, "")
            sentences_without_inclusion.append(sentence)
    return idk_phrases, sentences_without_inclusion


if __name__ == "__main__":
    # Read the CSV file
    args = create_argparser()
    category = args.category
    collection_date_time = args.collection_date_time
    # uses different getter function depending on the forum
    if args.forum == "bh":
        category_dataframe = get_bh_category_data(category, collection_date_time)
    elif args.forum == "mse":
        category_dataframe = get_mse_category_data(category, collection_date_time)
    else:
        logger.info("Please enter a valid forum name (bh or mse)")
        sys.exit(-1)
    FORUM_FAQ_PATH = os.path.join(
        PROJECT_DIR, f"outputs/extracted_questions/{args.forum}/forum_{category}/"
    )
    # Ensure the output directory exists
    os.makedirs(FORUM_FAQ_PATH, exist_ok=True)
    # Tokenize sentences
    nltk.download("punkt")
    # Convert all text to strings to avoid type errors
    category_dataframe["text"] = category_dataframe["text"].astype(str)
    category_dataframe["sentences"] = category_dataframe["text"].apply(sent_tokenize)
    # Apply the function to extract questions
    category_dataframe["questions"] = category_dataframe["sentences"].apply(
        extract_questions
    )
    (
        category_dataframe["idk_phrases"],
        category_dataframe["sentences_without_inclusion"],
    ) = zip(*category_dataframe["sentences"].apply(extract_idk))
    total_number_of_idk_phrases = category_dataframe["idk_phrases"].apply(len).sum()
    total_number_of_sentences = category_dataframe["text"].apply(len).sum()
    total_questions = category_dataframe["questions"].apply(len).sum()
    # print some basic statistics!
    logger.info(f"Total number of idk phrases: {total_number_of_idk_phrases}")
    logger.info(f"Total number of questions: {total_questions}")
    logger.info(f"Total number of sentences: {total_number_of_sentences}")
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
    # Check for duplicate questions
    duplicate_questions = questions_category_dataframe[
        questions_category_dataframe.duplicated(subset="Question")
    ]
    if not duplicate_questions.empty:
        # output the duplicate questions
        duplicate_questions.to_csv(
            FORUM_FAQ_PATH + "duplicate_questions_" + category + ".csv", index=False
        )
    # output the filtered questions to a csv file.
    questions_category_dataframe.to_csv(
        FORUM_FAQ_PATH + "extracted_questions_" + category + ".csv", index=False
    )
    # output the "i don't know" phrases to a dataframe.
    idk_phrases_category_dataframe = pd.DataFrame(
        {
            "idk_phrases": all_idk_phrases_in_category,
            "sentences_without_inclusion": all_idk_phrases_without_inclusion_in_category,
        }
    )
    idk_phrases_category_dataframe.to_csv(
        FORUM_FAQ_PATH + "idk_phrases_" + category + ".csv", index=False
    )
