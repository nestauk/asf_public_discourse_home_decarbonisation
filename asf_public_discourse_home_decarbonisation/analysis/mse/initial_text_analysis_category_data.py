"""
This script performs an initial text analysis on Money Saving Expert (MSE) data, by:
- Computing the frenquency os specific words and n-grams of interest;
- Identiyfying top words and n-grams;

Plots are saved in the folder `asf_public_discourse_home_decarbonisation/outputs/figures/mse/`.

To run this script:
`python asf_public_discourse_home_decarbonisation/analysis/mse/initial_text_analysis_category_data.py`

To change category or date/time range, use the following arguments:
`python asf_public_discourse_home_decarbonisation/analysis/mse/initial_text_analysis_category_data.py --category <category> --collection_date_time <collection_date_time>`
where
<category>: category or sub-forum to be analysed. See below the full list of categories.
<collection_date_time>: data collection date/time in the format YYYY_MM_DD.

Full list of categories:
"green-ethical-moneysaving": Green and Ethical Money Saving sub-forum.
"lpg-heating-oil-solid-other-fuels": LPG, heating, oil, solid and other fuels sub-forum.
"energy": Energy sub-forum.
"is-this-quote-fair": Is this quote fair? sub-forum.
"all": All the categories above combined.
"sample": Sample of data from the Green and Ethical Money Saving sub-forum (not representative).
"""

# ## Package imports
import warnings

# Ignoring warnings from Pandas
warnings.simplefilter(action="ignore")

import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import os
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_text,
    create_ngram_from_ordered_tokens,
    english_stopwords_definition,
    lemmatize_sentence,
    frequency_ngrams,
)
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    plot_and_save_top_ngrams,
    plot_and_save_wordcloud,
)
from asf_public_discourse_home_decarbonisation import PROJECT_DIR

import argparse
import logging

logger = logging.getLogger(__name__)

# Analysis-specific parameters
number_ngrams_wordcloud = 30
min_frequency_tokens = 100
min_frequency_bigrams = 50
min_frequency_trigrams = 10
top_ngrams_barplot = 10

# Specific groups of keywords we want to look at
keyword_dictionary = {
    "heat_pump_keywords": [
        "heat pump",
        "heatpump",
        "ashp",
        "gshp",
        "wshp",
        "air 2 air",
        "air to air",
        "a2a",
    ],
    "boiler_keywords": ["boiler"],
    "hydrogen_keywords": ["hydrogen", "h2"],
    "bus_keywords": ["boiler upgrade scheme", "bus"],
    "grants_keywords": [
        "boiler upgrade scheme",
        "bus",
        "renewable heat incentive",
        "domestic rhi",
        "rhi",
        "dhri" "clean heat grant",
        "home energy scotland grant",
        "home energy scotland loan",
        "home energy scotland scheme",
    ],
    "mcs_keywords": ["mcs", "microgeneration certification scheme"],
    "cost_est_keywords": ["cost estimator"],
    "nesta_keywords": ["nesta"],
    "installer_keywords": ["installer", "engineer"],
    "installation_keywords": ["installation"],
    "cost_keywords": ["cost", "price", "pay", "pound", "£"],
    "issue_keywords": ["issue"],
    "noise_keywords": ["noise", "noisy", "decibel", "db"],
    "flow_temp_keywords": ["flow temperature", "flow temp"],
    "msbc_keywords": ["money saving boiler challenge", "msbc", "boiler challenge"],
}

# Parameters for analysis top keywords and n-grams
top_n_grams_analysis_parameters = {
    "tokens_title": {
        "data": lambda data: data[data["is_original_post"] == 1].drop_duplicates("id"),
        "ngrams_col": "tokens_title",
        "var_used": "titles",
        "min_frequency": min_frequency_tokens,
    },
    "bigrams_title": {
        "data": lambda data: data[data["is_original_post"] == 1].drop_duplicates("id"),
        "ngrams_col": "2_grams_title",
        "var_used": "titles",
        "min_frequency": min_frequency_bigrams,
    },
    "trigrams_title": {
        "data": lambda data: data[data["is_original_post"] == 1].drop_duplicates("id"),
        "ngrams_col": "3_grams_title",
        "var_used": "titles",
        "min_frequency": min_frequency_trigrams,
    },
    "tokens_text": {
        "data": lambda data: data,
        "ngrams_col": "tokens_text",
        "var_used": "posts and replies",
        "min_frequency": min_frequency_tokens,
    },
    "bigrams_text": {
        "data": lambda data: data,
        "ngrams_col": "2_grams_text",
        "var_used": "posts and replies",
        "min_frequency": min_frequency_bigrams,
    },
    "trigrams_text": {
        "data": lambda data: data,
        "ngrams_col": "3_grams_text",
        "var_used": "posts and replies",
        "min_frequency": min_frequency_trigrams,
    },
    "tokens_text_op": {
        "data": lambda data: data[data["is_original_post"] == 1],
        "ngrams_col": "tokens_text",
        "var_used": "posts",
        "min_frequency": min_frequency_tokens,
    },
    "bigrams_text_op": {
        "data": lambda data: data[data["is_original_post"] == 1],
        "ngrams_col": "2_grams_text",
        "var_used": "posts",
        "min_frequency": min_frequency_bigrams,
    },
    "trigrams_text_op": {
        "data": lambda data: data[data["is_original_post"] == 1],
        "ngrams_col": "3_grams_text",
        "var_used": "posts",
        "min_frequency": min_frequency_trigrams,
    },
}


def process_text_and_titles(mse_data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic processing to text data and identifying stopwords.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data

    Returns:
        pd.DataFrame: MSE data with processed text
    """
    mse_data["processed_text"] = mse_data["text"].apply(lambda x: process_text(x))
    mse_data["processed_title"] = mse_data["title"].apply(lambda x: process_text(x))
    return mse_data


def create_a_mapping_between_tokens_and_lemmas(mse_data: pd.DataFrame) -> dict:
    """
    Create a mapping between tokens and lemmas for each token in either text or titles
    Args:
        mse_data (pd.DataFrame): Money Saving Expert data

    Returns:
        dict: a dictionary mapping tokens to lemmatised tokens
    """
    # Create a list of tokens in titles (removing repetitions from replies, which have the same title)
    tokens_title = mse_data[mse_data["is_original_post"] == 1][["tokens_title"]]
    tokens_title = tokens_title.explode("tokens_title").rename(
        columns={"tokens_title": "tokens"}
    )

    # Create a list of tokens in text
    tokens_text = mse_data[["tokens_text"]]
    tokens_text = tokens_text.explode("tokens_text").rename(
        columns={"tokens_text": "tokens"}
    )

    # Concatenate both lists
    all_tokens = pd.concat([tokens_text, tokens_title])

    # Unique tokens
    all_tokens = all_tokens.drop_duplicates("tokens")

    # Removing digits as they cannot be lemmatized
    # (needs to explicitly to have `== False` because is_digit can also be missing)
    all_tokens["is_digit"] = all_tokens["tokens"].str.isdigit()
    all_tokens = all_tokens[all_tokens["is_digit"] == False]

    lemmas_dictionary = lemmatize_sentence(list(all_tokens["tokens"]))

    return lemmas_dictionary


def tokenize_and_lemmatise_data(mse_data: pd.DataFrame) -> pd.DataFrame:
    """
    Tokenizes and lemmatises text data.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data

    Returns:
        pd.DataFrame: Money Saving Expert data with tokenized and lemmatised added columns
    """

    # Tokenizing and lemmatising text and titles variables
    mse_data["tokens_title"] = mse_data["processed_title"].apply(word_tokenize)
    mse_data["tokens_text"] = mse_data["processed_text"].apply(word_tokenize)

    # Creating a list with all tokens (either in title or text of posts/replies)
    # and creating a dictonary that maps tokens to lemmatised tokens:
    lemmas_dictionary = create_a_mapping_between_tokens_and_lemmas(mse_data)

    # Lemmatising tokens - using np.vectorize seems to be the fastest way of lemmatizing
    # if lemma is not found, the original word is kept
    replace_words_vec = np.vectorize(
        lambda sentence: " ".join(
            lemmas_dictionary.get(word, word) for word in sentence.split()
        )
    )
    mse_data["processed_title"] = replace_words_vec(mse_data["processed_title"])
    mse_data["processed_text"] = replace_words_vec(mse_data["processed_text"])

    # Tokenizing again after lemmatising applied
    mse_data["tokens_title"] = mse_data["processed_title"].apply(word_tokenize)
    mse_data["tokens_text"] = mse_data["processed_text"].apply(word_tokenize)

    return mse_data


def remove_stopwords_from_text_and_titles(
    mse_data: pd.DataFrame, stopwords: list
) -> pd.DataFrame:
    """
    Removes stopwords from text and titles.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        stopwords (list): list of stopwords to be removed
    Returns:
        pd.DataFrame: MSE data with stopwords removed from text and titles
    """
    stopwords = english_stopwords_definition()
    mse_data["tokens_title"] = mse_data["tokens_title"].apply(
        lambda x: [token for token in x if token not in stopwords]
    )
    mse_data["tokens_text"] = mse_data["tokens_text"].apply(
        lambda x: [token for token in x if token not in stopwords]
    )
    return mse_data


def create_ngram_columns(mse_data: pd.DataFrame, max_n_grams: int = 3) -> pd.DataFrame:
    """
    Creates ngram columns for titles and text.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        max_n_grams (int, optional): maximum n-gram size to generate. Defaults to 3, wich will create columns for bigrams and trigrams.

    Returns:
        pd.DataFrame: Dataframe with added ngram columns for titles and text
    """
    for n in range(2, max_n_grams + 1):
        mse_data[f"{n}_grams_title"] = mse_data.apply(
            lambda x: create_ngram_from_ordered_tokens(x["tokens_text"], n=n), axis=1
        )
        mse_data[f"{n}_grams_text"] = mse_data.apply(
            lambda x: create_ngram_from_ordered_tokens(x["tokens_text"], n=n), axis=1
        )
    return mse_data


def prepare_data_for_text_analysis(
    mse_data: pd.DataFrame, stopwords: list, max_n_grams: int = 3
) -> pd.DataFrame:
    """
    Prepares data for text analysis by
    - processing text and titles columns;
    - tokenizing and lemmatising;
    - removing stopwords
    - creating ngram columns.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        stopwords (list): a list of English stopwords
        max_n_grams (int, optional): maximum n-gram size to generate. Defaults to 3, wich will create columns for bigrams and trigrams.

    Returns:
        pd.DataFrame: Money Saving Expert data with added columns for text analysis
    """

    logger.info("Preparing data for text analysis...")
    mse_data = process_text_and_titles(mse_data)

    mse_data = tokenize_and_lemmatise_data(mse_data)

    mse_data = remove_stopwords_from_text_and_titles(mse_data, stopwords)

    mse_data = create_ngram_columns(mse_data, max_n_grams)

    return mse_data


def number_instances_containing_keywords(data: pd.DataFrame, keyword_list: list) -> int:
    """
    Computes the number of instances in the data that contain at least one of the keywords in the keyword list.

    Args:
        data (pd.DataFrame): Dataframe containing the text data
        keyword_list (list): A list of keywords to search for
    Returns:
        int: Number of instances containing at least one of the keywords in the keyword list
    """
    return data[
        data["title"].str.contains("|".join(keyword_list), case=False)
        | data["text"].str.contains("|".join(keyword_list), case=False)
    ].shape[0]


def create_and_save_table_with_keyword_counts(
    mse_data: pd.DataFrame,
    keyword_dictionary: dict,
    save_path: str,
    filter: str = "all",
):
    """
    Creates a table with the counts of the keywords in the keyword dictionary.

    Args:
        mse_data (pd.DataFrame): MSE data
        keyword_dictionary (dict): Dictionary containing the keywords
        save_path (str): Path to save the table
        filter (str): filter for "posts", "replies" or "all" where "all" contains "posts" and "replies"
    """
    logger.info("Creating table with keyword-specific counts...")

    if filter == "posts":
        filtered_data = mse_data[mse_data["is_original_post"] == 1][["title", "text"]]
    elif filter == "replies":
        filtered_data = mse_data[mse_data["is_original_post"] == 0][["title", "text"]]
        filtered_data["title"] = ""  # the title is the title of the post, not the reply
    elif filter == "all":
        filtered_data = mse_data[["title", "text", "is_original_post"]].copy()
        filtered_data["title"] = filtered_data.apply(
            lambda x: "" if x["is_original_post"] == 0 else x["title"], axis=1
        )
    else:
        raise ValueError(
            f"{filter} is not a valid filter! Choose between 'posts', 'replies' or 'all'"
        )

    keyword_counts = pd.DataFrame()
    keyword_counts["group"] = keyword_dictionary.keys()
    keyword_counts["counts"] = keyword_counts["group"].apply(
        lambda x: number_instances_containing_keywords(
            filtered_data, keyword_dictionary[x]
        )
    )
    keyword_counts["percentage"] = keyword_counts["counts"] / len(filtered_data) * 100
    keyword_counts.sort_values("counts", ascending=False, inplace=True)
    logger.info("Keyword groups counts:\n{}".format(keyword_counts))


def plot_and_save_analysis_top_n_grams(analysis_parameters: dict):
    """
    Plotting and saving the barplots and wordcloud of top ngrams.

    Args:
        analysis_parameters (dict): A dictionary containing the parameters for the analysis
    """
    logger.info("Analysing top keywords and ngrams...")
    for t in analysis_parameters.keys():
        params = analysis_parameters[t]
        filter_to_apply = params["data"]
        filtered_data = filter_to_apply(mse_data)
        frequencies = frequency_ngrams(filtered_data, params["ngrams_col"])

        # Bar plot with top ngrams
        plot_and_save_top_ngrams(
            frequencies,
            top_ngrams_barplot,
            category,
            params["var_used"],
            MSE_FIGURES_PATH,
        )

        # Wordcloud with top ngrams above min_frequency
        plot_and_save_wordcloud(
            frequencies,
            number_ngrams_wordcloud,
            params["min_frequency"],
            category,
            params["var_used"],
            MSE_FIGURES_PATH,
            stopwords,
        )


def create_argparser() -> argparse.ArgumentParser:
    """
    Creates an argument parser that can receive the following arguments:
    - category: category or sub-forum (defaults to "green-ethical-moneysaving")
    - collection_date_time: collection date/time (defaults to "2023_11_15")

    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        help="Category or sub-forum",
        default="green-ethical-moneysaving",
        type=str,
    )

    parser.add_argument(
        "--collection_date_time",
        help="Collection date/time",
        default="2023_11_15",
        type=str,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_argparser()
    category = args.category
    collection_date_time = args.collection_date_time

    MSE_FIGURES_PATH = f"{PROJECT_DIR}/outputs/figures/mse/{category}"
    os.makedirs(MSE_FIGURES_PATH, exist_ok=True)

    mse_data = get_mse_data(category, collection_date_time)

    stopwords = english_stopwords_definition()

    mse_data = prepare_data_for_text_analysis(mse_data, stopwords, max_n_grams=3)

    create_and_save_table_with_keyword_counts(
        mse_data, keyword_dictionary, MSE_FIGURES_PATH
    )

    plot_and_save_analysis_top_n_grams(top_n_grams_analysis_parameters)

    logger.info("Analysis completed!")
