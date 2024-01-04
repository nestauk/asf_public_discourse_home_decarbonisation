"""
This script performs an initial text analysis on Money Saving Expert (MSE) data, by:
- Computing the frenquency os specific words and expression of interest;
- Identiyfying top words and n-grams;

Plots are saved in the folder `asf_public_discourse_home_decarbonisation/outputs/figures/mse/`.

To run this script:
`python asf_public_discourse_home_decarbonisation/analysis/mse/initial_text_analysis_category_data.py`
which will run this analysis on the "green-ethical-moneysaving" sub-forum/category.

To change category or date/time range, use the following arguments:
`python asf_public_discourse_home_decarbonisation/analysis/mse/initial_text_analysis_category_data.py --category <category> --collection_date_time <collection_date_time>`
where
<category>: category or sub-forum to be analysed. See below the full list of categories below.
<collection_date_time>: data collection date/time in the format YYYY_MM_DD.

Full list of categories:
"green-ethical-moneysaving": Green and Ethical Money Saving sub-forum.
"lpg-heating-oil-solid-other-fuels": LPG, heating, oil, solid and other fuels sub-forum.
"energy": Energy sub-forum.
"is-this-quote-fair": Is this quote fair? sub-forum.
"all": All the categories above combined.
"sample": Sample of data from the Green and Ethical Money Saving sub-forum (not a representative sample).
"""

# ## Package imports
import warnings

# Ignoring warnings from Pandas
warnings.simplefilter(action="ignore")

import pandas as pd
import os
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.utils.ngram_utils import (
    create_ngram_from_ordered_tokens,
    frequency_ngrams,
)
from asf_public_discourse_home_decarbonisation.pipeline.data_processing_flows.flow_utils import (
    english_stopwords_definition,
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

# Number of top words/n-grams to show
top_ngrams_wordcloud = 30
top_ngrams_barplot = 15

# Minimum frequencies accepted
min_frequency_tokens = 100
min_frequency_bigrams = 50
min_frequency_trigrams = 10
min_frequency_4_grams = 10

# Specific groups of keywords/expressions we want to look at
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
        "dhri",
        "clean heat grant",
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
        "ngrams_col": "lemmatised_tokens_title_no_stopwords",
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
        "ngrams_col": "lemmatised_tokens_text_no_stopwords",
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
    "4_grams_text": {
        "data": lambda data: data,
        "ngrams_col": "4_grams_text",
        "var_used": "posts and replies",
        "min_frequency": min_frequency_4_grams,
    },
    "tokens_text_op": {
        "data": lambda data: data[data["is_original_post"] == 1],
        "ngrams_col": "lemmatised_tokens_text_no_stopwords",
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
    "4_grams_text_op": {
        "data": lambda data: data[data["is_original_post"] == 1],
        "ngrams_col": "4_grams_text",
        "var_used": "posts",
        "min_frequency": min_frequency_4_grams,
    },
}


def create_ngram_columns(
    mse_data: pd.DataFrame,
    lemmatised_tokens: bool = True,
    remove_stopwords: bool = True,
    max_n_grams: int = 3,
) -> pd.DataFrame:
    """
    Creates ngram columns for titles and text.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        lemmatised_tokens (bool): True to use lemmatised tokens
        remove_stopwords (bool): True to use tokens without stopwords
        max_n_grams (int, optional): maximum n-gram size to generate. Defaults to 3, wich will create columns for bigrams and trigrams.

    Returns:
        pd.DataFrame: Dataframe with added ngram columns for titles and text
    """
    prefix = ""
    if lemmatised_tokens:
        prefix = "lemmatised_"
    suffix = ""
    if remove_stopwords:
        suffix = "_no_stopwords"

    for n in range(2, max_n_grams + 1):
        mse_data[f"{n}_grams_title"] = mse_data.apply(
            lambda x: create_ngram_from_ordered_tokens(
                x[f"{prefix}tokens_title{suffix}"], n=n
            ),
            axis=1,
        )
        mse_data[f"{n}_grams_text"] = mse_data.apply(
            lambda x: create_ngram_from_ordered_tokens(
                x[f"{prefix}tokens_text{suffix}"], n=n
            ),
            axis=1,
        )
    return mse_data


def number_instances_containing_keywords(data: pd.DataFrame, keyword_list: list) -> int:
    """
    Computes the number of instances in the data that contain at least one of the keywords in the keyword list.
    Note that we're not performing keyword match.

    Args:
        data (pd.DataFrame): Dataframe containing the text data
        keyword_list (list): A list of keywords/expression to search for
    Returns:
        int: Number of instances containing at least one of the keywords in the keyword list
    """
    return data[
        data["title"].str.contains("|".join(keyword_list), case=False)
        | data["text"].str.contains("|".join(keyword_list), case=False)
    ].shape[0]


def create_table_with_keyword_counts(
    mse_data: pd.DataFrame,
    keyword_dictionary: dict,
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
    logger.info(f"Creating table with keyword-specific counts...")

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
    if filter == "all":
        filter = "all data"
    logger.info(f"Keyword groups counts filtered for `{filter}`:\n{keyword_counts}")


def plot_and_save_analysis_top_n_grams(analysis_parameters: dict, stopwords: list):
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
            top_ngrams_wordcloud,
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

    mse_data = get_mse_data(
        category, collection_date_time, processing_level="processed"
    )

    mse_data = create_ngram_columns(
        mse_data, lemmatised_tokens=True, remove_stopwords=True, max_n_grams=4
    )

    create_table_with_keyword_counts(
        mse_data,
        keyword_dictionary,
    )
    create_table_with_keyword_counts(mse_data, keyword_dictionary, "posts")

    plot_and_save_analysis_top_n_grams(
        top_n_grams_analysis_parameters, stopwords=english_stopwords_definition()
    )

    logger.info("Analysis completed!")
