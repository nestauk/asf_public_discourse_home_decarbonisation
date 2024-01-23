"""
This script performs an initial text analysis on Money Saving Expert (MSE) data, by:
- Computing the frenquency os specific words and n-grams of interest;
- Identiyfying top words and n-grams;

Plots are saved in the folder `asf_public_discourse_home_decarbonisation/outputs/figures/mse/`.

To run this script:
`python asf_public_discourse_home_decarbonisation/analysis/mse/text_analysis_case_study.py`

To change category or date/time range, use the following arguments:
`python asf_public_discourse_home_decarbonisation/analysis/mse/text_analysis_case_study.py --category <category> --collection_date_time <collection_date_time>`
where
<category>: category or sub-forum to be analysed. See below the full list of categories.
<collection_date_time>: data collection date/time in the format YYYY_MM_DD.
<tag>:
"""

# ## Package imports
import warnings

# Ignoring warnings from Pandas
warnings.simplefilter(action="ignore")

import os
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation import PROJECT_DIR

import argparse
import logging
from initial_text_analysis_category_data import (
    keyword_dictionary,
    top_n_grams_analysis_parameters,
)
from initial_text_analysis_category_data import (
    create_ngram_columns,
    create_table_with_keyword_counts,
    plot_and_save_analysis_top_n_grams,
)
from asf_public_discourse_home_decarbonisation.pipeline.data_processing_flows.text_processing_utils import (
    english_stopwords_definition,
)

logger = logging.getLogger(__name__)


def create_argparser() -> argparse.ArgumentParser:
    """
    Creates an argument parser that can receive the following arguments:

    Returns:
        argparse.ArgumentParser: argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_date_time",
        help="Collection date/time",
        default="2023_11_15",
        type=str,
    )
    parser.add_argument(
        "--keyword_group",
        help="Keyword group",
        default="heat_pump_keywords",
        type=str,
    )
    parser.add_argument(
        "--category",
        help="Category or sub-forum",
        default="all",
        type=str,
    )

    args = parser.parse_args()
    return args


def filter_data(mse_data, tags):
    keyword_list = keyword_dictionary[tags]
    filtered_data = mse_data[
        mse_data["title"].str.contains("|".join(keyword_list), case=False)
        | mse_data["text"].str.contains("|".join(keyword_list), case=False)
    ]

    filtered_data.reset_index(inplace=True)

    return filtered_data


if __name__ == "__main__":
    args = create_argparser()
    collection_date_time = args.collection_date_time
    keyword_group = args.keyword_group
    category = args.category

    MSE_FIGURES_PATH = (
        f'{PROJECT_DIR}/outputs/figures/mse/case_studies/{"_".join(keyword_group)}'
    )
    os.makedirs(MSE_FIGURES_PATH, exist_ok=True)

    if "," in category:
        mse_data = get_mse_data(
            "all", collection_date_time, processing_level="processed"
        )
        categories = category.split(",")
        mse_data = mse_data[mse_data["category"].isin(categories)]
        mse_data.reset_index(inplace=True)
    else:
        mse_data = get_mse_data(
            category, collection_date_time, processing_level="processed"
        )

    logger.info(f"Number of instances: {len(mse_data)}")

    filtered_data = filter_data(mse_data, keyword_group)

    logger.info(f"Number of instances in filtered data: {len(filtered_data)}")

    prop = len(filtered_data) / len(mse_data) * 100

    logger.info(f"Percentage: {prop}")
    del mse_data

    filtered_data = create_ngram_columns(
        filtered_data, lemmatised_tokens=True, remove_stopwords=True, max_n_grams=4
    )

    create_table_with_keyword_counts(
        filtered_data,
        keyword_dictionary,
    )
    create_table_with_keyword_counts(filtered_data, keyword_dictionary, "posts")

    plot_and_save_analysis_top_n_grams(
        filtered_data,
        top_n_grams_analysis_parameters,
        stopwords=english_stopwords_definition(),
        category=f"`all filtered for `{keyword_group}`",
        figures_path=MSE_FIGURES_PATH,
    )

    logger.info("Analysis completed!")
