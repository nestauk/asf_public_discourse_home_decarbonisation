"""
This script performs an exploratory data analysis on the data collected from BuildHub.
- It allows us to:
    - Computes the number of posts in the sub-forum (e.g. 119-air-source-heat-pumps-ashp has 1,000 posts)
    - Plot the distribution of posts over time
    - Plot the distribution of number of posts vs number of users
    - Generate a Word cloud of frequently used words in posts
    - Generate bigram and trigram frequency distributions
    - Generate bigram and trigram word clouds
    - Distribution of post lengths
    - Print the frequency of selected keywords in posts using the 'heating_technologies_ruleset_twitter' dictionary
To run this script:
    'python asf_public_discourse_home_decarbonisation/analysis/ASF_awayday_buildhub_eda.py'
To change category or date/time range, use the following arguments:
    `python asf_public_discourse_home_decarbonisation/analysis/ASF_awayday_buildhub_eda.py --category <category> --collection_date_time <collection_date_time>`
where:
    <category>: category or sub-forum to be analysed, (e.g. "120_ground_source_heat_pumps_gshp" from a list of possible categories: ["119_air_source_heat_pumps_ashp","120_ground_source_heat_pumps_gshp","125_general_alternative_energy_issues","136_underfloor_heating","137_central_heating_radiators","139_boilers_hot_water_tanks","140_other_heating_systems"])
    <collection_date_time>: data collection date/time in the format YYMMDD (e.g. 231120)
The figures are saved in the following directory: 'outputs/figures/buildhub/forum_<category>/'
"""
# import packages
import pandas as pd
from typing import List
from nltk import FreqDist
import argparse
from nltk.util import ngrams
import os
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
import logging

logger = logging.getLogger(__name__)
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    finding_path_to_font,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation.config.heating_technologies_ruleset import (
    heating_technologies_ruleset_twitter,
)

from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    plot_post_distribution_over_time,
    plot_users_post_distribution,
    plot_word_cloud,
    plot_top_ngrams,
    plot_post_length_distribution,
    plot_tag_vertical_bar_chart,
    plot_tag_horizontal_bar_chart,
)
from asf_public_discourse_home_decarbonisation.utils.preprocessing_utils import (
    preprocess_text,
    process_ngrams,
    wordcloud_preprocess_ngrams,
    update_keyword_frequencies,
    prepare_keyword_dataframe,
)
from asf_public_discourse_home_decarbonisation.getters.bh_getters import (
    get_bh_category_data,
)


set_plotting_styles()

font_path_ttf = finding_path_to_font("Averta-Regular")


def calculate_ngram_threshold(tokens: List[str], n: int, freq_multiplier: float) -> int:
    """
    Calculates and returns the frequency threshold for n-grams.

    Args:
        tokens (List[str]): A list of tokens from which n-grams are generated.
        n (int): The 'n' in n-grams, representing the number of elements in each gram.
        freq_multiplier (float): The multiplier to calculate the frequency threshold.

    Returns:
        int: The calculated threshold for n-grams.
    """
    # Calculate initial frequency distribution for n-grams
    raw_ngram_freq_dist = FreqDist(ngrams(tokens, n))

    # Calculate total count and threshold for n-grams
    total_ngrams = sum(raw_ngram_freq_dist.values())
    ngram_threshold = round(max(3, total_ngrams * freq_multiplier))

    return raw_ngram_freq_dist, ngram_threshold


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


# This is where script is executed.
if __name__ == "__main__":
    args = create_argparser()
    category = args.category
    collection_date_time = args.collection_date_time
    BUILDHUB_FIGURES_PATH = os.path.join(
        PROJECT_DIR, f"outputs/figures/buildhub/forum_{category}/"
    )
    # Ensure the output directory exists
    os.makedirs(BUILDHUB_FIGURES_PATH, exist_ok=True)
    buildhub_ashp_dataframe = get_bh_category_data(category, collection_date_time)
    buildhub_ashp_dataframe["date"] = pd.to_datetime(buildhub_ashp_dataframe["date"])
    n_posts = buildhub_ashp_dataframe[
        buildhub_ashp_dataframe["post_type"] == "Original"
    ].shape[0]
    logger.info(f"Number of posts: {n_posts}")
    n_posts_replies = buildhub_ashp_dataframe.shape[
        0
    ]  # shape[0] gives the number of rows
    logger.info(f"Number of posts + replies: {n_posts_replies}")
    ######### 1. Distribution of posts over time #########
    plot_post_distribution_over_time(
        buildhub_ashp_dataframe, BUILDHUB_FIGURES_PATH, NESTA_COLOURS[0]
    )
    ########### 2. Distribution of number of posts vs number of users ###########
    plot_users_post_distribution(buildhub_ashp_dataframe, BUILDHUB_FIGURES_PATH)
    # ######### 3. Word cloud of frequently used words in posts ############
    new_stopwords = [
        "would",
        "hours",
        "hour",
        "minute",
        "minutes",
        "ago",
        "dan",
        "(presumably",
        "looks",
        "like",
        "need",
        "ap50",
        ".3page",
        "fraser",
        "lamont",
        "got",
        "bit",
        "sure",
        "steamytea",
        "could",
        "get",
        "still",
        "october",
        "6",
        "2013",
        "january",
        "2016",
        "moderator",
        "thisreport",
        "pretty",
    ]
    filtered_tokens = preprocess_text(buildhub_ashp_dataframe, new_stopwords)
    freq_dist = FreqDist(filtered_tokens)
    # Plotting the top 10 words in a cloud...
    plot_word_cloud(freq_dist, BUILDHUB_FIGURES_PATH)

    ######### 4.Generate bigram and trigram frequency distributions ########

    raw_bigram_freq_dist, bigram_threshold = calculate_ngram_threshold(
        filtered_tokens, 2, 0.0002
    )
    raw_trigram_freq_dist, trigram_threshold = calculate_ngram_threshold(
        filtered_tokens, 3, 0.00005
    )

    bigram_freq_dist = process_ngrams(raw_bigram_freq_dist, bigram_threshold)
    trigram_freq_dist = process_ngrams(raw_trigram_freq_dist, trigram_threshold)
    plot_top_ngrams(
        raw_bigram_freq_dist, 10, "Bigram", NESTA_COLOURS[0], BUILDHUB_FIGURES_PATH
    )
    plot_top_ngrams(
        raw_trigram_freq_dist, 10, "Trigram", NESTA_COLOURS[1], BUILDHUB_FIGURES_PATH
    )

    ######### 5.Generate bigram and trigram word clouds ########
    bigram_string_freq_dist, trigram_string_freq_dist = wordcloud_preprocess_ngrams(
        [bigram_freq_dist, trigram_freq_dist]
    )
    plot_word_cloud(
        bigram_string_freq_dist, BUILDHUB_FIGURES_PATH, "Bigram", bigram_threshold
    )
    plot_word_cloud(
        trigram_string_freq_dist, BUILDHUB_FIGURES_PATH, "Trigram", trigram_threshold
    )

    # ############ 6. Distribution of post lengths #############
    plot_post_length_distribution(buildhub_ashp_dataframe, BUILDHUB_FIGURES_PATH)

    # ############ 7. Frequency of selected keywords in posts using the dictionary ###########
    keyword_counter = update_keyword_frequencies(
        buildhub_ashp_dataframe, "text", heating_technologies_ruleset_twitter
    )
    total_rows_df = len(buildhub_ashp_dataframe)
    tag_threshold = round(max(5, total_rows_df * 0.0001))
    keyword_df = prepare_keyword_dataframe(keyword_counter, tag_threshold)
    logger.info(f"keywords: {keyword_df}")
    plot_tag_vertical_bar_chart(
        keyword_df,
        BUILDHUB_FIGURES_PATH,
        "freq_of_selected_keywords_in_posts.png",
        tag_threshold,
    )
    plot_tag_horizontal_bar_chart(
        keyword_df,
        BUILDHUB_FIGURES_PATH,
        "freq_of_selected_keywords_in_posts_barh.png",
        tag_threshold,
    )
