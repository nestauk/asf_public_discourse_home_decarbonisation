"""
This script performs an exploratory data analysis on the data collected from BuildHub.
- It allows us to:
    - Collect the URLs from posts on a specific category or sub-forum (e.g. https://forum.buildhub.org.uk/forum/119-air-source-heat-pumps-ashp/)
    - Gives the number of posts in the sub-forum (e.g. 119-air-source-heat-pumps-ashp has 1,000 posts)
    - Distribution of posts over time
    - Distribution of number of posts vs number of users
    - Word cloud of frequently used words in posts
    - Generate bigram and trigram frequency distributions
    - Generate bigram and trigram word clouds
    - Distribution of post lengths
    - Frequency of selected keywords in posts using the 'heating_technologies_ruleset_twitter' dictionary
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
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import nltk
from nltk import FreqDist
import argparse
import os
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    set_plotting_styles,
    finding_path_to_font,
    NESTA_COLOURS,
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
from heating_technologies_ruleset import heating_technologies_ruleset_twitter

set_plotting_styles()

font_path_ttf = finding_path_to_font("Averta-Regular")


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
    post_count = buildhub_ashp_dataframe[
        buildhub_ashp_dataframe["post_type"] == "Original"
    ].shape[0]
    print("post count:")
    print(post_count)
    sample_size = buildhub_ashp_dataframe.shape[0]  # shape[0] gives the number of rows
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
        "October",
        "october",
        "6",
        "2013",
        "January",
        "january",
        "2016",
        "moderator",
        "thisreport",
        "pretty",
    ]
    filtered_tokens = preprocess_text(buildhub_ashp_dataframe, new_stopwords)
    freq_dist = FreqDist(filtered_tokens)
    plot_word_cloud(freq_dist, BUILDHUB_FIGURES_PATH)

    ######### 4.Generate bigram and trigram frequency distributions ########
    raw_bigram_freq_dist, bigram_freq_dist, bigram_threshold = process_ngrams(
        filtered_tokens, 2
    )
    raw_trigram_freq_dist, trigram_freq_dist, trigram_threshold = process_ngrams(
        filtered_tokens, 3
    )
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
    keyword_df, df_threshold = prepare_keyword_dataframe(
        keyword_counter, len(buildhub_ashp_dataframe)
    )
    plot_tag_vertical_bar_chart(
        keyword_df,
        BUILDHUB_FIGURES_PATH,
        "freq_of_selected_keywords_in_posts.png",
        df_threshold,
    )
    plot_tag_horizontal_bar_chart(
        keyword_df,
        BUILDHUB_FIGURES_PATH,
        "freq_of_selected_keywords_in_posts_barh.png",
        df_threshold,
    )