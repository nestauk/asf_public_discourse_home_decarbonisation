"""
Functions for easier generation of plots.
"""

from matplotlib import font_manager
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    FONT_NAME,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    identify_n_gram_type,
)
import os
import logging

logger = logging.getLogger(__name__)


def finding_path_to_font(font_name: str) -> str:
    """
    Finds path to specific font.
    Args:
        font_name (str): name of font
    Returns:
        str: local path to font with font name
    """

    all_font_files = font_manager.findSystemFonts()
    font_files = [f for f in all_font_files if font_name in f]
    if len(font_files) == 0:
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f]
    return font_files[0]


def create_wordcloud(frequencies: dict, max_words: int, stopwords: list):
    """
    Creates word cloud based on frequencies.
    Args:
        frequencies (dict): a dictionary with frequencies of words or n-grams
        max_words (int): maximum number of words or n-grams to be displayed
        stopwords (list): stopwords that should be removed from the wordcloud
    """
    font_path_ttf = finding_path_to_font(FONT_NAME)
    plt.figure()
    wordcloud = WordCloud(
        font_path=font_path_ttf,
        width=2000,
        height=1000,
        margin=0,
        collocations=True,
        stopwords=stopwords,
        background_color="white",
        max_words=max_words,
    )

    wordcloud = wordcloud.generate_from_frequencies(frequencies=frequencies)

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")


def plot_and_save_top_ngrams(
    n_gram_data: dict, top_n: int, category: str, var_used: str, fig_path: str
):
    """
    Creates and saves a barplot of the top `top_n` ngrams.

    Args:
        n_gram_data (dict): Dictionary containing the ngrams and their frequency
        top_n (int): Number of ngrams to display
        category (str): sub-forum category
        var_used (str): variable in which the ngrams are based e.g. "titles", "posts", "posts and replies"
        fig_path (str): path to figures folder
    """
    most_common = dict(n_gram_data.most_common(top_n))
    n_gram_type = identify_n_gram_type(most_common)
    plt.figure(figsize=(10, 6))
    plt.barh(
        list(most_common.keys()), list(most_common.values()), color=NESTA_COLOURS[0]
    )
    plt.xlabel("Frequency")
    plt.title(f"Top {top_n} {n_gram_type} in {var_used} for category\n`{category}`")
    plt.tight_layout()
    path_to_plot = os.path.join(
        fig_path, f"category_{category}_top_{top_n}_{n_gram_type}_{var_used}.png"
    )
    plt.savefig(path_to_plot)
    plt.clf()


def plot_and_save_wordcloud(
    n_gram_data: dict,
    top_n: int,
    min_frequency_allowed: int,
    category: str,
    var_used: str,
    fig_path: str,
    stopwords: list,
):
    """
    Creates and saves a wordcloud of the top `top_n` ngrams with a frenquency above `min_frequency_allowed`.

    Args:
        n_gram_data (dict): Dictionary containing the ngrams and their frequency
        top_n (int):  Number of ngrams to display
        min_frequency_allowed (int): Mininum frequency of ngrams to be displayed
        category (str): sub-forum category
        var_used (str): variable in which the ngrams are based e.g. "titles", "posts", "posts and replies"
        fig_path (str): path to figures folder
        stopwords (list): a list of stopwords to be removed from the wordcloud
    """
    n_gram_data_above_threshold = {
        key: value
        for key, value in n_gram_data.items()
        if value > min_frequency_allowed
    }
    n_gram_type = identify_n_gram_type(n_gram_data_above_threshold)

    if len(n_gram_data_above_threshold) > 0:
        create_wordcloud(n_gram_data_above_threshold, top_n, stopwords)
        plt.title(
            "Top {} {} with frequency above {}\nin {} for category\n`{}`".format(
                top_n, n_gram_type, min_frequency_allowed, var_used, category
            )
        )
        path_to_plot = os.path.join(
            fig_path, f"category_{category}_wordclouds_{n_gram_type}_{var_used}.png"
        )
        plt.savefig(path_to_plot, dpi=600)
        plt.clf()
    else:
        logger.warning(f"No {n_gram_type} above threshold for {category} {var_used}")
