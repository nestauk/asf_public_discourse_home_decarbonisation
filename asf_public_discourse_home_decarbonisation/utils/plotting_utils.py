"""
Functions for easier generation of plots.
"""

from matplotlib import font_manager
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from asf_public_discourse_home_decarbonisation.config.plotting_configs import FONT_NAME


def finding_path_to_font(font_name: str):
    """
    Finds path to specific font.
    Args:
        font_name: name of font
    """

    all_font_files = font_manager.findSystemFonts()
    font_files = [f for f in all_font_files if font_name in f]
    if len(font_files) == 0:
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f]
    return font_files[0]


def create_wordcloud(frequencies, max_words, stopwords, category, fig_name):
    """
    Creates word cloud based on frequencies.
    Args:
        frequencies: frequencies of words or n-grams
        max_words: maximum number of words or n-grams to be displayed
        stopwords: stopwords that should be removed from the wordcloud
    """
    font_path_ttf = finding_path_to_font(FONT_NAME)

    wordcloud = WordCloud(
        font_path=font_path_ttf,
        width=1000,
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

    # plt.savefig(
    #     f"{MSE_FIG_PATH}/{category}/{fig_name}.png",
    #     dpi=600,
    # )
