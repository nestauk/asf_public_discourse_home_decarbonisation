"""
Functions for easier generation of plots.
"""

import altair as alt
from matplotlib import font_manager
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams
from collections import Counter
from nltk.probability import FreqDist
from wordcloud import WordCloud
from typing import List


# ChartType = alt.vegalite.v4.api.Chart
ChartType = alt.Chart
# Fonts and colours
FONT = "Averta"
TITLE_FONT = "Averta"
FONTSIZE_TITLE = 20
FONTSIZE_SUBTITLE = 16
FONTSIZE_NORMAL = 13


NESTA_COLOURS = [
    "#0000FF",
    "#18A48C",
    "#FDB633",
    "#9A1BBE",
    "#EB003B",
    "#FF6E47",
    "#646363",
    "#0F294A",
    "#97D9E3",
    "#A59BEE",
    "#F6A4B7",
    "#D2C9C0",
    "#FFFFFF",
    "#000000",
]


def nestafont():
    """Define Nesta fonts"""
    return {
        "config": {
            "title": {"font": TITLE_FONT, "anchor": "start"},
            "axis": {"labelFont": FONT, "titleFont": FONT},
            "header": {"labelFont": FONT, "titleFont": FONT},
            "legend": {"labelFont": FONT, "titleFont": FONT},
            "range": {
                "category": NESTA_COLOURS,
                "ordinal": {
                    "scheme": NESTA_COLOURS
                },  # this will interpolate the colors
            },
        }
    }


alt.themes.register("nestafont", nestafont)
alt.themes.enable("nestafont")


def configure_plots(fig, chart_title: str = "", chart_subtitle: str = ""):
    """
    Adds titles, subtitles, configures font sizes andadjusts gridlines
    """
    return (
        fig.properties(
            title={
                "anchor": "start",
                "text": chart_title,
                "fontSize": FONTSIZE_TITLE,
                "subtitle": chart_subtitle,
                "subtitleFont": FONT,
                "subtitleFontSize": FONTSIZE_SUBTITLE,
            },
        )
        .configure_axis(
            gridDash=[1, 7],
            gridColor="grey",
            labelFontSize=FONTSIZE_NORMAL,
            titleFontSize=FONTSIZE_NORMAL,
        )
        .configure_legend(
            titleFontSize=FONTSIZE_NORMAL,
            labelFontSize=FONTSIZE_NORMAL,
        )
        .configure_view(strokeWidth=0)
    )


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


def set_spines():
    """
    Function to add or remove spines from plots.
    """
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = True


def set_plotting_styles():
    """
    Function that sets plotting styles.
    """

    sns.set_context("talk")

    set_spines()

    # Had trouble making it find the font I set so this was the only way to do it
    # without specifying the local filepath
    all_font_files = font_manager.findSystemFonts()

    try:
        mpl.rcParams["font.family"] = "sans-serif"
        font_files = [f for f in all_font_files if "Averta-Regular" in f]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.sans-serif"] = "Averta"
    except:
        print("Averta" + " font could not be located. Using 'DejaVu Sans' instead")
        font_files = [f for f in all_font_files if "DejaVuSans.ttf" in f][0]
        for font_file in font_files:
            font_manager.fontManager.addfont(font_file)
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = "DejaVu Sans"

    mpl.rcParams["xtick.labelsize"] = 18
    mpl.rcParams["ytick.labelsize"] = 18
    mpl.rcParams["axes.titlesize"] = 20
    mpl.rcParams["axes.labelsize"] = 18
    mpl.rcParams["legend.fontsize"] = 18
    mpl.rcParams["figure.titlesize"] = 20


##### Plotting functions for the dataframe #####
font_path_ttf = finding_path_to_font("Averta-Regular")


######### 1. Distribution of posts over time #########
def plot_post_distribution_over_time(
    dataframe: pd.DataFrame, output_path: str, color: str
) -> None:
    """
    Analyse and plot the distribution of posts over time.

    Parameters:
        dataframe : pd.DataFrame
            The DataFrame containing the data.
        output_path : str
            The path to save the plot.
        color : str
            Color for the plot.

    Returns:
        None: This function does not return anything. It generates and saves a plot.
    """

    sample_size = dataframe.shape[0]
    dataframe["year"] = dataframe["date"].dt.year
    post_count_by_year = dataframe.groupby("year").size().reset_index(name="post_count")

    min_year = post_count_by_year["year"].min()
    max_year = post_count_by_year["year"].max()
    all_years = pd.DataFrame({"year": list(range(min_year, max_year + 1))})
    post_count_by_year = pd.merge(
        all_years, post_count_by_year, on="year", how="left"
    ).fillna(0)

    plt.figure(figsize=(12, 8))
    sns.barplot(x="year", y="post_count", data=post_count_by_year, color=color)
    plt.title(
        f"Distribution of Posts Over Time (By Year) (total sample size n = {sample_size})"
    )
    plt.xlabel("Year")
    plt.ylabel("Number of Posts")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Distribution_of_Posts_By_Year.png"))
    plt.show()


########## 2. Plot of the number of users vs number of posts ###########


def plot_users_post_distribution(dataframe, output_path):
    """
    Creates and saves a plot showing the distribution of the number of posts per user.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing user post data with a 'username' column.
        output_path (str): The directory path to save the output plot.

    Returns:
        None: The function saves the plot as a PNG file and displays it.
    """
    # Calculate the post counts per user
    user_post_counts = dataframe["username"].value_counts()

    # Count how many users have the same number of posts
    post_count_distribution = user_post_counts.value_counts().sort_index()

    # Create the plot
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(
        x=post_count_distribution.index,
        y=post_count_distribution.values,
        palette="coolwarm",
    )
    # Customize x-axis ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Set plot title and labels
    plt.title("Distribution of Number of Posts per User")
    plt.xlabel("Number of Posts")
    plt.ylabel("Number of Users")

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Post_Count_Distribution.png"))
    plt.show()


######### 3. Word cloud of frequently used words in posts ############
def plot_word_cloud(freq_dist, output_path, title="", threshold=10, max_words=10):
    """
    Generates and saves a word cloud from a frequency distribution of words.

    Args:
        freq_dist (dict): Frequency distribution of words.
        output_path (str): Directory path to save the word cloud image.
        title (str, optional): Title of the plot (e.g. bigram) and also distinguisher for filename.
        threshold (int, optional): Minimum frequency for words to be included. Defaults to 100.
        max_words (int, optional): Maximum number of words to display in the word cloud. Defaults to 30.

    Returns:
        None: The function saves and displays the word cloud image but does not return any value.
    """
    # Apply frequency filter
    filtered_freq_dist = {
        word: freq for word, freq in freq_dist.items() if freq >= threshold
    }

    # Generate the word cloud
    wordcloud = WordCloud(
        font_path=font_path_ttf,
        background_color="white",
        width=800,
        height=400,
        max_words=max_words,
    ).generate_from_frequencies(filtered_freq_dist)

    # Show and save the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Enhanced {title} Word Cloud (with Frequency Filter > {threshold})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"Word_Cloud_freq_filter_{title}.png"))
    plt.show()


######## 4.Generate bigram and trigram frequency distributions ########
def plot_top_ngrams(freq_dist, n, ngram_type, color, output_path):
    """
    Processes and plots the top N most common n-grams.

    Args:
        freq_dist (FreqDist): Frequency distribution of n-grams.
        n (int): Number of top n-grams to plot.
        ngram_type (str): Type of n-grams ('bigram' or 'trigram').
        color (str): Color for the plot.
        output_path (str): Path to save the plot.
    """
    # Get the top N most common n-grams
    top_ngrams = freq_dist.most_common(n)

    # Separate n-grams and frequencies
    ngram_labels, ngram_freqs = zip(*top_ngrams)

    # Convert n-gram tuples to strings for labeling
    ngram_labels = [" ".join(label) for label in ngram_labels]

    # Plot n-grams
    plt.figure(figsize=(12, 8))
    plt.bar(ngram_labels, ngram_freqs, color=color)
    plt.ylabel("Frequency")
    plt.xlabel(f"{ngram_type.title()}s")
    plt.xticks(rotation=45)
    plt.title(f"Top {n} Most Common {ngram_type.title()}s")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, f"Top_{n}_Most_Common_{ngram_type.title()}s.png")
    )
    plt.show()

    # Plot n-grams - horizontal
    plt.figure(figsize=(12, 8))
    plt.barh(ngram_labels, ngram_freqs, color=color)
    plt.xlabel("Frequency")
    plt.ylabel(f"{ngram_type.title()}s")
    plt.title(f"Top {n} Most Common {ngram_type.title()}s")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_path, f"Top_{n}_Most_Common_{ngram_type.title()}s_horizontal.png"
        )
    )
    plt.show()


############ 6. Distribution of post lengths #############


def plot_post_length_distribution(
    dataframe, output_path=None, bins=50, color=NESTA_COLOURS[2]
):
    """
    Plots and saves histograms of the post lengths in a given DataFrame.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing the text data.
        output_path (str, optional): Path to save the histogram images. If None, images are not saved.
        bins (int, optional): Number of bins for the histogram. Default is 50.
        color (str, optional): Color of the histogram bars. Default is 'blue'.

    This function creates two histograms:
    1. Standard scale histogram of post lengths.
    2. Log scale histogram of post lengths.
    """
    # Calculate post length
    dataframe["post_length"] = dataframe["text"].apply(lambda x: len(str(x).split()))

    # Plot standard scale histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(dataframe["post_length"], bins=bins, kde=False, color=color)
    plt.title("Distribution of Post Lengths")
    plt.xlabel("Post Length (Number of Words)")
    plt.ylabel("Number of Posts")
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, "distribution_of_post_lengths.png"))
    plt.show()

    # Plot log scale histogram
    plt.figure(figsize=(12, 8))
    sns.histplot(dataframe["post_length"], bins=bins, kde=False, color=color)
    plt.title("Distribution of Post Lengths")
    plt.xlabel("Post Length (Number of Words)")
    plt.ylabel("Number of Posts (log scale)")
    plt.yscale("log")
    plt.tight_layout()
    if output_path:
        plt.savefig(
            os.path.join(output_path, "distribution_of_post_lengths_logscale.png")
        )
    plt.show()


############ 7. Frequency of selected keywords in posts using the dictionary ###########
# Function for vertical bar chart plotting
def plot_tag_vertical_bar_chart(dataframe, output_path, filename, df_threshold):
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Tag", y="Frequency", data=dataframe, palette="winter")
    plt.title(f"Frequency of Selected Keywords in Posts (Frequency > {df_threshold})")
    plt.xlabel("Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, filename))
    plt.show()


# Function for horizontal bar chart plotting
def plot_tag_horizontal_bar_chart(dataframe, output_path, filename, df_threshold):
    plt.figure(figsize=(12, 8))
    plt.barh(dataframe["Tag"], dataframe["Frequency"], color="blue")
    plt.title(f"Frequency of Selected Keywords in Posts (Frequency > {df_threshold})")
    plt.ylabel("Tag")
    plt.xlabel("Frequency")
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, filename))
    plt.show()
