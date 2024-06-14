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
from asf_public_discourse_home_decarbonisation.utils.ngram_utils import (
    identify_n_gram_type,
)
import os
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pandas as pd
import numpy as np
import math
from nltk.probability import FreqDist
from wordcloud import WordCloud
from typing import Dict, Optional, List
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


# MSE utils
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


# BuildHub utils
set_plotting_styles()
font_path_ttf = finding_path_to_font("Averta-Regular")


######### 1. Distribution of posts over time #########
def plot_post_distribution_over_time(
    dataframe: pd.DataFrame, output_path: str, color: str
):
    """
    Analyse and plot the distribution of posts over time.

    Parameters:
        dataframe : pd.DataFrame
            The DataFrame containing the data.
        output_path : str
            The path to save the plot.
        color : str
            Color for the plot.
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


def plot_users_post_distribution(dataframe: pd.DataFrame, output_path: str):
    """
    Creates and saves a plot showing the distribution of the number of posts vs the number of users.

    Args:
        dataframe (pandas.DataFrame): The DataFrame containing user post data with a 'username' column.
        output_path (str): The directory path to save the output plot.

    Returns:
        None: The function saves the plot as a PNG file and displays it.
    """
    # Calculate the post counts per user
    user_post_counts = dataframe["username"].value_counts()
    sample_size = user_post_counts.size
    logger.info(f"Sample size: {sample_size}")
    # Count how many users have the same number of posts
    post_count_distribution = user_post_counts.value_counts().sort_index()
    plt.figure(figsize=(12, 8))
    # Apply the Freedman-Diaconis Rule to determine bin size
    iqr = np.subtract(*np.percentile(post_count_distribution, [75, 25]))
    multiplier = 2.5  # Increase this for larger bins
    bin_width_fd = multiplier * iqr * (sample_size ** (-1 / 3))
    bin_size_fd = math.ceil(
        (max(post_count_distribution) - min(post_count_distribution)) / bin_width_fd
    )
    max_bins = 50  # Adjust as needed
    bin_size_fd = min(bin_size_fd, max_bins)
    post_range = max(post_count_distribution.index) - min(post_count_distribution.index)
    number_of_bins = post_range / bin_size_fd
    # Create a histogram
    plt.hist(
        post_count_distribution.index,
        bins=bin_size_fd,
        weights=post_count_distribution.values,
    )
    # Set the base size for locator based on the number of bins
    if number_of_bins > 50:
        base_size = 200
    elif number_of_bins > 30:
        base_size = 100
    elif number_of_bins > 10:
        base_size = 50
    else:
        base_size = 5
    loc = ticker.MultipleLocator(
        base=base_size
    )  # this locator puts ticks at regular intervals
    plt.gca().xaxis.set_major_locator(loc)

    # Set the title and labels
    plt.title("Number of Posts vs Number of Users")
    plt.xlabel("Number of Posts")
    plt.ylabel("Number of Users")
    # Rotate the x-tick labels for better readability if there are many bars
    plt.xticks(rotation=90)
    # Use a logarithmic scale for the y-axis if needed
    plt.yscale("log")
    # Set grid lines to help readability
    plt.grid(axis="y", linestyle="--", linewidth=1.5, alpha=0.7)
    # Show the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "Post_Count_Distribution.png"))
    plt.show()


######### 3. Word cloud of frequently used words in posts ############
def plot_word_cloud(
    freq_dist: Dict[str, int],
    output_path: str,
    title: Optional[str] = "",
    threshold: Optional[int] = 100,
):
    """
    Generates and saves a word cloud from a frequency distribution of words.

    Args:
        freq_dist (dict): Frequency distribution of words.
        output_path (str): Directory path to save the word cloud image.
        title (str, optional): Title of the plot (e.g. bigram) and also distinguisher for filename.
        threshold (int, optional): Minimum frequency for words to be included. Defaults to 1000.

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
def plot_top_ngrams(
    freq_dist: FreqDist,
    ngram_type: str,
    color: str,
    output_path: str,
    orientation: str = "horizontal",
    n: Optional[int] = None,
    threshold: Optional[int] = 20,
):
    """
    Processes and plots the top N most common n-grams or n-grams above a certain frequency threshold.

    Args:
        freq_dist (FreqDist): Frequency distribution of n-grams.
        ngram_type (str): Type of n-grams ('bigram' or 'trigram').
        color (str): Color for the plot.
        output_path (str): Path to save the plot.
        orientation (str, optional): Orientation of the plot ('horizontal' or 'vertical'). Defaults to 'horizontal'.
        n (int, optional): Number of top n-grams to plot. Defaults to None.
        threshold (int, optional): Minimum frequency threshold for n-grams to be included. Defaults to 20.
    """
    # Get the n-grams based on the specified condition
    top_ngrams = freq_dist.most_common(n)
    ngram_labels, ngram_freqs = zip(*top_ngrams)
    filtered_ngram_freqs = tuple(k for k in ngram_freqs if k >= threshold)
    n_filter_top = len(filtered_ngram_freqs)
    if n_filter_top < 3:
        return logging.info("Not enough entries as n-grams are below the threshold.")
    filtered_ngram_label = ngram_labels[:n_filter_top]
    # Convert n-gram tuples to strings for labeling
    filtered_ngram_labels = [" ".join(label) for label in filtered_ngram_label]
    if orientation == "horizontal":
        # Plot n-grams - horizontal
        plt.figure(figsize=(12, 8))
        plt.barh(filtered_ngram_labels, filtered_ngram_freqs, color=color)
        plt.xlabel("Frequency")
        plt.ylabel(f"{ngram_type.title()}s")
    else:
        # Plot n-grams - vertical
        plt.figure(figsize=(12, 8))
        plt.bar(filtered_ngram_labels, filtered_ngram_freqs, color=color)
        plt.ylabel("Frequency")
        plt.xlabel(f"{ngram_type.title()}s")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            output_path, f"Top_{n}_Most_Common_{ngram_type.title()}s_{orientation}.png"
        )
    )
    plt.show()


############ 6. Distribution of post lengths #############


def plot_post_length_distribution(
    dataframe: pd.DataFrame,
    output_path: Optional[str] = None,
    bins: int = 50,
    color: str = NESTA_COLOURS[2],
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
def plot_tag_vertical_bar_chart(
    dataframe: pd.DataFrame, output_path: str, filename: str, threshold_freq: int
):
    """
    Plots and saves a vertical bar chart showing the frequency of selected tags in posts.
    This function creates a vertical bar chart where each bar represents a tag and its frequency.
    Only tags with frequency greater than threshold_freq are displayed.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing tag frequencies with columns "Tag" and "Frequency".
        output_path (str): Directory path to save the output plot.
        filename (str): Name of the file to save the plot.
        threshold_freq (int): Threshold frequency value for filtering tags.

    """
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Tag", y="Frequency", data=dataframe, palette="winter")
    plt.title(f"Frequency of Selected Keywords in Posts (Frequency > {threshold_freq})")
    plt.xlabel("Tag")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, filename))
    plt.show()


# Function for horizontal bar chart plotting
def plot_tag_horizontal_bar_chart(
    dataframe: pd.DataFrame, output_path: str, filename: str, threshold_freq: int
):
    """
    Plots and saves a horizontal bar chart showing the frequency of selected tags in posts.
    This function creates a horizontal bar chart where each bar represents a tag and its frequency.
    Only tags with frequency greater than threshold_freq are displayed.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing tag frequencies with columns "Tag" and "Frequency".
        output_path (str): Directory path to save the output plot.
        filename (str): Name of the file to save the plot.
        threshold_freq (int): Threshold frequency value for filtering tags.

    """
    plt.figure(figsize=(12, 8))
    plt.barh(dataframe["Tag"], dataframe["Frequency"], color="blue")
    plt.title(f"Frequency of Selected Keywords in Posts (Frequency > {threshold_freq})")
    plt.ylabel("Tag")
    plt.xlabel("Frequency")
    plt.tight_layout()
    if output_path:
        plt.savefig(os.path.join(output_path, filename))
    plt.show()


def format_key_terms(key_terms_colour_dict: Dict[str, str]) -> str:
    """
    Format the key terms for display. If there are two key terms, they are joined with ' and '.
    If there are three or more key terms, the first n-1 terms are joined with ', ', and then the last term is appended with ' and '.

    Parameters:
    key_terms_colour_dict (Dict[str, str]): A dictionary where keys are the key terms and values are their corresponding colors.

    Returns:
    str: The formatted string of key terms.
    """
    key_terms = list(key_terms_colour_dict.keys())
    if len(key_terms) == 2:
        return " and ".join([term.capitalize() for term in key_terms])
    else:
        return (
            ", ".join([term.capitalize() for term in key_terms[:-1]])
            + " and "
            + key_terms[-1].capitalize()
        )


def plot_mentions_line_chart(
    df_monthly: pd.DataFrame, key_terms_colour_dict: dict, plot_type: str = "both"
):
    """
    Plot the distribution of posts mentioning the key terms over time.

    Parameters:
    df_monthly (pd.DataFrame): The resampled dataframe with mentions and averages.
    key_terms_colour_dict (dict): Dictionary of key terms to plot and their corresponding colors.
    plot_type (str): The type of plot to display. Options are "rolling", "both", "discrete". Default is "both".
    """
    plt.figure(figsize=(12, 6))

    # Plot scatter and line plots for each key term
    for term, colour in key_terms_colour_dict.items():
        column_name = f"mentions_{term.replace(' ', '_')}"
        avg_column_name = f"{column_name}_avg"

        if plot_type == "rolling":
            plt.plot(
                df_monthly.index,
                df_monthly[avg_column_name],
                label=f"{term.capitalize()} Mentions",
                color=colour,
            )
        elif plot_type == "both":
            plt.scatter(
                df_monthly.index,
                df_monthly[column_name],
                alpha=0.3,
                label=f"{term.capitalize()} Mentions",
                color=colour,
            )
            plt.plot(
                df_monthly.index,
                df_monthly[avg_column_name],
                label=f"{term.capitalize()} Rolling Average",
                color=colour,
            )
        elif plot_type == "discrete":
            plt.plot(
                df_monthly.index,
                df_monthly[column_name],
                label=f"{term.capitalize()} Mentions",
                color=colour,
            )

    # Set the labels and title
    plt.xlabel("Date")
    plt.ylabel("Number of Posts and Replies")
    # Get the key terms and join them with commas
    key_terms = format_key_terms(key_terms_colour_dict)
    plt.title(f"Distribution of Posts and Replies Mentioning {key_terms} over Time")

    # Set y-axis gridlines
    plt.grid(axis="y", color="0.95")
    plt.legend()
    plt.show()
