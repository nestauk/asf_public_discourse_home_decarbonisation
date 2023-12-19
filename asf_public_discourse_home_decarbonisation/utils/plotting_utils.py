"""
Functions for easier generation of plots.
"""
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    NESTA_COLOURS,
    finding_path_to_font,
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
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

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
    threshold: Optional[int] = 1000,
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
    freq_dist: FreqDist, n: int, ngram_type: str, color: str, output_path: str
):
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
