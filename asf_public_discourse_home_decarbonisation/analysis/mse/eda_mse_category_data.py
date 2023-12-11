"""
This script performs exploratory data analysis on Money Saving Expert data, by looking at:
- A few statistics such as: number of posts collected, number of replies to posts and range of dates;
- Number of posts per year;
- Distribution of posts per user;
- Distribution of replies per user;
- Distribution of interactions per user.

Computed statistics are logged and appear in your terminal.
Plots are saved in the folder `asf_public_discourse_home_decarbonisation/outputs/figures/mse/`.

To run this script:
`python asf_public_discourse_home_decarbonisation/analysis/mse/eda_mse_category_data.py`

To change category or date/time range, use the following arguments:
`python asf_public_discourse_home_decarbonisation/analysis/mse/eda_mse_category_data.py --category <category> --collection_date_time <collection_date_time>`
where
<category>: category or sub-forum to be analysed, e.g. "energy", "lpg-heating-oil-solid-other-fuels", etc.
<collection_date_time>: data collection date/time in the format YYYY_MM_DD
"""

# ## Package imports
import warnings

# Ignoring warnings from Pandas
warnings.simplefilter(action="ignore")

import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
import os
import asf_public_discourse_home_decarbonisation.config.plotting_configs as pc
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
import argparse
import logging

logger = logging.getLogger(__name__)


def add_datetime_variables(mse_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the datetime variable into a datetime object and add year and year_month variables.
    Args:
        mse_data (pd.DataFrame): dataframe with MSE data
    Returns:
        pd.DataFrame: Pandas DataFrame with datetime variables added
    """
    mse_data["datetime"] = mse_data["datetime"].apply(
        lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S+00:00")
    )
    mse_data["year"] = mse_data["datetime"].apply(lambda x: x.year)
    mse_data["year_month"] = mse_data["datetime"].dt.to_period("M")

    return mse_data


def create_statistics_from_data(mse_data: pd.DataFrame):
    """
    Create and prints statistics from the MSE data.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
    """
    if len(mse_data) == len(mse_data.drop_duplicates()):
        logger.info("There are no duplicated posts in the data.")
    else:
        logger.warning(
            f"There are duplicated posts in the data.\nNumber instances: {len(mse_data)}\nNumber unique instances: {len(mse_data.drop_duplicates())}"
        )

    logger.info(f"Categories in data: {mse_data['category'].unique()}")

    logger.info(
        f"Number of instances (posts & replies) collected for category '{category}': {len(mse_data)}"
    )

    number_original_posts = len(mse_data[mse_data["is_original_post"] == 1])
    logger.info(
        f"Number of original posts collected for category '{category}': {number_original_posts}"
    )

    logger.info(
        f"Data collected ranges from {mse_data['datetime'].min()} to {mse_data['datetime'].max()}"
    )


def plot_number_posts_per_year(mse_data: pd.DataFrame, category: str):
    """
    Plots and saves a bar plot with number of posts per year.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        category (str): MSE category/sub-forum
    """
    posts_per_year = mse_data.groupby("year")[["id"]].nunique()
    posts_per_year.columns = ["number_posts"]
    posts_per_year.reset_index(inplace=True)

    chart = (
        alt.Chart(posts_per_year)
        .mark_bar()
        .encode(
            y=alt.Y("year:O", title="Year"),
            x=alt.X("number_posts:Q", title="Number of posts"),
            color=alt.value(pc.NESTA_COLOURS[1]),
        )
        .properties(width=400, height=300)
        .interactive()
    )
    chart = pc.configure_plots(chart, "Number of posts per year", "")
    path_to_plot = f"{MSE_FIGURES_PATH}/category_{category}_posts_per_year.html"
    chart.save(path_to_plot)

    logger.info(
        f"Plot with number of posts per year for category '{category}' saved successfully in {path_to_plot}"
    )


def check_username_id_correspondence(mse_data: pd.DataFrame):
    """
    Checks if there is a 1-1 correspondence between `username` and `user_id`.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
    """
    check_id_username = mse_data.groupby("username")[["user_id"]].nunique()

    usernames_corresponding_to_multiple_ids = check_id_username[
        check_id_username["user_id"] > 1
    ]

    if len(usernames_corresponding_to_multiple_ids) > 0:
        logger.warning(
            f"There are {len(usernames_corresponding_to_multiple_ids)} username(s) corresponding to multiple user_ids."
        )
        if set(usernames_corresponding_to_multiple_ids.index) == {"[Deleted User]"}:
            logger.info(
                "All usernames corresponding to multiple user_ids are Deleted Users."
            )
        else:
            logger.warning(
                "There are usernames corresponding to multiple user_ids that are not Deleted Users."
            )

    check_id_username2 = mse_data.groupby("user_id")[["username"]].nunique()

    ids_corresponding_to_multiple_usernames = check_id_username2[
        check_id_username2["username"] > 1
    ]
    if len(ids_corresponding_to_multiple_usernames) > 0:
        logger.warning(
            f"There are {len(ids_corresponding_to_multiple_usernames)} user IDs corresponding to multiple usernames."
        )


def plot_number_interactions_per_user(mse_data: pd.DataFrame, category: str):
    """
    Plots and saves a histogram with number of interactions per user.
    The computation is based on all MSE posts, not necessarily the ones we collected data for.

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        cateogry (str): MSE category/sub-forum
    """

    interactions_per_user_overall = mse_data[["user_id", "n_posts_user"]]

    # due to data collection at different times, numbers might differ, so we need to remove duplicates for each user
    interactions_per_user_overall.sort_values("n_posts_user", ascending=False)
    interactions_per_user_overall.drop_duplicates("user_id", keep="first", inplace=True)

    plt.figure(figsize=(pc.figsize_x, pc.figsize_y))
    plt.hist(
        interactions_per_user_overall["n_posts_user"],
        bins=30,
        color=pc.NESTA_COLOURS[0],
        edgecolor="white",
    )
    plt.xlabel("Number of posts and replies")
    plt.ylabel("Number of users")
    plt.title("Number of users with a given\nnumber of posts and replies")
    plt.yscale("log")
    plt.tight_layout()
    path_to_plot = f"{MSE_FIGURES_PATH}/category_{category}_interactions_per_user.png"
    plt.savefig(path_to_plot)
    logger.info(
        f"Plot with number of interactions per user for category '{category}' saved successfully in {path_to_plot}"
    )


def plot_number_posts_per_user(mse_data: pd.DataFrame, category: str):
    """
    Plots and saves a histogram with number of posts per user.
    The computation is based on data collected (i.e. posts collected from the specified category,
    not the whole MSE)

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        category (str): MSE category/sub-forum
    """
    posts_per_user_data_collected = (
        mse_data[mse_data["is_original_post"] == 1].groupby("user_id")[["id"]].nunique()
    )

    posts_per_user_data_collected.reset_index(inplace=True)
    posts_per_user_data_collected.columns = ["user", "n_posts_user"]

    plt.figure(figsize=(pc.figsize_x, pc.figsize_y))
    plt.hist(
        posts_per_user_data_collected["n_posts_user"],
        bins=range(0, 90, 5),
        color=pc.NESTA_COLOURS[0],
        edgecolor="white",
    )
    plt.xlabel("Number of posts")
    plt.ylabel("Number of users")
    plt.title(f"Number of users with a given number of posts\nin {category} category")
    plt.yscale("log")
    plt.tight_layout()
    path_to_plot = (
        f"{MSE_FIGURES_PATH}/category_{category}_posts_per_user_data_collected.png"
    )
    plt.savefig(path_to_plot)
    logger.info(
        f"Plot with number of posts per user for category '{category}' saved successfully in {path_to_plot}"
    )


def plot_number_replies_per_user(mse_data: pd.DataFrame, category: str):
    """
    Plots and saves a histogram with number of replies per user.
    The computation is based on data collected (i.e. replies collected from the specified category,
    not the whole MSE)

    Args:
        mse_data (pd.DataFrame): Money Saving Expert data
        category (str): MSE category/sub-forum
    """
    replies_per_user_data_collected = (
        mse_data[mse_data["is_original_post"] == 0].groupby("user_id")[["id"]].count()
    )

    replies_per_user_data_collected.reset_index(inplace=True)
    replies_per_user_data_collected.columns = ["user", "n_replies_user"]

    plt.figure(figsize=(pc.figsize_x, pc.figsize_y))
    plt.hist(
        replies_per_user_data_collected["n_replies_user"],
        bins=30,
        color=pc.NESTA_COLOURS[0],
        edgecolor="white",
    )
    plt.xlabel("Number of replies")
    plt.ylabel("Number of users")
    plt.title(f"Number of users with a given number of replies\nin {category} category")
    plt.yscale("log")
    plt.tight_layout()
    path_to_plot = (
        f"{MSE_FIGURES_PATH}/category_{category}_replies_per_user_data_collected.png"
    )
    plt.savefig(path_to_plot)
    logger.info(
        f"Plot with number of replies per user for category '{category}' saved successfully in {path_to_plot}"
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
    mse_data = add_datetime_variables(mse_data)

    create_statistics_from_data(mse_data)

    check_username_id_correspondence(mse_data)

    plot_number_posts_per_year(mse_data, category)

    plot_number_interactions_per_user(mse_data, category)
    plot_number_posts_per_user(mse_data, category)
    plot_number_replies_per_user(mse_data, category)
