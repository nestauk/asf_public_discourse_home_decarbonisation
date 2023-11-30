# %% [markdown]
# In this notebook we perform exploratory data analysis on Money Saving Expert data, by looking at:
# - A few statistics such as: number of posts collected, number of replies to posts and range of dates;
# - Number of posts per year;
# - Distribution of posts per user;
# - Distribution of replies per user;
# - Distribution of interactions per user.
#
#
# To open it as a notebook please run the following in your terminal:
#
# `jupytext --to notebook asf_public_discourse_home_decarbonisation/notebooks/mse/eda_mse_category_data.py`
#
# If the correct kernel does not come up (`asf_public_discourse_home_decarbonisation`), please run the following in your terminal:
#
# `python -m ipykernel install --user --name=asf_public_discourse_home_decarbonisation`
#
# This notebook has been refactored into a python script. You can run the analysis as a script by running the following in your terminal:
#
# `python asf_public_discourse_home_decarbonisation/analysis/mse/eda_mse_category_data.py`

# %% [markdown]
# ## Package imports

# %%
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
import os
import asf_public_discourse_home_decarbonisation.config.plotting_configs as pc
from asf_public_discourse_home_decarbonisation.getters.mse_getters import (
    get_first_attempt_mse_data,
    get_all_mse_data,
    get_mse_category_data,
)
from asf_public_discourse_home_decarbonisation import PROJECT_DIR

# %% [markdown]
# ## Data import

# %% [markdown]
# Choose between **one of the existing categories**, **all data** (i.e. all categories together) and the first **sample** of data collected:

# %%
# One of the collected MSE categories
category = "green-ethical-moneysaving"  # Remaining categories: "lpg-heating-oil-solid-other-fuels", "is-this-quote-fair", "energy"
# All categories data together
# category = "all"
# A sample of data
# category = "sample"


# %% [markdown]
# Data collection date/time:

# %%
collection_datetime = "2023_11_15"

# %% [markdown]
# Getting the data from S3:

# %%
if category in [
    "green-ethical-moneysaving",
    "lpg-heating-oil-solid-other-fuels",
    "is-this-quote-fair",
    "energy",
]:
    mse_data = get_mse_category_data(category, collection_datetime)
elif category == "sample":
    category = "green-ethical-moneysaving"
    mse_data = get_mse_category_data(category, collection_datetime)
elif category == "all":
    mse_data = get_all_mse_data(collection_datetime)
else:
    print(f"{category} is not a valid category!")

# %% [markdown]
# Path to figures:

# %%
MSE_FIGURES_PATH = f"{PROJECT_DIR}/outputs/figures/mse/{category}"
os.makedirs(MSE_FIGURES_PATH, exist_ok=True)

# %% [markdown]
# First few insights into the data:

# %%
mse_data.head()

# %%
len(mse_data), len(mse_data.drop_duplicates())

# %%
mse_data["category"].unique()

# %% [markdown]
# ## Basic Processing

# %%
mse_data["datetime"] = mse_data["datetime"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S+00:00")
)
mse_data["year"] = mse_data["datetime"].apply(lambda x: x.year)
mse_data["year_month"] = mse_data["datetime"].dt.to_period("M")

# %% [markdown]
# ## Exploratory data analysis

# %% [markdown]
# ### A few statistics

# %%
print(
    f"Number of instances (posts & replies) collected for category '{category}': {len(mse_data)}"
)

# %%
number_original_posts = len(mse_data[mse_data["is_original_post"] == 1])
print(
    f"Number of original posts collected for category '{category}': {number_original_posts}"
)

# %% [markdown]
# ### Range of dates

# %%
print(
    f"Data collected ranges from {mse_data['datetime'].min()} to {mse_data['datetime'].max()}"
)

# %% [markdown]
# Number of posts per year

# %%
posts_per_year = mse_data.groupby("year")[["id"]].nunique()
posts_per_year.columns = ["number_posts"]
posts_per_year.reset_index(inplace=True)

# %%
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
chart.save(f"{MSE_FIGURES_PATH}/category_{category}_posts_per_year.html")
chart


# %% [markdown]
# ### Users
#
# 1. Double checking there is a 1-1 correspondence between `username` and `user_id`:
#
# Aparently, if a user deletes their profile, the username appears as `[Deleted User]` but we can still use the `user_id` to track posts across this user.
#
# If we want to uniquely identify users, we should use the `user_id`.

# %%
check_id_username = mse_data.groupby("username")[["user_id"]].nunique()

# %%
check_id_username[check_id_username["user_id"] > 1]

# %%
mse_data[mse_data["username"] == "[Deleted User]"]

# %%
check_id_username2 = mse_data.groupby("user_id")[["username"]].nunique()

# %%
check_id_username2[check_id_username2["username"] > 1]

# %% [markdown]
# 2. Number of interactions per user (considering the whole MSE)

# %% [markdown]
# Number of interactions:

# %%
interactions_per_user_overall = mse_data[["user_id", "n_posts_user"]]

# due to data collection at different times, numbers might differ, so we need to remove duplicates for each user
interactions_per_user_overall.sort_values("n_posts_user", ascending=False)
interactions_per_user_overall.drop_duplicates("user_id", keep="first", inplace=True)

# %%
plt.figure(figsize=(8, 4))
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
plt.savefig(f"{MSE_FIGURES_PATH}/category_{category}_interactions_per_user.png")


# %% [markdown]
# 3. Number of posts per user (considering collected data)

# %%
posts_per_user_data_collected = (
    mse_data[mse_data["is_original_post"] == 1].groupby("user_id")[["id"]].nunique()
)

# %%
posts_per_user_data_collected.reset_index(inplace=True)
posts_per_user_data_collected.columns = ["user", "n_posts_user"]

# %%
plt.figure(figsize=(8, 4))
plt.hist(
    posts_per_user_data_collected["n_posts_user"],
    bins=range(0, 90, 5),
    color=pc.NESTA_COLOURS[0],
    edgecolor="white",
)
plt.xlabel("Number of posts")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of posts\n(in data collected)")
plt.yscale("log")
plt.tight_layout()
plt.savefig(f"{MSE_FIGURES_PATH}/category_{category}_posts_per_user_data_collected.png")


# %% [markdown]
# 4. Number of replies per user

# %%
replies_per_user_data_collected = (
    mse_data[mse_data["is_original_post"] == 0].groupby("user_id")[["id"]].count()
)

# %%
replies_per_user_data_collected.reset_index(inplace=True)
replies_per_user_data_collected.columns = ["user", "n_replies_user"]

# %%
plt.figure(figsize=(8, 4))
plt.hist(
    replies_per_user_data_collected["n_replies_user"],
    bins=30,
    color=pc.NESTA_COLOURS[0],
    edgecolor="white",
)
plt.xlabel("Number of replies")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of replies\n(in data collected)")
plt.yscale("log")
plt.tight_layout()
plt.savefig(
    f"{MSE_FIGURES_PATH}/category_{category}_replies_per_user_data_collected.png"
)


# %%
