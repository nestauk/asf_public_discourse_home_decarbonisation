import numpy as np
import pandas as pd
import itertools
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
import nltk

nltk.download("stopwords")
import matplotlib.pyplot as plt
import altair as alt
import os
import asf_public_discourse_home_decarbonisation.utils.plotting_utils as pu
from asf_public_discourse_home_decarbonisation.getters.mse_getters import (
    get_first_attempt_mse_data,
    get_all_mse_data,
    get_mse_category_data,
)
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_text,
    create_ngram_frequencies,
    english_stopwords_definition,
    lemmatize_sentence_2,
)
from asf_public_discourse_home_decarbonisation import PROJECT_DIR

# %% [markdown]
# ## Data import

# %% [markdown]
# Either choose a specific category or get the original sample of data we collected:

# %%
# Choose one category/sub-forum from the list below
# list_of_categories_collected = ["green-ethical-moneysaving", "lpg-heating-oil-solid-other-fuels", "is-this-quote-fair",  "energy"]
category = "green-ethical-moneysaving"
mse_data = get_mse_category_data(category)

# %%
# Alternatively comment the code in the cell above, and uncomment this cell to get a sample of data
# mse_data = get_first_attempt_mse_data()
# category = "green-ethical-moneysaving"

# %%
MSE_FIGURES_PATH = f"{PROJECT_DIR}/outputs/figures/mse/{category}"
os.makedirs(MSE_FIGURES_PATH, exist_ok=True)

# %%
mse_data.head()

# %%
len(mse_data)

# %%
len(mse_data.drop_duplicates())

# %%
mse_data["category"].unique()

# %% [markdown]
# ## Basic Processing

# %%
mse_data["datetime"] = mse_data["datetime"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S+00:00")
)

# %%
mse_data["date"] = mse_data["datetime"].apply(lambda x: x.date())
mse_data["time"] = mse_data["datetime"].apply(lambda x: x.time())
mse_data["year"] = mse_data["datetime"].apply(lambda x: x.year)
mse_data["month"] = mse_data["datetime"].apply(lambda x: x.month)
mse_data["day"] = mse_data["datetime"].apply(lambda x: x.day)
mse_data["year_month"] = mse_data["datetime"].dt.to_period("M")

# %%
mse_data.head()

# %%


# %% [markdown]
# ## Text Processing

# %% [markdown]
# Apply basic processing to text data and identifying stopwords:

# %%
mse_data["processed_text"] = mse_data["text"].apply(lambda x: process_text(x))

# %%
mse_data["processed_title"] = mse_data["title"].apply(lambda x: process_text(x))

# %%
stopwords = english_stopwords_definition()

# %% [markdown]
# Creating tokens from title and text of post/replies:

# %%
mse_data["tokens_title"] = mse_data["processed_title"].apply(word_tokenize)

# %%
mse_data["tokens_text"] = mse_data["processed_text"].apply(word_tokenize)

# %% [markdown]
# Creating a list with all tokens (either in title or text of posts/replies) and creating a dictonary that maps tokens to lemmatised tokens:

# %%
tokens_title = mse_data[mse_data["is_original_post"] == 1][["tokens_title"]]
tokens_title = tokens_title.explode("tokens_title").rename(
    columns={"tokens_title": "tokens"}
)
tokens_text = mse_data[["tokens_text"]]
tokens_text = tokens_text.explode("tokens_text").rename(
    columns={"tokens_text": "tokens"}
)

# %%
all_tokens = pd.concat([tokens_text, tokens_title])

# %%
all_tokens = all_tokens.drop_duplicates("tokens")

# %%
all_tokens["is_digit"] = all_tokens["tokens"].str.isdigit()


# %%
# Removing digits as they cannot be lemmatized (needs to explicitly to have `== False` because is_digit can also be missing)
all_tokens = all_tokens[all_tokens["is_digit"] == False]

# %%
all_tokens

# %%
lemmas_dictionary = lemmatize_sentence_2(list(all_tokens["tokens"]))

# %% [markdown]
# Lemmatising tokens:

# %%
# using np.vectorize seems to be the fastest way of lemmatizing
replace_words_vec = np.vectorize(
    lambda sentence: " ".join(
        lemmas_dictionary.get(word, word) for word in sentence.split()
    )
)

# %%
mse_data["processed_title"] = replace_words_vec(mse_data["processed_title"])

# %%
mse_data["processed_text"] = replace_words_vec(mse_data["processed_text"])

# %% [markdown]
# Recreating tokens:

# %%
mse_data["tokens_title"] = mse_data["processed_title"].apply(word_tokenize)

# %%
mse_data["tokens_text"] = mse_data["processed_text"].apply(word_tokenize)

# %% [markdown]
# Removing stopwords from lists of tokens:

# %%
mse_data["tokens_title"] = mse_data["tokens_title"].apply(
    lambda x: [token for token in x if token not in stopwords]
)
mse_data["tokens_text"] = mse_data["tokens_text"].apply(
    lambda x: [token for token in x if token not in stopwords]
)

# %% [markdown]
# Creating new columns with n-grams info based on title and text of post/reply

# %%
mse_data["bigrams_text"] = mse_data.apply(
    lambda x: create_ngram_frequencies(x["tokens_text"], n=2), axis=1
)
mse_data["bigrams_title"] = mse_data.apply(
    lambda x: create_ngram_frequencies(x["tokens_title"], n=2), axis=1
)

# %%
mse_data["trigrams_text"] = mse_data.apply(
    lambda x: create_ngram_frequencies(x["tokens_text"], n=3), axis=1
)
mse_data["trigrams_title"] = mse_data.apply(
    lambda x: create_ngram_frequencies(x["tokens_title"], n=3), axis=1
)

# %% [markdown]
# ## Exploratory data analysis

# %% [markdown]
# ### Range of dates

# %%
mse_data["datetime"].min(), mse_data["datetime"].max(), mse_data["datetime"].median()

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
        color=alt.value(pu.NESTA_COLOURS[1]),
    )
    .properties(width=400, height=300)
    .interactive()
)
chart = pu.configure_plots(chart, "Number of posts per year", "")
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
# 2. Number of interactions per user (considering the whole MSE), where an interaction is a post + a reply; number of posts and number of replies

# %% [markdown]
# Number of interactions:

# %%
interactions_per_user_overall = mse_data[["user_id", "n_posts_user"]]

# due to data collection at different times, numbers might differ, so we need to remove duplicates for each user
interactions_per_user_overall.sort_values("n_posts_user", ascending=False)
interactions_per_user_overall.drop_duplicates("user_id", keep="first", inplace=True)

# %%
interactions_per_user_overall = (
    interactions_per_user_overall.groupby("n_posts_user")
    .size()
    .reset_index(name="user_count")
)

# %%
plt.figure(figsize=(8, 4))
plt.hist(
    interactions_per_user_overall["n_posts_user"], bins=30, color=pu.NESTA_COLOURS[0]
)
plt.xlabel("Number of posts and replies")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of posts and replies")
plt.yscale("log")
plt.savefig(f"{MSE_FIGURES_PATH}/category_{category}_interactions_per_user.png")


# %% [markdown]
# Number of posts:

# %%
posts_per_user_overall = mse_data[mse_data["is_original_post"] == 1][
    ["user_id", "n_posts_user"]
]

# due to data collection at different times, numbers might differ, so we need to remove duplicates for each user
posts_per_user_overall.sort_values("n_posts_user", ascending=False)
posts_per_user_overall.drop_duplicates("user_id", keep="first", inplace=True)

# %%
posts_per_user_overall = (
    posts_per_user_overall.groupby("n_posts_user").size().reset_index(name="user_count")
)

# %%
plt.figure(figsize=(8, 4))
plt.hist(posts_per_user_overall["n_posts_user"], bins=30, color=pu.NESTA_COLOURS[0])
plt.xlabel("Number of posts")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of posts")
plt.yscale("log")
plt.savefig(f"{MSE_FIGURES_PATH}/category_{category}_posts_per_user.png")


# %% [markdown]
# Number of replies:

# %%
replies_per_user_overall = mse_data[mse_data["is_original_post"] == 0][
    ["user_id", "n_posts_user"]
]

# due to data collection at different times, numbers might differ, so we need to remove duplicates for each user
replies_per_user_overall.sort_values("n_posts_user", ascending=False)
replies_per_user_overall.drop_duplicates("user_id", keep="first", inplace=True)

# %%
replies_per_user_overall = (
    replies_per_user_overall.groupby("n_posts_user")
    .size()
    .reset_index(name="user_count")
)

# %%
replies_per_user_overall

# %%
plt.figure(figsize=(8, 4))
plt.hist(replies_per_user_overall["n_posts_user"], bins=30, color=pu.NESTA_COLOURS[0])
plt.xlabel("Number of replies")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of replies")
plt.yscale("log")
plt.savefig(f"{MSE_FIGURES_PATH}/category_{category}_replies_per_user.png")


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
posts_per_user_data_collected = (
    posts_per_user_data_collected.groupby("n_posts_user")
    .size()
    .reset_index(name="user_count")
)

# %%
plt.figure(figsize=(8, 4))
plt.hist(
    posts_per_user_data_collected["n_posts_user"], bins=30, color=pu.NESTA_COLOURS[0]
)
plt.xlabel("Number of posts")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of posts (in data collected)")
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
replies_per_user_data_collected = (
    replies_per_user_data_collected.groupby("n_replies_user")
    .size()
    .reset_index(name="user_count")
)

# %%
plt.figure(figsize=(8, 4))
plt.hist(
    posts_per_user_data_collected["n_posts_user"], bins=30, color=pu.NESTA_COLOURS[0]
)
plt.xlabel("Number of replies")
plt.ylabel("Number of users")
plt.title("Number of users with a given number of replies (in data collected)")
plt.savefig(
    f"{MSE_FIGURES_PATH}/category_{category}_replies_per_user_data_collected.png"
)


# %% [markdown]
# ## How many posts contain certain keywords?


# %%
def number_posts_containing_keywords(data, keyword_list):
    return len(
        data[
            data["title"].str.contains("|".join(keyword_list), case=False)
            | data["text"].str.contains("|".join(keyword_list), case=False)
        ]
    )


# %%
heat_pump_keywords = ["heat pump", "ashp", "gshp", "wshp", "air 2 air"]
heat_pump_posts = number_posts_containing_keywords(mse_data, heat_pump_keywords)
heat_pump_posts, heat_pump_posts / len(mse_data) * 100

# %%
boiler_keywords = ["boiler"]
boiler_posts = number_posts_containing_keywords(mse_data, boiler_keywords)
boiler_posts, boiler_posts / len(mse_data) * 100

# %%
hydrogen_keywords = ["hydrogen"]
hydrogen_posts = number_posts_containing_keywords(mse_data, hydrogen_keywords)
hydrogen_posts, hydrogen_posts / len(mse_data) * 100

# %%
bus_keywords = ["boiler upgrade scheme", "bus"]
bus_posts = number_posts_containing_keywords(mse_data, bus_keywords)
bus_posts, bus_posts / len(mse_data) * 100

# %%
grants_keywords = bus_keywords + [
    "renewable heat incentive",
    "domestic rhi",
    "clean heat grant",
    "home energy scotland grant",
    "home energy scotland loan",
    "home energy scotland scheme",
]
grants_posts = number_posts_containing_keywords(mse_data, grants_keywords)
grants_posts, grants_posts / len(mse_data) * 100

# %%
mcs_keywords = ["mcs", "microgeneration certification scheme"]
mcs_posts = number_posts_containing_keywords(mse_data, mcs_keywords)
mcs_posts, mcs_posts / len(mse_data) * 100

# %%
cost_est_keywords = ["cost estimator"]
cost_est_posts = number_posts_containing_keywords(mse_data, cost_est_keywords)
cost_est_posts, cost_est_posts / len(mse_data) * 100

# %%
nesta_keywords = ["nesta"]
number_posts_containing_keywords(mse_data, nesta_keywords)

# %%
installer_keywords = ["installer", "engineer"]
number_posts_containing_keywords(
    mse_data, installer_keywords
), number_posts_containing_keywords(mse_data, installer_keywords) / len(mse_data) * 100

# %%
installation_keywords = ["installation"]
number_posts_containing_keywords(
    mse_data, installation_keywords
), number_posts_containing_keywords(mse_data, installation_keywords) / len(
    mse_data
) * 100

# %%
cost_keywords = ["cost"]
number_posts_containing_keywords(
    mse_data, cost_keywords
), number_posts_containing_keywords(mse_data, cost_keywords) / len(mse_data) * 100

# %%
issue_keywords = ["issue"]
number_posts_containing_keywords(
    mse_data, issue_keywords
), number_posts_containing_keywords(mse_data, issue_keywords) / len(mse_data) * 100

# %%
noise_keywords = ["noise", "noisy"]
number_posts_containing_keywords(
    mse_data, noise_keywords
), number_posts_containing_keywords(mse_data, noise_keywords) / len(mse_data) * 100

# %% [markdown]
# ## Top words & n-grams

# %%
number_ngrams_wordcloud = 30

# %%
# def frequency_tokens(data, tokens_col):

#     # Remove records looking exactly the same
#     tokens = data[[tokens_col]].drop_duplicates()

#     # Each line corresponds to a token (hence, it includes repetitions)
#     tokens = tokens.explode(tokens_col)
#     tokens["count"]=1

#     # Computing a count for each token
#     tokens = tokens.groupby(tokens_col, as_index=False).sum()

#     # Creating a dictionary of frequencies
#     freq_tokens = tokens.set_index(tokens_col)["count"].to_dict()

#     return freq_tokens

# %%
# def frequency_ngrams(data, ngrams_col):
#     ngrams = ngrams[ngrams_col].tolist()
#     ngrams= list(itertools.chain.from_iterable(ngrams))

#     frequency_ngrams = nltk.FreqDist(ngrams)

#     return frequency_ngrams

# %%
from collections import Counter


def frequency_ngrams(data, ngrams_col):
    ngrams = [ng for sublist in data[ngrams_col].tolist() for ng in sublist]
    frequency_ngrams = Counter(ngrams)

    return frequency_ngrams


# %% [markdown]
# ## All data

# %% [markdown]
# ### Titles: most common keywords

# %%
freq_tokens_titles = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1].drop_duplicates("id"), "tokens_title"
)

# %%
freq_tokens_titles = {
    key: value for key, value in freq_tokens_titles.items() if value > 10
}
len(freq_tokens_titles)


# %%
pu.create_wordcloud(
    freq_tokens_titles,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordclouds_tokens_titles",
)


# %% [markdown]
# ### Titles: most common bigrams

# %%
freq_bigrams_titles = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1].drop_duplicates("id"), "bigrams_title"
)

# %%
freq_bigrams_titles = {
    key: value for key, value in freq_bigrams_titles.items() if value > 10
}
len(freq_bigrams_titles)


# %%
pu.create_wordcloud(
    freq_bigrams_titles, 30, stopwords, f"category_{category}_wordcloud_bigrams_titles"
)

# %% [markdown]
# ### Titles: most common trigrams

# %%
freq_trigrams_titles = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1].drop_duplicates("id"), "trigrams_title"
)

# %%
freq_trigrams_titles = {
    key: value for key, value in freq_trigrams_titles.items() if value > 10
}
len(freq_trigrams_titles)

# %%
pu.create_wordcloud(
    freq_trigrams_titles,
    15,
    stopwords,
    f"category_{category}_wordcloud_trigrams_titles",
)

# %% [markdown]
# ### Posts & replies text: most common keywords

# %%
freq_tokens_text = frequency_ngrams(mse_data, "tokens_text")

# %%
freq_tokens_text = {key: value for key, value in freq_tokens_text.items() if value > 10}
len(freq_tokens_text)

# %%
pu.create_wordcloud(
    freq_tokens_text,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordclouds_tokens_text",
)

# %% [markdown]
# ### Posts & replies text: most common bigrams

# %%
freq_bigrams_text = frequency_ngrams(mse_data, "bigrams_text")

# %%
freq_bigrams_text = {
    key: value for key, value in freq_bigrams_text.items() if value > 10
}
len(freq_bigrams_text)

# %%
pu.create_wordcloud(
    freq_bigrams_text, 30, stopwords, f"category_{category}_wordcloud_bigrams_text"
)

# %% [markdown]
# ### Posts & replies text: most common trigrams

# %%
freq_trigrams_text = frequency_ngrams(mse_data, "trigrams_text")

# %%
freq_trigrams_text = {
    key: value for key, value in freq_trigrams_text.items() if value > 10
}
len(freq_trigrams_text)

# %%
pu.create_wordcloud(
    freq_trigrams_text,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordcloud_trigrams_text",
)

# %% [markdown]
# ### Just post text (no replies): most common keywords
# op = original post

# %%
tokens_text_op = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1], "tokens_text"
)

# %%
tokens_text_op = {key: value for key, value in tokens_text_op.items() if value > 10}
len(tokens_text_op)

# %%
pu.create_wordcloud(
    tokens_text_op, 30, stopwords, f"category_{category}_wordclouds_tokens_text_op"
)

# %% [markdown]
# ### Just post text (no replies): most common bigrams
# op = original post

# %%
freq_bigrams_text_op = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1], "bigrams_text"
)

# %%
freq_bigrams_text_op = {
    key: value for key, value in freq_bigrams_text_op.items() if value > 10
}
len(freq_bigrams_text_op)

# %%
pu.create_wordcloud(
    freq_bigrams_text_op,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordcloud_bigrams_text_op",
)

# %% [markdown]
# ### Just post text (no replies): most common trigrams
# op = original post

# %%
freq_trigrams_text_op = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1], "trigrams_text"
)

# %%
freq_trigrams_text_op = {
    key: value for key, value in freq_trigrams_text_op.items() if value > 10
}
len(freq_trigrams_text_op)

# %%
pu.create_wordcloud(
    freq_trigrams_text_op,
    30,
    stopwords,
    f"category_{category}_wordcloud_trigrams_text_op",
)

# %% [markdown]
# ## Heat Pump data

# %% [markdown]
# ### Titles (when HPs mentioned): most common keywords

# %%
hp_data = mse_data[
    mse_data["title"].str.contains("|".join(heat_pump_keywords), case=False)
    | mse_data["text"].str.contains("|".join(heat_pump_keywords), case=False)
]

# %%
freq_tokens_titles_hp = frequency_ngrams(
    hp_data[hp_data["is_original_post"] == 1].drop_duplicates("id"), "tokens_title"
)

# %%
freq_tokens_titles_hp = {
    key: value for key, value in freq_tokens_titles_hp.items() if value > 10
}
len(freq_tokens_titles_hp)

# %%
pu.create_wordcloud(
    freq_tokens_titles_hp,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordclouds_tokens_titles_hp",
)

# %% [markdown]
# ### Titles (when HPs mentioned): most common bigrams

# %%
freq_bigrams_titles_hp = frequency_ngrams(
    hp_data[hp_data["is_original_post"] == 1].drop_duplicates("id"), "bigrams_title"
)

# %%
freq_bigrams_titles_hp = {
    key: value for key, value in freq_bigrams_titles_hp.items() if value > 10
}
len(freq_bigrams_titles_hp)

# %%
pu.create_wordcloud(
    freq_bigrams_titles_hp,
    30,
    stopwords,
    f"category_{category}_wordcloud_bigrams_titles_hp",
)

# %% [markdown]
# ### Titles (when HPs mentioned): most common trigrams

# %%
freq_trigrams_titles_hp = frequency_ngrams(
    hp_data[hp_data["is_original_post"] == 1].drop_duplicates("id"), "trigrams_title"
)

# %%
freq_trigrams_titles_hp = {
    key: value for key, value in freq_trigrams_titles_hp.items() if value > 10
}
len(freq_trigrams_titles_hp)

# %%
pu.create_wordcloud(
    freq_trigrams_titles_hp,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordcloud_trigrams_titles_hp",
)

# %% [markdown]
# ### Posts & replies text (when HPs mentioned): most common keywords

# %%
freq_tokens_text_hp = frequency_ngrams(hp_data, "tokens_text")

# %%
freq_tokens_text_hp = {
    key: value for key, value in freq_tokens_text_hp.items() if value > 10
}
len(freq_tokens_text_hp)

# %%
pu.create_wordcloud(
    freq_tokens_text_hp, 30, stopwords, f"category_{category}_wordclouds_tokens_text_hp"
)

# %% [markdown]
# ### Posts & replies text (when HPs mentioned): most common bigrams

# %%
freq_bigrams_text_hp = frequency_ngrams(hp_data, "bigrams_text")

# %%
freq_bigrams_text_hp = {
    key: value for key, value in freq_bigrams_text_hp.items() if value > 10
}
len(freq_bigrams_text_hp)

# %%
pu.create_wordcloud(
    freq_bigrams_text_hp,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordcloud_bigrams_text_hp",
)

# %% [markdown]
# ### Posts & replies text (when HPs mentioned): most common trigrams

# %%
freq_trigrams_text_hp = frequency_ngrams(hp_data, "trigrams_text")

# %%
freq_trigrams_text_hp = {
    key: value for key, value in freq_trigrams_text_hp.items() if value > 10
}
len(freq_trigrams_text_hp)

# %%
pu.create_wordcloud(
    freq_trigrams_text_hp,
    30,
    stopwords,
    f"category_{category}_wordcloud_trigrams_text_hp",
)

# %% [markdown]
# ## Cost data

# %% [markdown]
# ### Titles (when cost mentioned): most common keywords

# %%
cost_data = mse_data[
    mse_data["title"].str.contains("|".join(cost_keywords), case=False)
    | mse_data["text"].str.contains("|".join(cost_keywords), case=False)
]

# %%
freq_tokens_titles_cost = frequency_ngrams(
    cost_data[cost_data["is_original_post"] == 1].drop_duplicates("id"), "tokens_title"
)

# %%
freq_tokens_titles_cost = {
    key: value for key, value in freq_tokens_titles_cost.items() if value > 10
}
len(freq_tokens_titles_cost)

# %%
pu.create_wordcloud(
    freq_tokens_titles_cost,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordclouds_tokens_titles_cost",
)

# %% [markdown]
# ### Titles (when cost mentioned): most common bigrams

# %%
freq_bigrams_titles_cost = frequency_ngrams(
    cost_data[cost_data["is_original_post"] == 1].drop_duplicates("id"), "bigrams_title"
)

# %%
freq_bigrams_titles_cost = {
    key: value for key, value in freq_bigrams_titles_cost.items() if value > 10
}
len(freq_bigrams_titles_cost)

# %%
pu.create_wordcloud(
    freq_bigrams_titles_cost,
    30,
    stopwords,
    f"category_{category}_wordcloud_bigrams_titles_cost",
)

# %% [markdown]
# ### Titles (when cost mentioned): most common trigrams

# %%
freq_trigrams_titles_cost = frequency_ngrams(
    cost_data[cost_data["is_original_post"] == 1].drop_duplicates("id"),
    "trigrams_title",
)

# %%
freq_trigrams_titles_cost = {
    key: value for key, value in freq_trigrams_titles_cost.items() if value > 10
}
len(freq_trigrams_titles_cost)

# %%
pu.create_wordcloud(
    freq_trigrams_titles_cost,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordcloud_trigrams_titles_cost",
)

# %% [markdown]
# ### Posts & replies text (when cost mentioned): most common keywords

# %%
freq_tokens_text_cost = frequency_ngrams(cost_data, "tokens_text")

# %%
freq_tokens_text_cost = {
    key: value for key, value in freq_tokens_text_cost.items() if value > 10
}
len(freq_tokens_text_cost)

# %%
pu.create_wordcloud(
    freq_tokens_text_cost,
    30,
    stopwords,
    f"category_{category}_wordclouds_tokens_text_cost",
)

# %% [markdown]
# ### Posts & replies text (when cost mentioned): most common bigrams

# %%
freq_bigrams_text_cost = frequency_ngrams(cost_data, "bigrams_text")

# %%
freq_bigrams_text_cost = {
    key: value for key, value in freq_bigrams_text_cost.items() if value > 10
}
len(freq_bigrams_text_cost)

# %%
pu.create_wordcloud(
    freq_bigrams_text_cost,
    number_ngrams_wordcloud,
    stopwords,
    f"category_{category}_wordcloud_bigrams_text_cost",
)

# %% [markdown]
# ### Posts & replies text (when cost mentioned): most common trigrams

# %%
freq_trigrams_text_cost = frequency_ngrams(cost_data, "trigrams_text")

# %%
freq_trigrams_text_cost = {
    key: value for key, value in freq_trigrams_text_cost.items() if value > 10
}
len(freq_trigrams_text_cost)

# %%
pu.create_wordcloud(
    freq_bigrams_text_cost,
    30,
    stopwords,
    f"category_{category}_wordcloud_bigrams_text_cost",
)

# %%
