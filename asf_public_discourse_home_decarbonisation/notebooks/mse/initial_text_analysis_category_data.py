# %% [markdown]
# In this notebook we perform an initial text analysis on Money Saving Expert data, by:
# - Computing the frenquency os specific words and n-grams of interest;
# - Identiyfying top words and n-grams;
#
#
# To open it as a notebook please run the following in your terminal:
#
# `jupytext --to notebook asf_public_discourse_home_decarbonisation/notebooks/mse/initial_text_analysis_category_data.py`
#
# If the correct kernel does not come up (`asf_public_discourse_home_decarbonisation`), please run the following in your terminal:
#
# `python -m ipykernel install --user --name=asf_public_discourse_home_decarbonisation`
#
# This notebook has been refactored into a python script. You can run the analysis as a script by running the following in your terminal:
#
# `python asf_public_discourse_home_decarbonisation/analysis/mse/initial_text_analysis_category_data.py`

# %% [markdown]
# ## Package imports

# %%
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import os
import asf_public_discourse_home_decarbonisation.utils.plotting_utils as pu
import asf_public_discourse_home_decarbonisation.config.plotting_configs as pc
from asf_public_discourse_home_decarbonisation.getters.mse_getters import (
    get_first_attempt_mse_data,
    get_all_mse_data,
    get_mse_category_data,
)
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_text,
    create_ngram_frequencies,
    english_stopwords_definition,
    lemmatize_sentence,
    frequency_ngrams,
)
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    plot_and_save_top_ngrams,
    plot_and_save_wordcloud,
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

# %%
MSE_FIGURES_PATH = f"{PROJECT_DIR}/outputs/figures/mse/{category}"
os.makedirs(MSE_FIGURES_PATH, exist_ok=True)

# %% [markdown]
# Path to figures:

# %%
mse_data.head()

# %%
len(mse_data), len(mse_data.drop_duplicates())

# %%
mse_data["category"].unique()

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
# vectorizing_tokenization = np.vectorize(lambda sentence: word_tokenize(sentence))

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
# Removing digits as they cannot be lemmatized (needs to explicitly to have `== False` because is_digit can also be missing)
all_tokens["is_digit"] = all_tokens["tokens"].str.isdigit()
all_tokens = all_tokens[all_tokens["is_digit"] == False]

# %%
all_tokens

# %%
lemmas_dictionary = lemmatize_sentence(list(all_tokens["tokens"]))

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
# ## How many posts contain certain keywords?


# %%
def number_instances_containing_keywords(
    data: pd.DataFrame, keyword_list: list, filter: str = "all"
) -> int:
    """
    Computes the number of instances in the data that contain at least one of the keywords in the keyword list.

    Args:
        data (pd.DataFrame): Dataframe containing the text data
        keyword_list (list): A list of keywords to search for
        filter (str): filter for "posts", "replies" or "all" where "all" contains "posts" and "replies"

    Returns:
        int: Number of instances containing at least one of the keywords in the keyword list
    """
    data_containing_keywords = data[
        data["title"].str.contains("|".join(keyword_list), case=False)
        | data["text"].str.contains("|".join(keyword_list), case=False)
    ]
    if filter == "posts":
        return data_containing_keywords[
            data_containing_keywords["is_original_post"] == 1
        ].shape[0]
    elif filter == "replies":
        return data_containing_keywords[
            data_containing_keywords["is_original_post"] == 0
        ].shape[0]
    elif filter == "all":
        return data_containing_keywords.shape[0]
    else:
        raise ValueError(
            f"{filter} is not a valid filter! Choose between 'posts', 'replies' or 'all'"
        )


# %%
keyword_dictionary = {
    "heat_pump_keywords": ["heat pump", "ashp", "gshp", "wshp", "air 2 air"],
    "boiler_keywords": ["boiler"],
    "hydrogen_keywords": ["hydrogen"],
    "bus_keywords": ["boiler upgrade scheme", "bus"],
    "grants_keywords": [
        "boiler upgrade scheme",
        "bus",
        "renewable heat incentive",
        "domestic rhi",
        "clean heat grant",
        "home energy scotland grant",
        "home energy scotland loan",
        "home energy scotland scheme",
    ],
    "mcs_keywords": ["mcs", "microgeneration certification scheme"],
    "cost_est_keywords": ["cost estimator"],
    "nesta_keywords": ["nesta"],
    "installer_keywords": ["installer", "engineer"],
    "installation_keywords": ["installation"],
    "cost_keywords": ["cost", "price", "pay"],
    "issue_keywords": ["issue"],
    "noise_keywords": ["noise", "noisy"],
    "flow_temp_keywords": ["flow temperature", "flow temp"],
    "msbc_keywords": ["money saving boiler challenge", "msbc", "boiler challenge"],
}

# %%
keyword_counts = pd.DataFrame()
keyword_counts["group"] = keyword_dictionary.keys()

# %%
keyword_counts["counts"] = keyword_counts["group"].apply(
    lambda x: number_instances_containing_keywords(mse_data, keyword_dictionary[x])
)

# %%
keyword_counts["percentage"] = keyword_counts["counts"] / len(mse_data) * 100

# %%
keyword_counts.sort_values("counts", ascending=False)

# %% [markdown]
# ## Top words & n-grams

# %%
number_ngrams_wordcloud = 30
min_frequency_tokens = 100
min_frequency_bigrams = 50
min_frequency_trigrams = 10
top_ngrams_barplot = 10

# %% [markdown]
# #### Titles: most common keywords

# %%
freq_tokens_titles = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1].drop_duplicates("id"), "tokens_title"
)

# %%
plot_and_save_top_ngrams(
    freq_tokens_titles, top_ngrams_barplot, category, "titles", MSE_FIGURES_PATH
)

# %%
plot_and_save_wordcloud(
    freq_tokens_titles,
    number_ngrams_wordcloud,
    min_frequency_tokens,
    category,
    "titles",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Titles: most common bigrams

# %%
freq_bigrams_titles = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1].drop_duplicates("id"), "bigrams_title"
)

# %%
plot_and_save_top_ngrams(
    freq_bigrams_titles, top_ngrams_barplot, category, "titles", MSE_FIGURES_PATH
)

# %%
plot_and_save_wordcloud(
    freq_bigrams_titles,
    number_ngrams_wordcloud,
    min_frequency_bigrams,
    category,
    "titles",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Titles: most common trigrams

# %%
freq_trigrams_titles = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1].drop_duplicates("id"), "trigrams_title"
)

# %%
plot_and_save_top_ngrams(
    freq_trigrams_titles, top_ngrams_barplot, category, "titles", MSE_FIGURES_PATH
)

# %%
plot_and_save_wordcloud(
    freq_trigrams_titles,
    number_ngrams_wordcloud,
    min_frequency_trigrams,
    category,
    "titles",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Posts & replies text: most common keywords

# %%
freq_tokens_text = frequency_ngrams(mse_data, "tokens_text")

# %%
plot_and_save_top_ngrams(
    freq_tokens_text,
    top_ngrams_barplot,
    category,
    "posts and replies",
    MSE_FIGURES_PATH,
)

# %%
plot_and_save_wordcloud(
    freq_tokens_text,
    number_ngrams_wordcloud,
    min_frequency_tokens,
    category,
    "posts and replies",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Posts & replies text: most common bigrams

# %%
freq_bigrams_text = frequency_ngrams(mse_data, "bigrams_text")

# %%
plot_and_save_top_ngrams(
    freq_bigrams_text,
    top_ngrams_barplot,
    category,
    "posts and replies",
    MSE_FIGURES_PATH,
)

# %%
plot_and_save_wordcloud(
    freq_bigrams_text,
    number_ngrams_wordcloud,
    min_frequency_bigrams,
    category,
    "all text",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Posts & replies text: most common trigrams

# %%
freq_trigrams_text = frequency_ngrams(mse_data, "trigrams_text")

# %%
plot_and_save_top_ngrams(
    freq_trigrams_text, top_ngrams_barplot, category, "all text", MSE_FIGURES_PATH
)

# %%
plot_and_save_wordcloud(
    freq_trigrams_text,
    number_ngrams_wordcloud,
    min_frequency_trigrams,
    category,
    "all text",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Just post text (no replies): most common keywords
# op = original post

# %%
tokens_text_op = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1], "tokens_text"
)

# %%
plot_and_save_top_ngrams(
    tokens_text_op, top_ngrams_barplot, category, "original posts", MSE_FIGURES_PATH
)

# %%
plot_and_save_wordcloud(
    tokens_text_op,
    number_ngrams_wordcloud,
    min_frequency_tokens,
    category,
    "original posts",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Just post text (no replies): most common bigrams
# op = original post

# %%
freq_bigrams_text_op = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1], "bigrams_text"
)

# %%
plot_and_save_top_ngrams(
    freq_bigrams_text_op,
    top_ngrams_barplot,
    category,
    "original posts",
    MSE_FIGURES_PATH,
)

# %%
plot_and_save_wordcloud(
    freq_bigrams_text_op,
    number_ngrams_wordcloud,
    min_frequency_bigrams,
    category,
    "original posts",
    MSE_FIGURES_PATH,
    stopwords,
)

# %% [markdown]
# #### Just post text (no replies): most common trigrams
# op = original post

# %%
freq_trigrams_text_op = frequency_ngrams(
    mse_data[mse_data["is_original_post"] == 1], "trigrams_text"
)

# %%
plot_and_save_top_ngrams(
    freq_trigrams_text_op,
    top_ngrams_barplot,
    category,
    "original posts",
    MSE_FIGURES_PATH,
)

# %%
plot_and_save_wordcloud(
    freq_trigrams_text_op,
    number_ngrams_wordcloud,
    min_frequency_bigrams,
    category,
    "original posts",
    MSE_FIGURES_PATH,
    stopwords,
)

# %%
