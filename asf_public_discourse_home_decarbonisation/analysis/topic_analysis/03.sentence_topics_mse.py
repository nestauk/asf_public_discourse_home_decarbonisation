# %% [markdown]
# # Identifying topics of conversation from sentences and how they change over time in Money Saving Expert
#
# What the code in this notebook does:
#
# - Loads data from Money Saving Expert (MSE) forum threads;
# - Cleans the data by removing URLs, username patterns/introductory patterns and by processing abbreviations (replacing them with their full forum). Note that no additional data cleaning is needed as BERTopic is able to handle the data as is;
# - Break text into **sentences**;
# - Uses BERTopic to identify topics of conversation in the **sentences**;
#     - Optionally assigning outliers to their closest topic;
# - Plots the topics of conversation in the forum threads;
# - Computing growth rates of topics from 2020 to 2022;
# - Plots the changes in topics of conversation over time;
# - We use BERTopic to recluster certain topics of conversation - like noise!
#
#
# To convert this script to a jupyter notebook and explore results:
# ```
# pip install jupytext
# jupytext --to notebook 03.sentence_topics_mse.py
# ```

# %%
# package imports
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import openpyxl  # not "used" but needed to save excel files
from bertopic import BERTopic
from umap import UMAP
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
)
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    remove_urls,
    remove_username_pattern,
    remove_introduction_patterns,
    process_abbreviations,
    ends_with_punctuation,
)
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    NESTA_FLOURISH_COLOURS,
)

# %%
# setting plotting styles
set_plotting_styles()

# %%
# If you want to assign each document to a topic, set this to True and you'll have no outliers
reduce_outliers_to_zero = False
only_include_sentences_mentioning_hp = False

# %% [markdown]
# ## 1. Loading, data coleaning and & filtering data

# %% [markdown]
# ### 1.1. Load data

# %%
mse_data = get_mse_data(
    category="all", collection_date="2023_11_15", processing_level="raw"
)

# %% [markdown]
# ### 1.2. Data cleaning

# %%
mse_data["text"] = mse_data["text"].apply(remove_urls)
mse_data["text"] = mse_data["text"].apply(remove_username_pattern)
mse_data["text"] = mse_data["text"].apply(remove_introduction_patterns)

# %%
mse_data["title"] = mse_data.apply(
    lambda x: "" if x["is_original_post"] == 0 else x["title"], axis=1
)
mse_data["whole_text"] = mse_data.apply(
    lambda x: (
        x["title"] + " " + x["text"]
        if (ends_with_punctuation(x["title"]) or x["is_original_post"] == 0)
        else x["title"] + ". " + x["text"]
    ),
    axis=1,
)

# %%
# Processing abbreviations such as "ashp" to "air source heat pump"
mse_data["whole_text"] = mse_data["whole_text"].apply(process_abbreviations)

# %% [markdown]
# ### 1.3. Adding date/time variable

# %%
mse_data["datetime"] = pd.to_datetime(mse_data["datetime"])
mse_data["date"] = mse_data["datetime"].dt.date
mse_data["year"] = mse_data["datetime"].dt.year

# %% [markdown]
# ### 1.4. Focusing on posts from the past 5 years

# %%
mse_data = mse_data[mse_data["year"] >= 2018]

# %%
len(mse_data)

# %% [markdown]
# ### 1.5. Focusing on conversations mentioning HPs

# %%
ids_to_keep = mse_data[
    (mse_data["whole_text"].str.contains("heat pump", case=False))
    & (mse_data["is_original_post"] == 1)
]["id"].unique()

# %%
mse_data = mse_data[mse_data["id"].isin(ids_to_keep)]

# %%
len(mse_data)

# %% [markdown]
# ### 1.6. Adding space after punctuation ("?", ".", "!"), so that sentences are correctly split
#
# Sentence tokenizers don't work well with punctuation marks that don't have a space after them. We add a space after punctuation marks to ensure that sentences are correctly split.

# %%
# adding space after punctuation
mse_data["whole_text"] = mse_data["whole_text"].apply(
    lambda t: re.sub(r"([.!?])([A-Za-z])", r"\1 \2", t)
)

# %% [markdown]
# ### 1.7. Transforming text into sentences and striping white sapces

# %%

mse_data["sentences"] = mse_data["whole_text"].apply(sent_tokenize)

# %%
sentences_data = mse_data.explode("sentences")
sentences_data["sentences"] = sentences_data["sentences"].astype(str)
sentences_data["sentences"] = sentences_data["sentences"].str.strip()

# %% [markdown]
# ### 1.8. Remove small sentences

# %%
# Looking to identify any other patterns we might want to remove
sentences_data["sentences"].str.lower().value_counts().head(10)

# %%
sentences_data["tokens"] = sentences_data["sentences"].apply(word_tokenize)
sentences_data["non_punctuation_tokens"] = sentences_data["tokens"].apply(
    lambda x: [token for token in x if token not in string.punctuation]
)

# %%
sentences_data["n_tokens"] = sentences_data["non_punctuation_tokens"].apply(len)

# %%
len(sentences_data[sentences_data["n_tokens"] > 5]) / len(sentences_data)

# %%
sentences_data = sentences_data[sentences_data["n_tokens"] > 5]

# %% [markdown]
# ### 1.9. Removing very common sentences

# %%
# Looking to identify any other patterns we might want to remove
sentences_data["sentences"].str.lower().value_counts().head(20)

# %%
sentences_data = sentences_data[
    ~(
        sentences_data["sentences"].str.contains("thank", case=False)
        | sentences_data["sentences"].str.contains("happy to help", case=False)
        | sentences_data["sentences"].str.contains("kind wishes", case=False)
        | sentences_data["sentences"].str.contains("kind regards", case=False)
    )
]

# %%
# Looking to identify any other patterns we might want to remove
sentences_data["sentences"].str.lower().value_counts().head(10)

# %% [markdown]
# ## 2. Topic identification

# %% [markdown]
# ### 2.1. Defining text variable used for topic analysis
#
# - We're also including sentences coming from replies;
# - We always deduplicate tthe sentences in use as duplicates shouldn't be used when applying BERTopic.

# %%
len(sentences_data)

# %%
if only_include_sentences_mentioning_hp:
    sentences_data = sentences_data[
        sentences_data["sentences"].str.contains("heat pump", case=False)
    ]
    print(len(sentences_data))

# %%


# %%
docs = sentences_data.drop_duplicates("sentences")["sentences"]
dates = list(sentences_data.drop_duplicates("sentences")["date"])

# %%
len(docs)

# %%


# %% [markdown]
# ### 2.2. Topic model definition
#
# The arguments below should be changed for each specific dataset:
# - We use BERTopic to identify topics of conversation in the forum threads.
# - We're using `CountVectorizer(stop_words="english")` as the vectorizer to have meaningful representations, without stopwords;
#     - Alternatively you can use OpenAI to generate a custom topic label for each topic;
# - We're setting `random_state=42` in UMAP for reproducibility, not changing any of the remaining default parameters used in BERTopic UMAP;
# - We're setting `nr_topics=100` as the number of topics to identify so that we have a manageable number of topics.

# %%
vectorizer_model = CountVectorizer(stop_words="english")
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
if reduce_outliers_to_zero:
    topic_model = BERTopic(
        umap_model=umap_model,
        min_topic_size=100,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )
else:
    topic_model = BERTopic(
        umap_model=umap_model, min_topic_size=100, vectorizer_model=vectorizer_model
    )

topics, probs = topic_model.fit_transform(docs)

# %%
topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
# clusters/topics
len(topics_info) - 1

# %%
# shows information about each topic
# notice that Topic -1 is the outliers topic
topics_info

# %% [markdown]
# ### 2.3. Exploring the presence of HP sentences in the outlier cluster

# %%
len(
    doc_info[
        (doc_info["Topic"] == -1) & (doc_info["Document"].str.contains("heat pump"))
    ]
) / len(doc_info[doc_info["Topic"] == -1]) * 100

# %%
doc_info[(doc_info["Topic"] == -1) & (doc_info["Document"].str.contains("heat pump"))][
    "Document"
].values

# %%
doc_info[(doc_info["Topic"] == -1) & ~(doc_info["Document"].str.contains("heat pump"))][
    "Document"
].values[:20]

# %% [markdown]
# ### 2.4. Looking for topics containing specific keywords
#
# - We're looking for topics containing specific keywords to identify topics of conversation related to:
#     - "heat pumps"
#     - "boilers"
#     - "noise"
#     - "efficiency"
#     - "cost

# %%
topics_info[
    topics_info["Name"].str.contains("heat") | topics_info["Name"].str.contains("pump")
]

# %%
topics_info[topics_info["Name"].str.contains("radiator")]

# %%
topics_info[topics_info["Name"].str.contains("boiler")]

# %%
topics_info[topics_info["Name"].str.contains("tariff")]

# %%
topics_info[
    topics_info["Name"].str.contains("nois") | topics_info["Name"].str.contains("quiet")
]

# %%
topics_info[
    topics_info["Name"].str.contains("cost")
    | topics_info["Name"].str.contains("price")
    | topics_info["Name"].str.contains("expensive")
    | topics_info["Name"].str.contains("cheap")
]

# %%
topics_info[
    topics_info["Name"].str.contains("planning")
    | topics_info["Name"].str.contains("permi")
]

# %%
topics_info[topics_info["Name"].str.contains("cold")]

# %%
topics_info[
    topics_info["Name"].str.contains("tumble") | topics_info["Name"].str.contains("dry")
]

# %%
topics_info[
    topics_info["Name"].str.contains("hydrogen")
    | topics_info["Name"].str.contains("h2")
]

# %%
topics_info[topics_info["Name"].str.contains("effic")]

# %% [markdown]
# ### 2.5. Optionally assining a topic to each document in the outliers cluster
#
# - If we set `reduce_outliers_to_zero` to True in the beginning of the notebook, then we can assign the outliers to their closest topic.
# - Be careful when doing so, you might need to do some manual checking.

# %%
if reduce_outliers_to_zero:
    new_topics = topic_model.reduce_outliers(
        docs, topics, probabilities=probs, strategy="probabilities"
    )
    topic_model.update_topics(docs, topics=new_topics)
    topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)
    topics_info.sort_values("Count", ascending=False)

# %% [markdown]
# Looking at changes in number of docs in specific topics:

# %%
topics_info[
    topics_info["Name"].str.contains("heat") | topics_info["Name"].str.contains("pump")
]

# %%
topics_info[topics_info["Name"].str.contains("radiator")]

# %%
topics_info[topics_info["Name"].str.contains("boiler")]

# %%
topics_info[topics_info["Name"].str.contains("tariff")]

# %%
topics_info[
    topics_info["Name"].str.contains("nois") | topics_info["Name"].str.contains("quiet")
]

# %%
topics_info[
    topics_info["Name"].str.contains("cost")
    | topics_info["Name"].str.contains("price")
    | topics_info["Name"].str.contains("expensive")
    | topics_info["Name"].str.contains("cheap")
]

# %%
topics_info[
    topics_info["Name"].str.contains("planning")
    | topics_info["Name"].str.contains("permi")
]

# %%
topics_info[topics_info["Name"].str.contains("cold")]

# %%
topics_info[
    topics_info["Name"].str.contains("tumble") | topics_info["Name"].str.contains("dry")
]

# %%
topics_info[
    topics_info["Name"].str.contains("hydrogen")
    | topics_info["Name"].str.contains("h2")
]

# %%
topics_info[topics_info["Name"].str.contains("effic")]

# %% [markdown]
# ### 2.6. When doing topic analysis, we remove duplicates. Here we add them back on.
#
# - We add duplicates back on. These represent different forum posts with the same title!
# - Note that the order of topics might change!

# %%
sentences_data

# %%
data = sentences_data[["sentences", "datetime", "date", "year", "id"]]

# %%
data = data.merge(doc_info, left_on="sentences", right_on="Document")

# %%
updated_topics_info = (
    data.groupby("Topic", as_index=False)[["id"]]
    .count()
    .rename(columns={"id": "updated_count"})
    .merge(topics_info, on="Topic")
)

# %%
updated_topics_info["updated_%"] = (
    updated_topics_info["updated_count"]
    / sum(updated_topics_info["updated_count"])
    * 100
)

# %%
updated_topics_info = updated_topics_info.sort_values(
    "updated_count", ascending=False
).reset_index(drop=True)

# %%
updated_topics_info

# %% [markdown]
# ## 3. Changes in topics over time & growth in the past couple of years
#
# Here we compute:
#
# - Number of documents per topic and  per year;
# - Average number of documents in a rolling window of 3 years;
# - Growth rates for each topic between 2020 and 2022.

# %%
topics_per_year = data.groupby(["Topic", "year"])[["sentences"]].count()

# %%
topics_per_year.head()

# %%
topics_per_year = topics_per_year.pivot_table(
    index="year", columns="Topic", values="sentences"
)
topics_per_year.fillna(0, inplace=True)

# %%
topics_per_year

# %%
smoothed_avg_topics_per_year = topics_per_year.rolling(window=3, axis=0).mean()

# %%
smoothed_avg_topics_per_year


# %%
def growth_rate(topic):
    return (
        (
            smoothed_avg_topics_per_year.loc[2022][topic]
            - smoothed_avg_topics_per_year.loc[2020][topic]
        )
        / smoothed_avg_topics_per_year.loc[2020][topic]
        * 100
    )


growth_rate_2018_2022 = smoothed_avg_topics_per_year.columns.to_series().apply(
    growth_rate
)

# %%
updated_topics_info["growth_rate"] = updated_topics_info["Topic"].apply(
    lambda x: growth_rate_2018_2022[x]
)

# %%
updated_topics_info.sort_values("growth_rate", ascending=False).head(20)

# %% [markdown]
# ## 4. Using BERTopic to look at changes over time
#
# BERTopic offers the functionality to automatically identify changes in topics of time.

# %%
data["date_str"] = data["datetime"].dt.strftime("%Y-%m-%d")
dates = list(data.drop_duplicates("sentences")["date_str"])

# %%
topics_over_time = topic_model.topics_over_time(docs, dates, nr_bins=5)

# %%
topics_over_time

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=range(10), width=900, height=450
)

# %%


# %% [markdown]
# ## 5. Exploring subtopics of conversation
#
# Here we re-cluster certain topics of conversation to identify subtopics.

# %% [markdown]
# ### 5.1. Exploring conversations about noise

# %%
topics_info[topics_info["Name"].str.contains("noise")]

# %%
noise_topic = 11

# %%
topics_info[topics_info["Topic"] == noise_topic]

# %%
noise_docs = list(doc_info[doc_info["Topic"] == noise_topic]["Document"])

# %%
noise_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=50, vectorizer_model=vectorizer_model
)
noise_topics, noise_probs = noise_topic_model.fit_transform(noise_docs)

# %%
noise_topics_, noise_topics_info, noise_doc_info = get_outputs_from_topic_model(
    noise_topic_model, noise_docs
)

# %%
noise_topics_info

# %% [markdown]
# ### 5.2. Exploring conversations about ashps

# %%
topics_info[topics_info["Name"].str.contains("pump")]

# %%
ashp_topic = 2

# %%
topics_info[topics_info["Topic"] == ashp_topic]

# %%
ashp_docs = list(doc_info[doc_info["Topic"] == ashp_topic]["Document"])

# %%
ashp_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=10, vectorizer_model=vectorizer_model
)
ashp_topics, boiler_probs = ashp_topic_model.fit_transform(ashp_docs)

# %%
ashp_topics_, ashp_topics_info, ashp_doc_info = get_outputs_from_topic_model(
    ashp_topic_model, ashp_docs
)

# %%
ashp_topics_info

# %%
