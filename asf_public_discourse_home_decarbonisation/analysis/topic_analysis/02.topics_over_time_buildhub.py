# %% [markdown]
# # Identifying topics of conversation and how they change over time in Buildhub
#
# What the code in this notebook does:
#
# - Loads data from Buildhub forum threads;
# - Cleans the data by removing URLs, username patterns/introductory patterns and by processing abbreviations (replacing them with their full forum). Note that no additional data cleaning is needed as BERTopic is able to handle the data as is;
# - Uses BERTopic to identify topics of conversation in the forum threads;
#     - Optionally assigning outliers to their closest topic;
# - Plots the topics of conversation in the forum threads;
# - Computing growth rates of topics from 2018 to 2022 (we're then plotting the growth rates in Flourish);
# - Plots the changes in topics of conversation over time;
# - We use BERTopic to recluster certain topics of conversation - the "heat pump" and "boilers" topic.
#
#
# Note that the growth plots live in Flourish.
#
#
# To convert this script to a jupyter notebook and explore results:
# ```
# pip install jupytext
# jupytext --to notebook 02.topics_over_time_buildhub.py
# ```

# %%
# package imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import openpyxl  # not "used" but needed to save excel files
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
    create_bar_plot_most_common_topics,
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

# %% [markdown]
# ## Loading & filtering data

# %%
bh_data = get_bh_data(category="all")

# %%
len(bh_data)

# %%
# Defining the text column we're using for clustering documents
# "title" or "text" or "whole_text" (a combination of "title" and "text")
text_col = "title"  # or "text" or "whole_text"


# If you want to assign each document to a topic, set this to True and you'll have no outliers
reduce_outliers_to_zero = False

# %% [markdown]
# ## Data cleaning

# %%
if text_col != "title":
    bh_data["text"] = bh_data["text"].apply(remove_urls)
    bh_data["text"] = bh_data["text"].apply(remove_username_pattern)
    bh_data["text"] = bh_data["text"].apply(remove_introduction_patterns)

# %%
if text_col == "whole_text":
    bh_data["whole_text"] = bh_data.apply(
        lambda x: (
            x["title"] + " " + x["text"]
            if ends_with_punctuation(x["title"])
            else x["title"] + ". " + x["text"]
        ),
        axis=1,
    )

# %%
# Processing abbreviations such as "ashp" to "air source heat pump"
bh_data[text_col] = bh_data[text_col].apply(process_abbreviations)

# %%
bh_data.head()

# %%
bh_data.rename(columns={"date": "datetime"}, inplace=True)
bh_data["datetime"] = pd.to_datetime(bh_data["datetime"])
bh_data["date"] = bh_data["datetime"].dt.date
bh_data["year"] = bh_data["datetime"].dt.year

# %% [markdown]
# ## 1. Topic identification

# %% [markdown]
# ### 1.1. Defining text variable used for topic analysis
#
# - At this point we don't to cluster replies to we always set "is_original_post" to 1 (True).
# - We always deduplicate the text column in use as duplicates shouldn't be used when applying BERTopic.

# %%
if text_col == "title":
    # titles
    docs = bh_data[bh_data["is_original_post"] == 1].drop_duplicates("title")["title"]

    dates = list(
        bh_data[bh_data["is_original_post"] == 1].drop_duplicates("title")["date"]
    )
elif text_col == "whole_text":
    # titles + text (we don't care about replies!)
    docs = bh_data[bh_data["is_original_post"] == 1].drop_duplicates("whole_text")[
        "whole_text"
    ]

    dates = list(
        bh_data[bh_data["is_original_post"] == 1].drop_duplicates("whole_text")["date"]
    )
else:
    # text
    docs = bh_data[bh_data["is_original_post"] == 1].drop_duplicates("text")["text"]

    dates = list(
        bh_data[bh_data["is_original_post"] == 1].drop_duplicates("text")["date"]
    )

# %%
len(docs)

# %% [markdown]
# ### 1.2. Topic model definition
#
# The arguments below should be changed for each specific dataset:
# - We use BERTopic to identify topics of conversation in the forum threads.
# - We're using `CountVectorizer(stop_words="english")` as the vectorizer to have meaningful representations, without stopwords;
#     - Alternatively you can use OpenAI to generate a custom topic label for each topic;
# - We're setting `random_state=42` in UMAP for reproducibility, not changing any of the remaining default parameters used in BERTopic UMAP;
# - We're setting `nr_topics=30` as the number of topics to identify so that we have a manageable number of topics.

# %%
vectorizer_model = CountVectorizer(stop_words="english")
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
if reduce_outliers_to_zero:
    topic_model = BERTopic(
        umap_model=umap_model,
        min_topic_size=40,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )
else:
    topic_model = BERTopic(
        umap_model=umap_model, min_topic_size=40, vectorizer_model=vectorizer_model
    )

topics, probs = topic_model.fit_transform(docs)

# %%
topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
# shows information about each topic
# notice that Topic -1 is the outliers topic
topics_info

# %% [markdown]
# ### 1.4. Looking for topics containing specific keywords
#
# - We're looking for topics containing specific keywords to identify topics of conversation related to:
#     - "heat pumps"
#     - "boilers"
#     - Retrofitting and low carbon heating.

# %%
topics_info[topics_info["Name"].str.contains("pump")]

# %%
topics_info[topics_info["Name"].str.contains("radiator")]

# %%
topics_info[topics_info["Name"].str.contains("boiler")]

# %%
topics_info[topics_info["Name"].str.contains("tariff")]

# %%
doc_info[doc_info["Document"].str.contains("heat pump")].groupby("Topic").size()

# %% [markdown]
# ### 1.5. Optionally assining a topic to each document in the outliers cluster
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
topics_info[topics_info["Name"].str.contains("boiler")]

# %%
len(topics_info[topics_info["Name"].str.contains("boiler")]), topics_info[
    topics_info["Name"].str.contains("boiler")
]["Count"].sum()

# %%
topics_info[
    topics_info["Name"].str.contains("pump") & topics_info["Name"].str.contains("heat")
]

# %%
topics_info[
    topics_info["Name"].str.contains("pump") & topics_info["Name"].str.contains("heat")
]["Count"].sum()

# %%
topics_info[topics_info["Name"].str.contains("grant")]

# %%
len(topics_info)

# %%
doc_info[doc_info["Document"].str.contains("heat pump")].groupby("Topic").size()

# %%


# %% [markdown]
# ### 1.6. When doing topic analysis, we remove duplicates. Here we add them back on.
#
# - We add duplicates back on. These represent different forum posts with the same title!
# - Note that the order of topics might change!

# %%
data = bh_data[bh_data["is_original_post"] == 1][
    ["title", "datetime", "date", "year", "url"]
]

# %%
data = data.merge(doc_info, left_on="title", right_on="Document")

# %%
updated_topics_info = (
    data.groupby("Topic", as_index=False)[["url"]]
    .nunique()
    .rename(columns={"url": "updated_count"})
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
# ## 2. Changes in topics over time & growth in the past couple of years
#
# Here we compute:
#
# - Number of documents per topic and  per year;
# - Average number of documents in a rolling window of 3 years;
# - Growth rates for each topic between 2018 and 2022.

# %%
topics_per_year = data.groupby(["Topic", "year"])[["url"]].nunique()

# %%
topics_per_year.head()

# %%
topics_per_year = topics_per_year.pivot_table(
    index="year", columns="Topic", values="url"
)
topics_per_year.fillna(0, inplace=True)

# %%
topics_per_year

# %%
smoothed_avg_topics_per_year = topics_per_year.rolling(window=3, axis=0).mean()


# %%
def growth_rate(topic):
    return (
        (
            smoothed_avg_topics_per_year.loc[2022][topic]
            - smoothed_avg_topics_per_year.loc[2018][topic]
        )
        / smoothed_avg_topics_per_year.loc[2018][topic]
        * 100
    )


growth_rate_2018_2022 = smoothed_avg_topics_per_year.columns.to_series().apply(
    growth_rate
)

# %%
growth_rate_2018_2022

# %%
updated_topics_info["growth_rate"] = updated_topics_info["Topic"].apply(
    lambda x: growth_rate_2018_2022[x]
)

# %%
# updated_topics_info[["Name", "growth_rate", "updated_count"]].to_excel("bh_topics_growth_rate.xlsx", index=False)

# %% [markdown]
# ## 3. Visualising the top topics of conversation & changes in the past couple of years

# %% [markdown]
# ### 3.1. Top topics of conversation

# %%
create_bar_plot_most_common_topics(topics_info=topics_info, top_n_topics=16)


# %%
topics_info_excel = pd.read_excel(
    f"s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/topic_analysis/bh_topics_growth_rate.xlsx"
)

# %%
topics_info_excel

# %%


# %% [markdown]
# ## 4. Using BERTopic to look at changes over time
#
# BERTopic offers the functionality to automatically identify changes in topics of time.

# %%
bh_data["date_str"] = bh_data["datetime"].dt.strftime("%Y-%m-%d")
if text_col == "title":
    # titles
    dates = list(
        bh_data[bh_data["is_original_post"] == 1].drop_duplicates("title")["date_str"]
    )
elif text_col == "whole_text":
    # titles + text (we don't care about replies!)
    dates = list(
        bh_data[bh_data["is_original_post"] == 1].drop_duplicates("whole_text")[
            "date_str"
        ]
    )

else:
    # text
    dates = list(
        bh_data[bh_data["is_original_post"] == 1].drop_duplicates("text")["date_str"]
    )


# %%
topics_over_time = topic_model.topics_over_time(docs, dates, nr_bins=8)

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
# ### 5.1. Exploring conversations about heat pumps

# %%
topics_info

# %%
ashp_topic = 0

# %%
topics_info[topics_info["Topic"] == ashp_topic]

# %%
ashp_docs = list(doc_info[doc_info["Topic"] == ashp_topic]["Document"])

# %%
hp_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=10, vectorizer_model=vectorizer_model
)
hp_topics, hp_probs = hp_topic_model.fit_transform(ashp_docs)

# %%
hp_topics_, hp_topics_info, hp_doc_info = get_outputs_from_topic_model(
    hp_topic_model, ashp_docs
)

# %%
hp_topics_info

# %%
