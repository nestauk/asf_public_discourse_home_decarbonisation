# %% [markdown]
# # Identifying topics of conversation and how they change over time in Money Saving Expert
#
# What the code in this notebook does:
#
# - Loads data from Money Saving Expert (MSE) forum threads;
# - Cleans the data by removing URLs, username patterns/introductory patterns and by processing abbreviations (replacing them with their full forum). Note that no additional data cleaning is needed as BERTopic is able to handle the data as is;
# - Uses BERTopic to identify topics of conversation in the forum threads;
#     - Optionally assigning outliers to their closest topic;
# - Plots the topics of conversation in the forum threads;
# - Computing growth rates of topics from 2018 to 2022 (we're then plotting the growth rates in Flourish);
# - Plots the changes in topics of conversation over time;
# - We use BERTopic to recluster certain topics of conversation - the "heat pump" and "boilers" topic.
#
#
# To convert this script to a jupyter notebook and explore results:
# ```
# pip install jupytext
# jupytext --to notebook 01.topics_over_time_mse.py
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

# %% [markdown]
# ## Loading & filtering data

# %%
mse_data = get_mse_data(
    category="all", collection_date="2023_11_15", processing_level="raw"
)

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
    mse_data["text"] = mse_data["text"].apply(remove_urls)
    mse_data["text"] = mse_data["text"].apply(remove_username_pattern)
    mse_data["text"] = mse_data["text"].apply(remove_introduction_patterns)

# %%
if text_col == "whole_text":
    mse_data["whole_text"] = mse_data.apply(
        lambda x: (
            x["title"] + " " + x["text"]
            if ends_with_punctuation(x["title"])
            else x["title"] + ". " + x["text"]
        ),
        axis=1,
    )

# %%
# Processing abbreviations such as "ashp" to "air source heat pump"
mse_data[text_col] = mse_data[text_col].apply(process_abbreviations)

# %%
mse_data["datetime"] = pd.to_datetime(mse_data["datetime"])
mse_data["date"] = mse_data["datetime"].dt.date
mse_data["year"] = mse_data["datetime"].dt.year

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
    docs = mse_data[mse_data["is_original_post"] == 1].drop_duplicates("title")["title"]

    dates = list(
        mse_data[mse_data["is_original_post"] == 1].drop_duplicates("title")["date"]
    )
elif text_col == "whole_text":
    # titles + text (we don't care about replies!)
    docs = mse_data[mse_data["is_original_post"] == 1].drop_duplicates("whole_text")[
        "whole_text"
    ]

    dates = list(
        mse_data[mse_data["is_original_post"] == 1].drop_duplicates("whole_text")[
            "date"
        ]
    )
else:
    # text
    docs = mse_data[mse_data["is_original_post"] == 1].drop_duplicates("text")["text"]

    dates = list(
        mse_data[mse_data["is_original_post"] == 1].drop_duplicates("text")["date"]
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
        min_topic_size=300,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True,
    )
else:
    topic_model = BERTopic(
        umap_model=umap_model, min_topic_size=300, vectorizer_model=vectorizer_model
    )

topics, probs = topic_model.fit_transform(docs)

# %%
topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
# shows information about each topic
# notice that Topic -1 is the outliers topic
topics_info

# %% [markdown]
# ### 1.3. Looking for topics containing specific keywords
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

# %% [markdown]
# When a document contains the expression "heat pump", where is it clustered? Hopefully most of the documents are clustered in the same topic.

# %%
doc_info[doc_info["Document"].str.contains("heat pump")].groupby("Topic").size()

# %% [markdown]
# ### 1.4. Optionally assining a topic to each document in the outliers cluster
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

# %% [markdown]
# ### 1.5. When doing topic analysis, we remove duplicates. Here we add them back on.
#
# - We add duplicates back on. These represent different forum posts with the same title!
# - Note that the order of topics might change!

# %%
data = mse_data[mse_data["is_original_post"] == 1][
    [text_col, "datetime", "date", "year", "id"]
]

# %%
data = data.merge(doc_info, left_on=text_col, right_on="Document")

# %%
updated_topics_info = (
    data.groupby("Topic", as_index=False)[["id"]]
    .nunique()
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
# ## 2. Changes in topics over time & growth in the past couple of years
#
# Here we compute:
#
# - Number of documents per topic and  per year;
# - Average number of documents in a rolling window of 3 years;
# - Growth rates for each topic between 2018 and 2022.

# %%
topics_per_year = data.groupby(["Topic", "year"])[["id"]].nunique()

# %%
topics_per_year.head()

# %%
topics_per_year = topics_per_year.pivot_table(
    index="year", columns="Topic", values="id"
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
updated_topics_info["growth_rate"] = updated_topics_info["Topic"].apply(
    lambda x: growth_rate_2018_2022[x]
)

# %%
# updated_topics_info[["Name", "growth_rate", "updated_count"]].to_excel("mse_topics_growth_rate.xlsx", index=False)

# %% [markdown]
# ## 3. Visualising the top topics of conversation & changes in the past couple of years

# %% [markdown]
# ### 3.1. Top topics of conversation

# %%
# Outside of this notebook we've manually added a new name to each topic and a high-level category
updated_topics_info_excel = pd.read_excel(
    f"s3://asf-public-discourse-home-decarbonisation/data/mse/outputs/topic_analysis/mse_topics_growth_rate_updated.xlsx"
)

# %%
updated_topics_info_excel["Category"].unique()

# %%
# We create a colour mapping for each category
category_color_mapping = dict(
    zip(updated_topics_info_excel["Category"].unique(), NESTA_FLOURISH_COLOURS)
)

# %%
category_color_mapping

# %%
updated_topics_info_excel["colour"] = updated_topics_info_excel["Category"].map(
    category_color_mapping
)

# %%
updated_topics_info_excel

# %%
plt.figure(figsize=(10, 10))

filtered_topics_info_excel = updated_topics_info_excel.sort_values(
    "updated_count", ascending=True
)

names = filtered_topics_info_excel["Name new"]
categories = filtered_topics_info_excel["Category"]

values = filtered_topics_info_excel["updated_count"]
colors = filtered_topics_info_excel["colour"]

unique_pairs = set(zip(categories, colors))

# Create the horizontal bar plot
bars = plt.barh(names[-20:], values[-20:], color=colors[-20:])

# Create legend handles
legend_handles = [
    Patch(color=color, label=category) for category, color in unique_pairs
]

# Add legend
plt.legend(handles=legend_handles, title="Category")

# Add labels and title
plt.xlabel("Count")
# plt.ylabel('Category')
plt.title("Top 20 topics of conversation in the past 20 years")

# Show the plot
plt.show()

# %% [markdown]
# ### 3.2. Changes in number of new conversations with time

# %%
plt.figure(figsize=(12, 6))
(topics_per_year[27] / sum(topics_per_year[27])).plot(
    color="#9a1bbe", label="Heat Pumps"
)
(topics_per_year[4] / sum(topics_per_year[4])).plot(color="#0000ff", label="Boilers")
plt.xticks(range(2002, 2025, 2))
plt.title("Density of new posts")
plt.legend()

# %%


# %% [markdown]
# ## 4. Using BERTopic to look at changes over time
#
# BERTopic offers the functionality to automatically identify changes in topics of time.

# %%
mse_data["date_str"] = mse_data["datetime"].dt.strftime("%Y-%m-%d")
if text_col == "title":
    # titles
    dates = list(
        mse_data[mse_data["is_original_post"] == 1].drop_duplicates("title")["date_str"]
    )
elif text_col == "whole_text":
    # titles + text (we don't care about replies!)
    dates = list(
        mse_data[mse_data["is_original_post"] == 1].drop_duplicates("whole_text")[
            "date_str"
        ]
    )

else:
    # text
    dates = list(
        mse_data[mse_data["is_original_post"] == 1].drop_duplicates("text")["date_str"]
    )


# %%
topics_over_time = topic_model.topics_over_time(docs, dates, nr_bins=20)

# %%
topics_over_time

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=range(10), width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=[5, 6, 9, 10, 20, 30], width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=[0, 13, 15], width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=[11, 4], width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=range(30, 40), width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=range(30, 40), width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=[27, 29], width=900, height=450
)

# %%
topic_model.visualize_topics_over_time(
    topics_over_time, topics=[29], width=900, height=450
)

# %%


# %% [markdown]
# ## 5. Exploring subtopics of conversation
#
# Here we re-cluster certain topics of conversation to identify subtopics.

# %% [markdown]
# ### 5.1. Exploring conversations about heat pumps

# %%
hp_topic = 27

# %%
topics_info[topics_info["Topic"] == hp_topic]

# %%
hp_docs = list(doc_info[doc_info["Topic"] == hp_topic]["Document"])

# %%
hp_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=10, vectorizer_model=vectorizer_model
)
hp_topics, hp_probs = hp_topic_model.fit_transform(hp_docs)

# %%
hp_topics_, hp_topics_info, hp_doc_info = get_outputs_from_topic_model(
    hp_topic_model, hp_docs
)

# %%
hp_topics_info

# %% [markdown]
# ### 5.1. Exploring conversations about boilers

# %%
boilers_topic = 4

# %%
topics_info[topics_info["Topic"] == boilers_topic]

# %%
boiler_docs = list(doc_info[doc_info["Topic"] == boilers_topic]["Document"])

# %%
boiler_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=50, vectorizer_model=vectorizer_model
)
boiler_topics, boiler_probs = boiler_topic_model.fit_transform(boiler_docs)

# %%
boiler_topics_, boiler_topics_info, boiler_doc_info = get_outputs_from_topic_model(
    boiler_topic_model, boiler_docs
)

# %%
boiler_topics_info.head(30)

# %%
len(boiler_topics_info)

# %%
