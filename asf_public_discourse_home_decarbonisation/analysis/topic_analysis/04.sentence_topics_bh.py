#!/usr/bin/env python
# coding: utf-8

# # Identifying topics of conversation from sentences and how they change over time in Money Saving Expert
#
# What the code in this notebook does:
#
# - Loads data from BuildHub (BH) forum threads;
# - Cleans the data by removing URLs, username patterns/introductory patterns and by processing abbreviations (replacing them with their full forum). Note that no additional data cleaning is needed as BERTopic is able to handle the data as is;
# - Break text into **sentences**;
# - Uses BERTopic to identify topics of conversation in the **sentences**;
#     - Optionally assigning outliers to their closest topic;
# - Plots the topics of conversation in the forum threads;
# - Computing growth rates of topics from 2020 to 2022;
# - Plots the changes in topics of conversation over time;
# - We use BERTopic to recluster certain topics of conversation
#
#
# To convert this script to a jupyter notebook and explore results:
# ```
# pip install jupytext
# jupytext --to notebook 04.sentence_topics_bh.py
# ```

# In[19]:


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


# In[20]:


# setting plotting styles
set_plotting_styles()


# In[21]:


# If you want to assign each document to a topic, set this to True and you'll have no outliers
reduce_outliers_to_zero = False
only_include_sentences_mentioning_hp = False


# ## 1. Loading, data coleaning and & filtering data

# ### 1.1 Load Data

# In[22]:


bh_data = get_bh_data(category="all")


# In[23]:


len(bh_data)


# ### 1.2. Data Cleaning

# In[25]:


bh_data["text"] = bh_data["text"].astype(str)
bh_data["text"] = bh_data["text"].apply(remove_urls)
bh_data["text"] = bh_data["text"].apply(remove_username_pattern)
bh_data["text"] = bh_data["text"].apply(remove_introduction_patterns)


#

# In[26]:


bh_data["title"] = bh_data.apply(
    lambda x: "" if x["is_original_post"] == 0 else x["title"], axis=1
)
bh_data["whole_text"] = bh_data.apply(
    lambda x: (
        x["title"] + " " + x["text"]
        if (ends_with_punctuation(x["title"]) or x["is_original_post"] == 0)
        else x["title"] + ". " + x["text"]
    ),
    axis=1,
)


# In[27]:


# Processing abbreviations such as "ashp" to "air source heat pump"
bh_data["whole_text"] = bh_data["whole_text"].apply(process_abbreviations)


# ### 1.3 Adding date/time variable

# In[30]:


bh_data.rename(columns={"date": "datetime"}, inplace=True)
bh_data["datetime"] = pd.to_datetime(bh_data["datetime"])
bh_data["date"] = bh_data["datetime"].dt.date
bh_data["year"] = bh_data["datetime"].dt.year


# ### 1.4 Focusing on posts from the past 5 years

# In[31]:


bh_data = bh_data[bh_data["year"] >= 2018]


# In[32]:


len(bh_data)


# ### 1.5 Focusing on conversations mentioning HPs

# In[34]:


print(bh_data.head())


# In[41]:


def create_id(title, title_to_id):
    if title.strip() == "":
        return ""
    if title not in title_to_id:
        title_to_id[title] = len(title_to_id) + 1
    return f"discussion_{title_to_id[title]}_{title}"


# Create a dictionary to map titles to IDs
title_to_id = {}

# Apply the function to each row
bh_data["id"] = [create_id(title, title_to_id) for title in bh_data["title"]]
bh_data.head()


# In[42]:


ids_to_keep = bh_data[
    (bh_data["whole_text"].str.contains("heat pump", case=False))
    & (bh_data["is_original_post"] == 1)
]["id"].unique()


# In[43]:


bh_data = bh_data[bh_data["id"].isin(ids_to_keep)]


# In[44]:


len(bh_data)


# ### 1.6. Adding space after punctuation ("?", ".", "!"), so that sentences are correctly split
#
# Sentence tokenizers don't work well with punctuation marks that don't have a space after them. We add a space after punctuation marks to ensure that sentences are correctly split.

# In[46]:


# adding space after punctuation
bh_data["whole_text"] = bh_data["whole_text"].apply(
    lambda t: re.sub(r"([.!?])([A-Za-z])", r"\1 \2", t)
)
len(bh_data)


# ### 1.7. Transforming text into sentences and striping white sapces

# In[49]:


bh_data["sentences"] = bh_data["whole_text"].apply(sent_tokenize)
print(bh_data["sentences"].head())


# In[50]:


sentences_data = bh_data.explode("sentences")
sentences_data["sentences"] = sentences_data["sentences"].astype(str)
sentences_data["sentences"] = sentences_data["sentences"].str.strip()


# In[51]:


# Looking to identify any other patterns we might want to remove
sentences_data["sentences"].str.lower().value_counts().head(10)


# In[52]:


# this code is tokenising each sentence into individual words, and then removing punctuation tokens from each list of tokens.
sentences_data["tokens"] = sentences_data["sentences"].apply(word_tokenize)
sentences_data["non_punctuation_tokens"] = sentences_data["tokens"].apply(
    lambda x: [token for token in x if token not in string.punctuation]
)


# In[53]:


sentences_data["n_tokens"] = sentences_data["non_punctuation_tokens"].apply(len)


# In[54]:


len(sentences_data[sentences_data["n_tokens"] > 5]) / len(sentences_data)


# In[55]:


sentences_data = sentences_data[sentences_data["n_tokens"] > 5]


# ### 1.9. Removing very common sentences

# In[57]:


# Looking to identify any other patterns we might want to remove
sentences_data["sentences"].str.lower().value_counts().head(20)
len(sentences_data["sentences"])


# In[58]:


sentences_data = sentences_data[
    ~(
        sentences_data["sentences"].str.contains("thank", case=False)
        | sentences_data["sentences"].str.contains("happy to help", case=False)
        | sentences_data["sentences"].str.contains("kind wishes", case=False)
        | sentences_data["sentences"].str.contains("kind regards", case=False)
    )
]


# In[60]:


# Looking to identify any other patterns we might want to remove
sentences_data["sentences"].str.lower().value_counts().head(10)


# ## 2. Topic identification

# ### 2.1. Defining text variable used for topic analysis
#
# - We're also including sentences coming from replies;
# - We always deduplicate tthe sentences in use as duplicates shouldn't be used when applying BERTopic.

# In[61]:


len(sentences_data)


# In[63]:


docs = sentences_data.drop_duplicates("sentences")["sentences"]
dates = list(sentences_data.drop_duplicates("sentences")["date"])


# In[64]:


len(docs)


# ### 2.2. Topic model definition
#
# The arguments below should be changed for each specific dataset:
# - We use BERTopic to identify topics of conversation in the forum threads.
# - We're using `CountVectorizer(stop_words="english")` as the vectorizer to have meaningful representations, without stopwords;
#     - Alternatively you can use OpenAI to generate a custom topic label for each topic;
# - We're setting `random_state=42` in UMAP for reproducibility, not changing any of the remaining default parameters used in BERTopic UMAP;
# - We're setting `nr_topics=100` as the number of topics to identify so that we have a manageable number of topics.

# In[71]:


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
        umap_model=umap_model, min_topic_size=30, vectorizer_model=vectorizer_model
    )

topics, probs = topic_model.fit_transform(docs)


# In[72]:


topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)


# In[79]:


# shows information about each topic
# notice that Topic -1 is the outliers topic
topics_info.head(20)


# ### 2.3. Exploring the presence of HP sentences in the outlier cluster

# In[74]:


len(
    doc_info[
        (doc_info["Topic"] == -1) & (doc_info["Document"].str.contains("heat pump"))
    ]
) / len(doc_info[doc_info["Topic"] == -1]) * 100


# In[75]:


doc_info[(doc_info["Topic"] == -1) & (doc_info["Document"].str.contains("heat pump"))][
    "Document"
].values


# In[76]:


doc_info[(doc_info["Topic"] == -1) & ~(doc_info["Document"].str.contains("heat pump"))][
    "Document"
].values[:20]


# ### 2.4. Looking for topics containing specific keywords
#
# - We're looking for topics containing specific keywords to identify topics of conversation related to:
#     - "heat pumps"
#     - "boilers"
#     - "noise"
#     - "efficiency"
#     - "cost

# In[77]:


topics_info[
    topics_info["Name"].str.contains("heat") | topics_info["Name"].str.contains("pump")
]


# In[86]:


topics_info[topics_info["Name"].str.contains("rad")]


# In[81]:


topics_info[topics_info["Name"].str.contains("noise")]


# In[82]:


topics_info[topics_info["Name"].str.contains("ground")]


# In[83]:


topics_info[
    topics_info["Name"].str.contains("cost")
    | topics_info["Name"].str.contains("price")
    | topics_info["Name"].str.contains("expensive")
    | topics_info["Name"].str.contains("cheap")
]


# ### 2.6. When doing topic analysis, we remove duplicates. Here we add them back on.
#
# - We add duplicates back on. These represent different forum posts with the same title!
# - Note that the order of topics might change!

#

# In[87]:


sentences_data


# In[88]:


data = sentences_data[["sentences", "datetime", "date", "year", "id"]]


# In[89]:


data = data.merge(doc_info, left_on="sentences", right_on="Document")


# In[91]:


updated_topics_info = (
    data.groupby("Topic", as_index=False)[["id"]]
    .count()
    .rename(columns={"id": "updated_count"})
    .merge(topics_info, on="Topic")
)


# In[92]:


updated_topics_info["updated_%"] = (
    updated_topics_info["updated_count"]
    / sum(updated_topics_info["updated_count"])
    * 100
)


# In[93]:


updated_topics_info = updated_topics_info.sort_values(
    "updated_count", ascending=False
).reset_index(drop=True)


# In[94]:


updated_topics_info


# ## 3. Changes in topics over time & growth in the past couple of years
#
# Here we compute:
#
# - Number of documents per topic and  per year;
# - Average number of documents in a rolling window of 3 years;
# - Growth rates for each topic between 2020 and 2022.

# In[96]:


topics_per_year = data.groupby(["Topic", "year"])[["sentences"]].count()


# In[97]:


topics_per_year


# In[98]:


topics_per_year = topics_per_year.pivot_table(
    index="year", columns="Topic", values="sentences"
)
topics_per_year.fillna(0, inplace=True)


# In[99]:


topics_per_year


# In[100]:


smoothed_avg_topics_per_year = topics_per_year.rolling(window=3, axis=0).mean()


# In[101]:


smoothed_avg_topics_per_year


# In[102]:


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


# In[103]:


updated_topics_info["growth_rate"] = updated_topics_info["Topic"].apply(
    lambda x: growth_rate_2018_2022[x]
)


# In[104]:


updated_topics_info.sort_values("growth_rate", ascending=False).head(20)


# ## 4. Using BERTopic to look at changes over time
#
# BERTopic offers the functionality to automatically identify changes in topics of time.

# In[105]:


data["date_str"] = data["datetime"].dt.strftime("%Y-%m-%d")
dates = list(data.drop_duplicates("sentences")["date_str"])


# In[106]:


topics_over_time = topic_model.topics_over_time(docs, dates, nr_bins=5)


# In[107]:


topic_model.visualize_topics_over_time(
    topics_over_time, topics=range(10), width=900, height=450
)


# ## 5. Exploring subtopics of conversation
#
# Here we re-cluster certain topics of conversation to identify subtopics.

# ### 5.1. Exploring conversations about noise

# In[108]:


topics_info[topics_info["Name"].str.contains("noise")]


# In[110]:


noise_topic = 12


# In[111]:


topics_info[topics_info["Topic"] == noise_topic]


# In[126]:


noise_docs = list(doc_info[doc_info["Topic"] == noise_topic]["Document"])
print(noise_docs)


# In[114]:


noise_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=10, vectorizer_model=vectorizer_model
)
noise_topics, noise_probs = noise_topic_model.fit_transform(noise_docs)


# In[115]:


noise_topics_, noise_topics_info, noise_doc_info = get_outputs_from_topic_model(
    noise_topic_model, noise_docs
)


# In[116]:


noise_topics_info


# ### 5.2. Exploring conversations about ashps

# In[117]:


topics_info[topics_info["Name"].str.contains("pump")]


# In[119]:


ashp_topic = 2


# In[120]:


topics_info[topics_info["Topic"] == ashp_topic]


# In[121]:


ashp_docs = list(doc_info[doc_info["Topic"] == ashp_topic]["Document"])
ashp_topic_model = BERTopic(
    umap_model=umap_model, min_topic_size=10, vectorizer_model=vectorizer_model
)
ashp_topics, boiler_probs = ashp_topic_model.fit_transform(ashp_docs)


# In[123]:


ashp_topics_, ashp_topics_info, ashp_doc_info = get_outputs_from_topic_model(
    ashp_topic_model, ashp_docs
)


# In[124]:


ashp_topics_info


# ### 6 Explore sentiment analysis for BH data.

# In[125]:


from transformers import pipeline

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
result = sentiment_task("Covid cases are increasing fast!")
print(result)


# In[139]:


noise_sentiment_result_data = []
print(len(noise_docs))
for sentence in noise_docs:
    sentiment_result = sentiment_task(sentence)
    noise_sentiment_result_data.append(
        {
            "sentence": sentence,
            "label": sentiment_result[0]["label"],
            "score": sentiment_result[0]["score"],
        }
    )

# Convert the list to a DataFrame
noise_sentiment_df = pd.DataFrame(noise_sentiment_result_data)
noise_sentiment_df.to_csv("noise_sentiment.csv", index=False)


# In[151]:


# Read in the HUMAN LABELED CSV file
df = pd.read_csv("noise_sentiment_human_label.csv")

# Get the first 40 rows
df = df.head(40)
print(df.head())

# Create a confusion matrix
confusion_matrix = pd.crosstab(
    df["label"], df["human_label"], rownames=["Predicted"], colnames=["Actual"]
)

print(confusion_matrix)

# Calculate the number of correct predictions
correct_predictions = (df["label"] == df["human_label"]).sum()

# Calculate the total number of predictions
total_predictions = len(df)

# Calculate the percentage of correct predictions
accuracy = correct_predictions / total_predictions * 100

print(f"Accuracy: {accuracy}%")
# Print unique categories in the 'human_label' column
print("Unique categories in 'human_label':", df["human_label"].unique())


# In[152]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a confusion matrix
confusion_matrix = pd.crosstab(
    df["label"], df["human_label"], rownames=["Predicted"], colnames=["Actual"]
)

# Create a heatmap from the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

plt.show()


# In[153]:


ashp_sentiment_result_data = []
print(len(ashp_docs))
for sentence in ashp_docs:
    sentiment_result = sentiment_task(sentence)
    ashp_sentiment_result_data.append(
        {
            "sentence": sentence,
            "label": sentiment_result[0]["label"],
            "score": sentiment_result[0]["score"],
        }
    )

# Convert the list to a DataFrame
ashp_sentiment_df = pd.DataFrame(ashp_sentiment_result_data)
ashp_sentiment_df.to_csv("ashp_sentiment.csv", index=False)


# In[156]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
ashp_file_path = "ashp_sentiment.csv"
noise_file_path = "noise_sentiment.csv"

ashp_data = pd.read_csv(ashp_file_path)
noise_data = pd.read_csv(noise_file_path)

# Determine the bin edges based on the combined range of scores
bin_edges = pd.cut(
    pd.concat([ashp_data["score"], noise_data["score"]]), bins=20, retbins=True
)[1]

# Create overlapping normalized histograms with the same bin sizes and range
plt.figure(figsize=(10, 6))

# Normalized histogram for Air Source Heat Pump sentiment scores
plt.hist(
    ashp_data["score"],
    bins=bin_edges,
    alpha=0.5,
    label="Air Source Heat Pump",
    color="blue",
    edgecolor="black",
    density=True,
)

# Normalized histogram for Noise sentiment scores
plt.hist(
    noise_data["score"],
    bins=bin_edges,
    alpha=0.5,
    label="Noise",
    color="green",
    edgecolor="black",
    density=True,
)

# Titles and labels
plt.title("Normalised Overlapping Histograms of Sentiment Scores")
plt.xlabel("Sentiment Score")
plt.ylabel("Density")
plt.legend(loc="upper left")

# Show plot
plt.tight_layout()
plt.show()


# In[158]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the datasets
ashp_file_path = "ashp_sentiment.csv"
noise_file_path = "noise_sentiment.csv"

ashp_data = pd.read_csv(ashp_file_path)
noise_data = pd.read_csv(noise_file_path)

# Calculate the counts of each sentiment label for both clusters
ashp_label_counts = ashp_data["label"].value_counts()
noise_label_counts = noise_data["label"].value_counts()

# Ensure both have the same index for consistent comparison
all_labels = sorted(set(ashp_label_counts.index).union(set(noise_label_counts.index)))
ashp_label_counts = ashp_label_counts.reindex(all_labels, fill_value=0)
noise_label_counts = noise_label_counts.reindex(all_labels, fill_value=0)

# Normalize the counts
ashp_normalized = ashp_label_counts / ashp_label_counts.sum()
noise_normalized = noise_label_counts / noise_label_counts.sum()

# Sample sizes
ashp_sample_size = len(ashp_data)
noise_sample_size = len(noise_data)

# Create the normalized side-by-side bar chart
fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.35
index = np.arange(len(all_labels))

ax.bar(
    index,
    ashp_normalized,
    bar_width,
    label=f"Air Source Heat Pump (n={ashp_sample_size})",
    color="blue",
    edgecolor="black",
)
ax.bar(
    index + bar_width,
    noise_normalized,
    bar_width,
    label=f"Noise (n={noise_sample_size})",
    color="green",
    edgecolor="black",
)

# Titles and labels
ax.set_title("Normalised Side-by-Side Bar Chart of Sentiment Labels")
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(all_labels)
ax.set_ylabel("Proportion")
ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()
