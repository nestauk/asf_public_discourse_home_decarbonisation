"""
This script is used to perform topic analysis on year 2021, 2022 and 2023 data from the MSE forum.

We apply BERTopic model:
- to the titles of the threads in each year;
- We fix a random seed to ensure reproducibility of the results and use;

After running the topic model, we then visualise the topics and the documents associated with them, and we analyse the most common topics.

POTENTIAL PROBLEMS WITH THIS APPROACH:
- This doesn't take into account comments to older posts, just new posts in the forum!
- Not much testing has been done on this specific result!
- These are not really comparable

Another possible approach: run BERTopic on all data and then look at the proportion of different topics in each year.

To transform this script into a jupyter notebook, we can use the following commands:
- `jupytext --to notebook topic_analysis_mse_2021_vs_2022_vs_2023.py --output topic_analysis_mse_2021_vs_2022_vs_2023.ipynb`
- If the correct kernel does not come up (`asf_public_discourse_home_decarbonisation`), please run the following in your terminal:  `python -m ipykernel install --user --name=asf_public_discourse_home_decarbonisation`
"""

# %%
from bertopic import BERTopic
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
    create_bar_plot_most_common_topics,
)


# %% [markdown]
# ## Loading & filtering data

# %%
mse_data = get_mse_data(
    category="all", collection_date="2023_11_15", processing_level="raw"
)

# %%
# Replacing abbreviations
mse_data["title"] = mse_data["title"].apply(
    lambda x: x.lower()
    .replace("ashps", "ashp")
    .replace("ashp", "air source heat pump")
    .replace("gshps", "gshp")
    .replace("gshp", "ground source heat pump")
    .replace("hps", "hp")
    .replace("hp", "heat pump")
    .replace("ufh", "under floor heating")
)
mse_data["text"] = mse_data["text"].apply(
    lambda x: x.lower()
    .replace("ashps", "ashp")
    .replace("ashp", "air source heat pump")
    .replace("gshps", "gshp")
    .replace("gshp", "ground source heat pump")
    .replace("hps", "hp")
    .replace("hp", "heat pump")
    .replace("ufh", "under floor heating")
)

# %%
mse_data["category"].unique()

# %%


# %%
mse_data["datetime"] = mse_data["datetime"].apply(
    lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S+00:00")
)
mse_data["year"] = mse_data["datetime"].apply(lambda x: x.year)

# %% [markdown]
# ## MSE titles in 2023

# %%
data_2023 = mse_data[mse_data["year"] == 2023]

# %%
docs = data_2023[data_2023["is_original_post"] == 1].drop_duplicates("title")["title"]

# %%
len(docs)

# %%

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
topic_model = BERTopic(umap_model=umap_model)
topics, probs = topic_model.fit_transform(docs)


# %%
topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
create_bar_plot_most_common_topics(topics_info=topics_info, top_n_topics=16)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_barchart(
    topics=range(15), title="Topic word scores - MSE Top topics in 2023"
)

# %%
topic_model.visualize_documents(
    docs=list(docs),
    width=1200,
    height=750,
    topics=range(15),
    title="MSE Top 15 topics in 2023",
)

# %%
topic_model.visualize_hierarchy()

# %%


# %% [markdown]
# ## MSE titles in 2022

# %%
data_2022 = mse_data[mse_data["year"] == 2022]

# %%
docs = data_2022[data_2022["is_original_post"] == 1].drop_duplicates("title")["title"]

# %%
len(docs)

# %%

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
topic_model = BERTopic(umap_model=umap_model)
topics, probs = topic_model.fit_transform(docs)


# %%
topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
create_bar_plot_most_common_topics(topics_info=topics_info, top_n_topics=16)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_barchart(
    topics=range(15), title="Topic word scores - MSE Top topics in 2022"
)

# %%
topic_model.visualize_documents(
    docs=list(docs),
    width=1200,
    height=750,
    topics=range(15),
    title="MSE Top 15 topics in 2022",
)

# %%
topic_model.visualize_hierarchy()

# %%


# %% [markdown]
# ## MSE titles in 2021

# %%
data_2021 = mse_data[mse_data["year"] == 2021]

# %%
docs = data_2021[data_2021["is_original_post"] == 1].drop_duplicates("title")["title"]

# %%
len(docs)

# %%

umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
topic_model = BERTopic(umap_model=umap_model)
topics, probs = topic_model.fit_transform(docs)


# %%
topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
create_bar_plot_most_common_topics(topics_info=topics_info, top_n_topics=16)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_barchart(
    topics=range(15), title="Topic word scores - MSE Top topics in 2021"
)

# %%
topic_model.visualize_documents(
    docs=list(docs),
    width=1200,
    height=750,
    topics=range(15),
    title="MSE Top 15 topics in 2021",
)

# %%
topic_model.visualize_hierarchy()
