"""
This script is used to perform topic analysis on the data from the MSE and Buildhub forums,
focusing on the mentions of "heat pump" related keywords.

We apply BERTopic model:
- to the titles of the threads where heat pumps are mentioned (either in the title, the text or in replies);
- We fix a random seed to ensure reproducibility of the results and use;
- We set specific BERTopic parameters after having evaluated the model with different configurations.

After running the topic model, we then visualise the topics and the documents associated with them, and we analyse the most common topics.

To transform this script into a jupyter notebook, we can use the following commands:
- `jupytext --to notebook topic_analysis_heat_pump_keywords.py`
- If the correct kernel does not come up (`asf_public_discourse_home_decarbonisation`), please run the following in your terminal:  `python -m ipykernel install --user --name=asf_public_discourse_home_decarbonisation`
"""

# %%
from bertopic import BERTopic
from umap import UMAP
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data
from asf_public_discourse_home_decarbonisation.config.keywords_dictionary import (
    keyword_dictionary,
)
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    create_bar_plot_most_common_topics,
    get_outputs_from_topic_model,
)

# %% [markdown]
# ## Loading & filtering data

# %%
mse_data = get_mse_data(
    category="all", collection_date="2023_11_15", processing_level="raw"
)

# %%
bh_data = get_bh_data(category="all", collection_date="24_02_01")

# %%
# Replacing abbreviations
mse_data["title"] = mse_data["title"].apply(
    lambda x: x.lower()
    .replace("ashps", "air source heat pumps")
    .replace("ashp", "air source heat pump")
    .replace("gshps", "ground source heat pumps")
    .replace("gshp", "ground source heat pump")
    .replace("hps", "heat pumps")
    .replace("hp", "heat pump")
    .replace("ufh", "under floor heating")
)
mse_data["text"] = mse_data["text"].apply(
    lambda x: x.lower()
    .replace("ashps", "air source heat pumps")
    .replace("ashp", "air source heat pump")
    .replace("gshps", "ground source heat pumps")
    .replace("gshp", "ground source heat pump")
    .replace("hps", "heat pumps")
    .replace("hp", "heat pump")
    .replace("ufh", "under floor heating")
)
# Replacing abbreviations
bh_data["title"] = (
    bh_data["title"]
    .astype(str)
    .apply(
        lambda x: x.lower()
        .replace("ashps", "air source heat pumps")
        .replace("ashp", "air source heat pump")
        .replace("gshps", "ground source heat pumps")
        .replace("gshp", "ground source heat pump")
        .replace("hps", "heat pumps")
        .replace("hp", "heat pump")
        .replace("ufh", "under floor heating")
    )
)
bh_data["text"] = (
    bh_data["text"]
    .astype(str)
    .apply(
        lambda x: x.lower()
        .replace("ashps", "air source heat pumps")
        .replace("ashp", "air source heat pump")
        .replace("gshps", "ground source heat pumps")
        .replace("gshp", "ground source heat pump")
        .replace("hps", "heat pumps")
        .replace("hp", "heat pump")
        .replace("ufh", "under floor heating")
    )
)

# %%
mse_data["category"].unique()

# %%
keywords = "heat_pump_keywords"

# %% [markdown]
# ## MSE titles mentioning "heat pump"

# %%
hp_data_mse = mse_data[
    mse_data["title"].str.contains("|".join(keyword_dictionary[keywords]), case=False)
    | mse_data["text"].str.contains("|".join(keyword_dictionary[keywords]), case=False)
]

# %%
docs = hp_data_mse[hp_data_mse["is_original_post"] == 1].drop_duplicates("title")[
    "title"
]

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
topics_info.head()

# %%
doc_info.head()

# %%
create_bar_plot_most_common_topics(topics_info=topics_info, top_n_topics=16)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_barchart(
    topics=range(15), title="Topic word scores - Heat Pump Mentions in MSE"
)

# %%
topic_model.visualize_documents(
    docs=list(docs),
    width=1200,
    height=750,
    topics=range(15),
    title="Top 15 topics in Heat Pump mentions in MSE",
)

# %%
topic_model.visualize_hierarchy()

# %%
topic_model.visualize_term_rank()


# %%


# %% [markdown]
# ## BuildHub titles mentioning "heat pump"

# %%
hp_data_bh = bh_data[
    bh_data["title"].str.contains("|".join(keyword_dictionary[keywords]), case=False)
    | bh_data["text"].str.contains("|".join(keyword_dictionary[keywords]), case=False)
]

# %%
docs = hp_data_bh[hp_data_bh["is_original_post"] == 1].drop_duplicates("title")["title"]

# %%
len(docs)

# %%
umap_model = UMAP(
    n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42
)
# vectorizer_model = CountVectorizer(stop_words="english")
# representation_model = KeyBERTInspired()

topic_model = BERTopic(umap_model=umap_model)
topics, probs = topic_model.fit_transform(docs)


# %%
topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

# %%
topics_info.head()

# %%
doc_info.head()

# %%
create_bar_plot_most_common_topics(topics_info=topics_info, top_n_topics=16)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_barchart(
    topics=range(15), title="Topic word scores - Heat Pump Mentions in Buildhub"
)

# %%
topic_model.visualize_documents(
    docs=list(docs),
    width=1200,
    height=750,
    topics=range(15),
    title="Top 15 topics in Heat Pump mentions in Buildhub",
)

# %%
topic_model.visualize_hierarchy()

# %%
topic_model.visualize_term_rank()
