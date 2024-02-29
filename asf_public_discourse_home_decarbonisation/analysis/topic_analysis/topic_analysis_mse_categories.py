"""
This script is used to perform topic analysis on MSE category data;

We apply BERTopic model:
- to the titles of the threads in a specific MSE category
- We fix a random seed to ensure reproducibility of the results and use;

After running the topic model, we then visualise the topics and the documents associated with them, and we analyse the most common topics.

To transform this script into a jupyter notebook, we can use the following commands:
- `jupytext --to notebook topic_analysis_mse_categories.py --output topic_analysis_mse_categories.ipynb`
- If the correct kernel does not come up (`asf_public_discourse_home_decarbonisation`), please run the following in your terminal:  `python -m ipykernel install --user --name=asf_public_discourse_home_decarbonisation`

"""

# %%
from bertopic import BERTopic
from umap import UMAP
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
    create_bar_plot_most_common_topics,
)

# %% [markdown]
# ## Loading & filtering data

# %%
category = "is-this-quote-fair"

# %%
mse_data = get_mse_data(
    category=category, collection_date="2023_11_15", processing_level="raw"
)

# %%
mse_data.head()

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

# %%
mse_data["category"].unique()

# %% [markdown]
# ## MSE titles

# %%
mse_data["category"].unique()

# %%
category_data = mse_data[mse_data["category"] == category]

# %%
docs = category_data[category_data["is_original_post"] == 1].drop_duplicates("title")[
    "title"
]

# %%
len(docs)

# %%

umap_model = UMAP(
    n_neighbors=30, n_components=5, min_dist=0.0, metric="cosine", random_state=42
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
create_bar_plot_most_common_topics(
    topics_info=topics_info,
    top_n_topics=16,
    title=f"Most common topics in the\n `{category}` category",
)

# %%
topic_model.visualize_topics()

# %%
topic_model.visualize_barchart(
    topics=range(15), title=f"Topic word scores - category {category}"
)

# %%
topic_model.visualize_documents(
    docs=list(docs),
    width=1200,
    height=750,
    topics=range(16),
    title=f"Top 16 topics- category {category}",
)

# %%
topic_model.visualize_hierarchy(title=f"Topics hierarchy - category: {category}")
