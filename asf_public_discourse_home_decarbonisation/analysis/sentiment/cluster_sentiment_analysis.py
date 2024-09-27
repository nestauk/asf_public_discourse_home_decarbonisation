# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    set_plotting_styles,
    NESTA_COLOURS,
)

set_plotting_styles()

# %%
source = "mse"

# %% [markdown]
# ### data imports

# %%
sentiment_data = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/{source}/outputs/sentiment/{source}_heat pump_sentence_topics_sentiment.csv"
)
sentences_data = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/{source}/outputs/topic_analysis/{source}_heat pump_sentences_data.csv"
)
doc_info = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/{source}/outputs/topic_analysis/{source}_heat pump_sentence_docs_info.csv"
)
topics_info = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/{source}/outputs/topic_analysis/{source}_heat pump_sentence_topics_info.csv"
)

# %%
sentiment_data.head()

# %%
sentences_data.head()

# %%
doc_info.head()

# %%
topics_info.head()

# %%
len(sentiment_data[sentiment_data["score"] > 0.75]) / len(sentiment_data) * 100

# %%
sentiment_data

# %% [markdown]
# # Visualisations

# %%
doc_info = doc_info.merge(sentiment_data, left_on="Document", right_on="text")

# %%
topic_sentimet = (
    doc_info.groupby(["Name", "sentiment"]).nunique()["Document"].unstack().fillna(0)
)

# %%
topic_sentimet["proportion_negative"] = topic_sentimet["negative"] / (
    topic_sentimet["negative"] + topic_sentimet["positive"] + topic_sentimet["neutral"]
)

# %%
topic_sentimet["ratio_to_negative"] = (
    topic_sentimet["positive"] + topic_sentimet["neutral"]
) / (topic_sentimet["negative"])

# %%
topic_sentimet

# %%
topic_sentimet.sort_values("proportion_negative", ascending=False)

# %%


# %% [markdown]
# ## Sentiment stacked plot

# %%
topic_sentimet = topic_sentimet[["negative", "neutral", "positive"]]
topic_sentimet = topic_sentimet.div(topic_sentimet.sum(axis=1), axis=0) * 100

# %%
topic_sentimet.sort_values("negative", ascending=False, inplace=True)

# %%
topic_sentimet.plot(
    kind="barh", stacked=True, color=["red", "grey", "green"], figsize=(9, 20)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")


# %%
topic_sentimet.sort_values("positive", ascending=False, inplace=True)

# %%
topic_sentimet.plot(
    kind="barh", stacked=True, color=["red", "grey", "green"], figsize=(9, 20)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")


# %% [markdown]
# ## Top topics and average sentiment

# %%
doc_info["sentiment_number"] = doc_info["sentiment"].map(
    {"negative": -1, "neutral": 0, "positive": 1}
)

# %%
sentiment_and_counts_per_topic = (
    doc_info.groupby("Name", as_index=False)
    .agg({"sentiment_number": "mean", "count": "sum"})
    .sort_values("sentiment_number")
)

# %%
import altair as alt

# %%
sentiment_and_counts_per_topic.sort_values("count", inplace=True)

# %%
a = (
    alt.Chart(sentiment_and_counts_per_topic)
    .mark_bar()
    .encode(
        y=alt.X("Name", title="Topic name"),
        x=alt.Y("count", title="Number of sentences"),
        color=alt.Color(
            "sentiment_number:Q",
            scale=alt.Scale(
                scheme="redyellowgreen", domain=[-1, 0.5]  # Set the color scale range
            ),
            title="Average sentiment",
        ),
    )
    .properties(
        title="Average sentiment per topic of conversation about heat pumps",
        width=600,  # Set the width of the chart
        height=400,  # Set the height of the chart
    )
    .interactive()
)
a

# %%
top_topics = (
    topics_info[~topics_info["Topic"].isin([-1, 0, 6])]
    .head(20)[["Name", "updated_count"]]
    .sort_values("updated_count", ascending=True)
)

# %% [markdown]
# ## Top topics - once I have the sentiment for each one we can have the plot above for the top 20 topics

# %%
top_topics.plot(
    kind="barh", x="Name", y="updated_count", figsize=(6, 9), color=NESTA_COLOURS[0]
)
plt.legend().remove()
plt.xlabel("Number of sentences")
plt.ylabel("Topic")
plt.title("Top 20 topics of conversation about heat pumps")

# %%


# %% [markdown]
# ## Sentiment over time
#
# Not sure we have enough data to be plotting this. need to check.

# %%
sentences_data["month"] = pd.to_datetime(sentences_data["date"]).dt.to_period("M")

# %%
sentiment_over_time = sentences_data[["sentences", "id", "month", "year"]]

# %%
sentiment_over_time = sentiment_over_time.merge(
    sentiment_data, left_on="sentences", right_on="text"
)

# %%
sentiment_over_time["sentiment_number"] = sentiment_over_time["sentiment"].map(
    {"negative": -1, "neutral": 0, "positive": 1}
)

# %%
sentiment_over_time = sentiment_over_time.merge(
    doc_info[["Document", "Topic", "Name"]], left_on="sentences", right_on="Document"
)

# %%
topics_sentiment_over_time = (
    sentiment_over_time.groupby(["Name", "year"])
    .agg({"sentiment_number": "mean"})
    .unstack()
)

# %%
topics_sentiment_over_time.columns = topics_sentiment_over_time.columns.droplevel()

# %%
topics_sentiment_over_time = topics_sentiment_over_time.T

# %%
topics_sentiment_over_time.plot(kind="line", figsize=(14, 8), color=NESTA_COLOURS)
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("Sentiment over time per topic of conversation about heat pumps")

# %%
topics_sentiment_over_time["all_topics"] = sentiment_over_time.groupby(["year"]).agg(
    {"sentiment_number": "mean"}
)["sentiment_number"]

# %%
topics_sentiment_over_time.columns

# %%
topics_sentiment_over_time[["11_noise_quiet_noisy_microgeneration", "all_topics"]].plot(
    kind="line", figsize=(14, 8), color=NESTA_COLOURS
)
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("Sentiment over time per topic of conversation about heat pumps")

# %%


# %%


# %% [markdown]
# ## Examples

# %%
examples = (
    sentiment_over_time[
        (sentiment_over_time["sentiment"] == "negative")
        & (sentiment_over_time["Document"].str.contains("heat pump"))
    ]
    .drop_duplicates("sentences")
    .sort_values("score", ascending=False)
    .drop_duplicates("Name")[["Document", "score", "Name"]]
)
for i in range(len(examples)):
    print(examples["Document"].iloc[i])
    print(examples["score"].iloc[i])
    print(examples["Name"].iloc[i])
    print("---")

# %%
examples = (
    sentiment_over_time[(sentiment_over_time["sentiment"] == "negative")]
    .drop_duplicates("sentences")
    .sort_values("score", ascending=False)
    .drop_duplicates("Name")[["Document", "score", "Name"]]
)
for i in range(len(examples)):
    print(examples["Document"].iloc[i])
    print(examples["score"].iloc[i])
    print(examples["Name"].iloc[i])
    print("---")

# %%


# %%


# %% [markdown]
# ## Number of sentences over time

# %%
counts_over_time = (
    sentiment_over_time.groupby(["Name", "month"])["text"].count().unstack().fillna(0).T
)

# %%
counts_over_time = counts_over_time.div(counts_over_time.sum(axis=1), axis=0)

# %%
counts_over_time[-11:].plot(kind="line", figsize=(14, 8), color=NESTA_COLOURS)
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.title("Proportion of conversations per topic")

# %%
