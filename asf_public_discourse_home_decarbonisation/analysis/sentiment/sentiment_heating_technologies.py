# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    set_plotting_styles,
    NESTA_COLOURS,
)

set_plotting_styles()
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment_technologies import (
    prepping_data_for_topic_analysis,
)

# %%
mse_data = get_mse_data(category="all", collection_date="2024_06_03")

hp_sentences_data = prepping_data_for_topic_analysis(
    mse_data, "heat pump", "2018-01-01", "2024-05-22"
)

solar_sentences_data = prepping_data_for_topic_analysis(
    mse_data, "solar panel", "2018-01-01", "2024-05-22"
)

boilers_sentences_data = prepping_data_for_topic_analysis(
    mse_data, "boiler", "2018-01-01", "2024-05-22"
)

# %%
hp_sentences_data[["sentences"]]

# %%
hp_sentences_data["count"] = 1
solar_sentences_data["count"] = 1
boilers_sentences_data["count"] = 1

# %%
hp_counts = hp_sentences_data.groupby("sentences")[["count"]].sum()
solar_counts = solar_sentences_data.groupby("sentences")[["count"]].sum()
boilers_counts = boilers_sentences_data.groupby("sentences")[["count"]].sum()


# %%
hp_counts.reset_index(inplace=True)
solar_counts.reset_index(inplace=True)
boilers_counts.reset_index(inplace=True)

# %%
sentiment_hps = pd.read_csv(
    "s3://asf-public-discourse-home-decarbonisation/data/mse/outputs/sentiment/desnz/mse_heat_pump_sentences_sentiment.csv"
)
sentiment_solar = pd.read_csv(
    "s3://asf-public-discourse-home-decarbonisation/data/mse/outputs/sentiment/desnz/mse_solar_panel_sentences_sentiment.csv"
)
sentiment_boilers = pd.read_csv(
    "s3://asf-public-discourse-home-decarbonisation/data/mse/outputs/sentiment/desnz/mse_boiler_sentences_sentiment.csv"
)

# %%
sentiment_hps = sentiment_hps.merge(hp_counts, left_on="text", right_on="sentences")
sentiment_solar = sentiment_solar.merge(
    solar_counts, left_on="text", right_on="sentences"
)
sentiment_boilers = sentiment_boilers.merge(
    boilers_counts, left_on="text", right_on="sentences"
)

# %%
len(sentiment_hps)

# %%
len(sentiment_solar)

# %%
len(sentiment_boilers)

# %%
sentiment_hps

# %%
sentiment_all = (
    sentiment_hps.groupby("sentiment")
    .sum()[["count"]]
    .rename(columns={"count": "Heat Pumps"})
)

# %%
sentiment_boilers

# %%
sentiment_all["Boilers"] = (
    sentiment_boilers.groupby("sentiment")
    .sum()[["count"]]
    .rename(columns={"count": "Boilers"})["Boilers"]
)

# %%
sentiment_all["Solar panels/PV"] = (
    sentiment_solar.groupby("sentiment")
    .sum()[["count"]]
    .rename(columns={"count": "Solar panels/PV"})["Solar panels/PV"]
)

# %%
sentiment_all

# %%
sentiment_all = sentiment_all.div(sentiment_all.sum(axis=0)) * 100

# %%
sentiment_all.sum()

# %%
sentiment_all = sentiment_all.T

# %%
sentiment_all.sort_values("negative", ascending=True, inplace=True)

# %%
sentiment_all.plot(
    kind="barh",
    stacked=True,
    color=[NESTA_COLOURS[4], NESTA_COLOURS[11], NESTA_COLOURS[1]],
    figsize=(6, 6),
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel("Percentage of sentences", fontsize=20)
plt.ylabel("")

# %%
sentiment_hps

# %%
counts = pd.DataFrame(index=["Heat Pumps", "Boilers", "Solar panels/PV"])
counts["counts"] = [
    sentiment_hps["count"].sum(),
    sentiment_boilers["count"].sum(),
    sentiment_solar["count"].sum(),
]

# %%
counts.sort_values("counts", ascending=True, inplace=True)

# %%
counts.plot(kind="barh", color=NESTA_COLOURS[0], figsize=(6, 6))
plt.legend().remove()
plt.xlabel("Number of sentences", fontsize=20)

# %%


# %% [markdown]
# ## proportion of negative sentiment over time

# %%
sent_over_time = hp_sentences_data[["sentences", "year"]].merge(
    sentiment_hps, left_on="sentences", right_on="sentences"
)
sent_over_time = (
    sent_over_time.groupby(["sentiment", "year"]).sum()[["count"]].unstack(level=0)
)
sent_over_time.columns = sent_over_time.columns.droplevel()
sent_over_time = sent_over_time.div(sent_over_time.sum(axis=1), axis=0)
sent_over_time = sent_over_time[["negative"]]
sent_over_time.rename(columns={"negative": "Heat Pumps"}, inplace=True)

# %%
sent_over_time_boilers = boilers_sentences_data[["sentences", "year"]].merge(
    sentiment_boilers, left_on="sentences", right_on="sentences"
)
sent_over_time_boilers = (
    sent_over_time_boilers.groupby(["sentiment", "year"])
    .sum()[["count"]]
    .unstack(level=0)
)
sent_over_time_boilers.columns = sent_over_time_boilers.columns.droplevel()
sent_over_time_boilers = sent_over_time_boilers.div(
    sent_over_time_boilers.sum(axis=1), axis=0
)
sent_over_time_boilers = sent_over_time_boilers[["negative"]]
sent_over_time_boilers.rename(columns={"negative": "Boilers"}, inplace=True)

sent_over_time["Boilers"] = sent_over_time_boilers["Boilers"]

# %%
sent_over_time_solar = solar_sentences_data[["sentences", "year"]].merge(
    sentiment_solar, left_on="sentences", right_on="sentences"
)
sent_over_time_solar = (
    sent_over_time_solar.groupby(["sentiment", "year"])
    .sum()[["count"]]
    .unstack(level=0)
)
sent_over_time_solar.columns = sent_over_time_solar.columns.droplevel()
sent_over_time_solar = sent_over_time_solar.div(
    sent_over_time_solar.sum(axis=1), axis=0
)
sent_over_time_solar = sent_over_time_solar[["negative"]]
sent_over_time_solar.rename(columns={"negative": "Solar panels/PV"}, inplace=True)

sent_over_time["Solar panels/PV"] = sent_over_time_solar["Solar panels/PV"]

# %%
sent_over_time = sent_over_time * 100

# %%


# %%
sent_over_time.loc[:2023].plot(
    kind="line",
    figsize=(6, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.title("Proportion of negative sentiment")
plt.ylim(0, 30)
plt.xlabel("")
plt.xticks(np.arange(2018, 2024, 1))
plt.legend(title="Technology")

# %%
