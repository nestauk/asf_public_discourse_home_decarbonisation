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
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data

mse_data = get_mse_data(category="all", collection_date="2024_06_03")

# %%


# %%
len(mse_data)

# %%
mse_data["datetime"] = pd.to_datetime(mse_data["datetime"])

# %%
mse_data = mse_data[
    (mse_data["datetime"].dt.year < 2024)
    | ((mse_data["datetime"].dt.year == 2024) & (mse_data["datetime"].dt.month < 6))
    | (
        (mse_data["datetime"].dt.year == 2024)
        & (mse_data["datetime"].dt.month == 5)
        & (mse_data["datetime"].dt.day <= 22)
    )
]

# %%
len(mse_data)

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
doc_info[doc_info["Topic"] == 1]["Document"].iloc[7]

# %%
topics_info[topics_info["Name"].str.contains("dryer")]["Representative_Docs"].values

# %%
topics_info["Topic"].nunique(), topics_info["Topic"].max()

# %%
topics_info["Count"].sum()

# %% [markdown]
# ## Processing data

# %%
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_abbreviations,
    remove_urls,
)

# %%
mse_data["text"] = mse_data["text"].astype(str)
mse_data["title"] = mse_data["title"].astype(str)

mse_data["text"] = mse_data["text"].apply(remove_urls)

mse_data["text"] = mse_data["text"].apply(process_abbreviations)
mse_data["title"] = mse_data["title"].apply(process_abbreviations)


# %%
mse_data[
    mse_data["text"].str.contains("heat pump")
    | mse_data["title"].str.contains("heat pump")
]["id"].nunique()

# %%
mse_data["year"] = mse_data["datetime"].dt.year

# %%
aux = mse_data[mse_data["year"] >= 2018]

print(
    len(
        aux[
            aux["text"].str.contains("heat pump")
            | aux["title"].str.contains("heat pump")
        ]
    )
)
print(
    aux[aux["text"].str.contains("heat pump") | aux["title"].str.contains("heat pump")][
        "id"
    ].nunique()
)

# %% [markdown]
# ## Mentions over time

# %%
mentions_df = mse_data.copy()

key_terms = ["heat pump", "boiler"]  # , "solar"]
# Preprocess data
for term in key_terms:
    column_name = f"mentions_{term.replace(' ', '_')}"
    if term != "solar":
        mentions_df[column_name] = mentions_df["text"].str.contains(
            term, case=False
        ) | (
            mentions_df["title"].str.contains(term, case=False)
            & (mentions_df["is_original_post"] == 1)
        )
    else:
        terms = ["solar pv", "solar panel", "solar photovoltaic"]
        mentions_df[column_name] = mentions_df["text"].str.contains(
            "|".join(terms), case=False
        ) | (
            mentions_df["title"].str.contains("|".join(terms), case=False)
            & (mentions_df["is_original_post"] == 1)
        )

# %%
for col in key_terms:
    column_name = f"mentions_{col.replace(' ', '_')}"
    mentions_df[column_name] = mentions_df[column_name].astype(int)

# %%
mentions_df["year_month"] = mentions_df["datetime"].dt.to_period("M")
mentions_df["year"] = mentions_df["datetime"].dt.year

# %%
monthly_mentions = mentions_df.groupby("year_month")[
    ["mentions_heat_pump", "mentions_boiler"]
].sum()
yearly_mentions = mentions_df.groupby("year")[
    ["mentions_heat_pump", "mentions_boiler"]
].sum()

# monthly_mentions = mentions_df.groupby("year_month")[
#     ["mentions_heat_pump", "mentions_boiler", "mentions_solar"]
# ].sum()
# yearly_mentions = mentions_df.groupby("year")[
#     ["mentions_heat_pump", "mentions_boiler", "mentions_solar"]
# ].sum()

# %%
monthly_prop = monthly_mentions.div(monthly_mentions.sum(axis=0))
yearly_prop = yearly_mentions.div(yearly_mentions.sum(axis=0))

# %%
monthly_mentions_rolling_avg = monthly_mentions.rolling(window=3).mean()
yearly_mentions_rolling_avg = yearly_mentions.rolling(window=3).mean()

# %%
yearly_mentions_rolling_avg = yearly_mentions_rolling_avg[15:-1]
monthly_mentions_rolling_avg = monthly_mentions_rolling_avg[175:-1]

# %%
monthly_prop_rolling = monthly_mentions_rolling_avg.div(
    monthly_mentions_rolling_avg.sum(axis=0)
)
yearly_prop_rolling = yearly_mentions_rolling_avg.div(
    yearly_mentions_rolling_avg.sum(axis=0)
)

# %%
monthly_prop_rolling.index = monthly_prop_rolling.index.astype(str)
yearly_prop_rolling.index = yearly_prop_rolling.index.astype(str)
monthly_mentions.index = monthly_mentions.index.astype(str)
yearly_mentions.index = yearly_mentions.index.astype(str)
monthly_mentions_rolling_avg.index = monthly_mentions_rolling_avg.index.astype(str)
yearly_mentions_rolling_avg.index = yearly_mentions_rolling_avg.index.astype(str)

# %%
monthly_prop_rolling.plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Proportion of mentions of heating technologies in posts")
plt.xlabel("")

# %%


# %%
monthly_mentions_rolling_avg.plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Number of mentions of heating technologies in posts")
plt.xlabel("")

# %%
monthly_mentions

# %%
monthly_mentions[175:].plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Number of mentions of heating technologies in posts")
plt.xlabel("")

# %%
yearly_mentions[:-1].plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Number of mentions of heating technologies in posts")
plt.xlabel("")

# %%
yearly_mentions[:-1].div(yearly_mentions.sum(axis=0)).plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Proportion of mentions of heating technologies in posts")
plt.xlabel("")

# %%


# %% [markdown]
# ## Mentions over time - posts + replies

# %%
mentions_df = mse_data.copy()

key_terms = ["heat pump", "boiler", "solar"]
# Preprocess data
for term in key_terms:
    column_name = f"mentions_{term.replace(' ', '_')}"
    if term != "solar":
        ids = mentions_df[
            (
                mentions_df["title"].str.contains(term, case=False)
                | mentions_df["text"].str.contains(term, case=False)
            )
            & (mentions_df["is_original_post"] == 1)
        ]["id"].unique()

        mentions_df[column_name] = mentions_df["text"].str.contains(
            term, case=False
        ) | mentions_df["id"].isin(ids)
    else:
        terms = ["solar pv", "solar panel", "solar photovoltaic"]
        ids = mentions_df[
            (
                mentions_df["title"].str.contains("|".join(terms), case=False)
                | mentions_df["text"].str.contains("|".join(terms), case=False)
            )
            & (mentions_df["is_original_post"] == 1)
        ]["id"].unique()

        mentions_df[column_name] = mentions_df["text"].str.contains(
            "|".join(terms), case=False
        ) | mentions_df["id"].isin(ids)

# %%
for col in key_terms:
    column_name = f"mentions_{col.replace(' ', '_')}"
    mentions_df[column_name] = mentions_df[column_name].astype(int)

# %%
mentions_df["year_month"] = mentions_df["datetime"].dt.to_period("M")
mentions_df["year"] = mentions_df["datetime"].dt.year

# %%
# monthly_mentions = mentions_df.groupby("year_month")[
#     ["mentions_heat_pump", "mentions_boiler"]
# ].sum()
# yearly_mentions = mentions_df.groupby("year")[
#     ["mentions_heat_pump", "mentions_boiler"]
# ].sum()


monthly_mentions = mentions_df.groupby("year_month")[
    ["mentions_heat_pump", "mentions_boiler", "mentions_solar"]
].sum()
yearly_mentions = mentions_df.groupby("year")[
    ["mentions_heat_pump", "mentions_boiler", "mentions_solar"]
].sum()

# %%
monthly_prop = monthly_mentions.div(monthly_mentions.sum(axis=0))
yearly_prop = yearly_mentions.div(yearly_mentions.sum(axis=0))

# %%
monthly_mentions_rolling_avg = monthly_mentions.rolling(window=3).mean()
yearly_mentions_rolling_avg = yearly_mentions.rolling(window=3).mean()

# %%
yearly_mentions_rolling_avg = yearly_mentions_rolling_avg[15:-1]
monthly_mentions_rolling_avg = monthly_mentions_rolling_avg[175:-1]

# %%
monthly_prop_rolling = monthly_mentions_rolling_avg.div(
    monthly_mentions_rolling_avg.sum(axis=0)
)
yearly_prop_rolling = yearly_mentions_rolling_avg.div(
    yearly_mentions_rolling_avg.sum(axis=0)
)

# %%
monthly_prop_rolling.index = monthly_prop_rolling.index.astype(str)
yearly_prop_rolling.index = yearly_prop_rolling.index.astype(str)
monthly_mentions.index = monthly_mentions.index.astype(str)
yearly_mentions.index = yearly_mentions.index.astype(str)
monthly_mentions_rolling_avg.index = monthly_mentions_rolling_avg.index.astype(str)
yearly_mentions_rolling_avg.index = yearly_mentions_rolling_avg.index.astype(str)

# %%
monthly_prop_rolling.plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Proportion of mentions of heating technologies in posts")
plt.xlabel("")

# %%


# %%
monthly_mentions_rolling_avg.plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Number of mentions of heating technologies in posts and replies")
plt.xlabel("")

# %%
monthly_mentions

# %%
monthly_mentions[175:].plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
# monthly_mentions[175:].plot(
#     kind="line", figsize=(10, 6), color=[NESTA_COLOURS[8], NESTA_COLOURS[7]]
# )
# plt.legend(["Heat Pumps", "Boilers"], loc="upper left")
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Number of mentions of heating technologies in posts and replies")
plt.xlabel("")

# %%
yearly_mentions[:-1].plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Number of mentions of heating technologies in posts")
plt.xlabel("")

# %%
yearly_mentions[:-1].div(yearly_mentions.sum(axis=0)).plot(
    kind="line",
    figsize=(10, 6),
    color=[NESTA_COLOURS[8], NESTA_COLOURS[7], NESTA_COLOURS[11]],
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Proportion of mentions of heating technologies in posts")
plt.xlabel("")

# %%


# %% [markdown]
# ## Renaming and grouping topics

# %%
renaming_and_grouping_topics = {
    "Solar panels and solar PV": "0_solar_panels_battery_pv",
    "Unrelated to HPs": {
        "General sentences": "1_thread_im_think_post",
        "Tumble dryers": "5_dryer_tumble_dry_washing",
    },
    "Other heating systems": {
        "Old gas boilers": "2_boiler_boilers_gas_old",
        "Gas and fossil fuels": "16_gas_fossil_fuels_fuel",
        "Heat Pumps vs. boilers": "20_boiler_pump_gas_oil",
        "LPG and oil": "56_lpg_oil_bulk_tank",
        "Oil vs. ASHP": "59_source_air_oil_pump",
        "Hydrogen": "62_hydrogen_hvo_h2_gas",
    },
    "Insulation": "3_insulation_loft_insulated_wall",
    "Underfloor heating and radiators": {
        "Radiators": "6_radiators_radiator_delta_temperature",
        "Underfloor heating and radiators": "13_underfloor_floor_heating_radiators",
    },
    "Property": "4_house_property_bed_bungalow",
    "Money and costs": {
        "Savings": "7_savings_price_money_suppliers",
        "Bills and credit": "17_bills_credit_year_month",
        "Heating systems and costs": "23_heating_systems_costs_heat",
        "Electricity cost": "33_electricity_year_kwh_cost",
        "Heat pump cost": "35_pump_cost_source_air",
        "Energy prices and electricity cost": "61_electricity_energy_prices_cost",
        "Oil cost": "54_oil_oils_club_cost",
    },
    "Domestic hot water": "8_water_hot_cylinder_domestic",
    "Smart meters and readings": "10_meter_smart_meters_readings",
    "Noise": "11_noise_noisy_quiet_microgeneration",
    "Electricity and gas consumption": {
        "Average daily consumption": "12_kwh_average_day_kw",
        "Heating hot water consumption": "27_kwh_hot_heating_water",
        "Heat pump consumption": "28_pump_heat_kw_kwh",
        "Gas usage": "57_gas_usage_kwh_year",
    },
    "Heat Pump types and suitability": {
        "Air source heat pumps": "14_air_source_pump_heat",
        "General heat pumps": "15_pumps_heat_pump_dont",
        "Ground source heat pumps": "36_ground_source_pump_heat",
    },
    "Installations and installers": "18_installer_installers_installation_install",
    "Heat pump performance": "19_cop_scop_flow_degrees",
    "Tariffs": {
        "Octopus agile": "21_octopus_tariffs_tariff_agile",
        "Octopus cosy and heat pumps": "50_octopus_cosy_pump_heat",
        "Economy7 tariff": "26_e7_rate_tariff_economy",
        "Time of use tariffs": "46_tariff_tariffs_rate_tou",
    },
    "Showers and baths": "22_shower_showers_bath_electric",
    "Settings and Controls": "24_settings_controls_serviced_control",
    "Grants": {
        "Renewable heat incentive": "30_renewable_incentive_payments_heat",
        "Eco4 and other gov grants": "42_grant_grants_eco4_government",
    },
    "Numbers and calculations": "31_figures_numbers_calculations_sums",
    "Planning permissions": {
        "Planning permissions and councils": "32_planning_permission_council_ipswich",
        "Planning/development permissions": "55_permission_planning_development_permitted",
    },
    "Pipework and plumbing": "34_pipes_pipe_pipework_plumbing",
    "Wood burners and stoves": "37_wood_burner_stove_log",
    "Tanks and storage heaters": {
        "Storage heaters": "25_storage_heaters_heater_electric",
        "Tanks": "38_tank_tanks_bunded_litres",
        "Sunamp": "53_sunamp_phase_sunamps_change",
    },
    "Flow temperature": "39_flow_temp_temperature_temperatures",
    "Phase change materials": "40_phase_change_pcm_liquid",
    "Weather and temperature": {
        "Setting thermostast temperature": "9_thermostat_degrees_set_temperature",
        "Weather compensation": "29_compensation_weather_curve_temperature",
        "Winter and cold/mild weather": "41_winter_cold_mild_weather",
        "Heat pump temperature": "49_pump_heat_temperature_source",
    },
    "EPCs and energy performance": {
        "EPCs and properties": "43_epc_epcs_rating_property",
        "EPCs and heating": "48_epc_rating_heating_heat",
    },
    "Legionella in domestic hot water systems": "44_legionella_cycle_immersion_bacteria",
    "Complaints": "52_email_complaint_phone_emails",
    "Aircon units": "58_air_units_aircon_split",
    "MCS": "60_certification_microgeneration_scheme_certificate",
    "Time of use": "45_hours_247_hour_minutes",
    "Technical: other": "47_fuse_phase_fuses_3phase",
    "Green planet and sustainability": "51_green_planet_sustainability_environmental",
}

# %%
# Create a flat mapping dictionary
flat_mapping = {}


# Function to flatten the mapping recursively
def flatten_mapping(mapping, parent_key=None):
    for key, value in mapping.items():
        if isinstance(value, dict):
            flatten_mapping(value, parent_key=key)
        else:
            if parent_key is not None:
                flat_mapping[value] = parent_key
            else:
                flat_mapping[value] = key

    return flat_mapping


flat_mapping = flatten_mapping(renaming_and_grouping_topics)


# Function to map the values
def map_values(value):
    return flat_mapping.get(value, value)


# %%
# Create a flat mapping dictionary
flat_mapping_child = {}


# Function to flatten the mapping recursively
def flatten_mapping_child_key(mapping):
    for key, value in mapping.items():
        if isinstance(value, dict):
            flatten_mapping_child_key(value)
        else:
            flat_mapping_child[value] = key

    return flat_mapping_child


flat_mapping_child = flatten_mapping_child_key(renaming_and_grouping_topics)


# Function to map the values
def map_values_child(value):
    return flat_mapping_child.get(value, value)


# %%
topics_info["aggregated_topic_names"] = topics_info["Name"].apply(map_values)
topics_info["topic_names"] = topics_info["Name"].apply(map_values_child)
doc_info["aggregated_topic_names"] = doc_info["Name"].apply(map_values)
doc_info["topic_names"] = doc_info["Name"].apply(map_values_child)

# %%
topics_info[
    ~topics_info["aggregated_topic_names"].isin(
        ["-1_heat_pump_heating_air", "Unrelated to HPs"]
    )
]["aggregated_topic_names"].nunique()

# %% [markdown]
# ## Aggregated topics

# %%
aggregated_topics = (
    topics_info[
        (topics_info["Topic"] != -1)
        & (topics_info["aggregated_topic_names"] != "Unrelated to HPs")
    ]
    .groupby("aggregated_topic_names", as_index=False)[["updated_count"]]
    .sum()
)
aggregated_topics.sort_values("updated_count", ascending=True, inplace=True)

# %%
aggregated_topics.plot(
    kind="barh",
    x="aggregated_topic_names",
    y="updated_count",
    figsize=(9, 20),
    color=NESTA_COLOURS[0],
)
plt.legend().remove()
plt.xlabel("Number of sentences")
plt.ylabel("")
plt.yticks(fontsize=22)

# %% [markdown]
# # Sentiment of aggregated topics

# %%
len(doc_info[doc_info["Topic"] != -1]), len(sentiment_data)

# %%
doc_info[~doc_info["sentences"].isin(sentiment_data["text"])][
    "aggregated_topic_names"
].unique()

# %%
agg_topic_sentiment = doc_info.merge(
    sentiment_data, how="left", left_on="Document", right_on="text"
)

# %%
agg_topic_sentiment = (
    agg_topic_sentiment.groupby(["aggregated_topic_names", "sentiment"])
    .nunique()["Document"]
    .unstack()
    .fillna(0)
)

# %%
agg_topic_sentiment = (
    agg_topic_sentiment.div(agg_topic_sentiment.sum(axis=1), axis=0) * 100
)

# %%
agg_topic_sentiment = agg_topic_sentiment[
    ~agg_topic_sentiment.index.isin(["-1_heat_pump_heating_air", "Unrelated to HPs"])
]

# %%
agg_topic_sentiment.sort_values("negative", ascending=True, inplace=True)

# %%


# %%
agg_topic_sentiment.plot(
    kind="barh",
    stacked=True,
    color=[NESTA_COLOURS[4], NESTA_COLOURS[11], NESTA_COLOURS[1]],
    figsize=(9, 20),
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel("Percentage of sentences", fontsize=20)
plt.ylabel("")
plt.yticks(fontsize=26)
plt.xticks(fontsize=20)
x_values = range(20, 100, 20)
for x in x_values:
    plt.axvline(x=x, color="grey", linestyle="--", linewidth=0.8, alpha=1)

# %%


# %% [markdown]
# ## Aggregated topics: size (total count) vs sentiment
#

# %%
agg_topic_sentiment.reset_index(inplace=True)

# %%
agg_topic_sentiment

# %%
sentiment_vs_size = agg_topic_sentiment[["aggregated_topic_names", "negative"]].merge(
    aggregated_topics, on="aggregated_topic_names"
)

# %%
ax = sentiment_vs_size.plot(
    kind="scatter", x="updated_count", y="negative", figsize=(10, 10)
)

# Add text labels on each dot
for i, row in sentiment_vs_size.iterrows():
    ax.text(
        row["updated_count"],
        row["negative"],
        row["aggregated_topic_names"],
        fontsize=8,
        ha="left",
    )

plt.show()

# %%


# %% [markdown]
# ## Computing n-grams

# %%
#!python -m spacy download en_core_web_sm

# %%
import nltk
from collections import Counter

# Download the stopwords
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Get spaCy's stopwords
stop_words2 = nlp.Defaults.stop_words
stop_words = set(stop_words.union(stop_words2))
import ast


# %%
def filter_ngrams(ngrams):
    filtered_ngrams = Counter(
        {
            ngram: count
            for ngram, count in ngrams.items()
            if not any(word in stop_words for word in ngram.split())
        }
    )

    # filtered_ngrams = Counter({ngram: count for ngram, count in filtered_ngrams.items() if count >= threshold})

    return filtered_ngrams


# %%
def revert_abbreviations(text):
    text = (
        text.replace("air source heat pumps", "ashps")
        .replace("air source heat pump", "ashp")
        .replace(
            "ground source heat pumps",
            "gshps",
        )
        .replace("ground source heat pump", "gshp")
        .replace("under floor heating", "ufh")
        .replace("renewable heat incentive", "rhi")
        .replace("microgeneration certification scheme", "mcs")
        .replace("domestic hot water", "dhw")
        .replace(
            "air to air",
            "a2a",
        )
        .replace(
            "infrared",
            "IR",
        )
        .replace("unvented cylinders", "uvcs")
        .replace("unvented cylinder", "uvc")
    )

    # additional ones
    text = (
        text.replace("heat pumps", "HPs")
        .replace("heat pump", "HP")
        .replace("boiler upgrade scheme", "bus")
        .replace("underfloor heating", "ufh")
    )

    return text


# %%
import string
from nltk.tokenize import word_tokenize

n_gram_data = sentences_data[["sentences"]]
n_gram_data["sentences_revert_abbrev"] = n_gram_data["sentences"].apply(
    revert_abbreviations
)

n_gram_data["tokens"] = n_gram_data["sentences_revert_abbrev"].apply(word_tokenize)
n_gram_data["non_punctuation_tokens"] = n_gram_data["tokens"].apply(
    lambda x: [token for token in x if token not in string.punctuation]
)

# %%
from asf_public_discourse_home_decarbonisation.utils.ngram_utils import (
    create_ngram_from_ordered_tokens,
    frequency_ngrams,
)


# %%
n_gram_data["bigrams"] = n_gram_data.apply(
    lambda x: create_ngram_from_ordered_tokens(x["non_punctuation_tokens"], n=2), axis=1
)
n_gram_data["trigrams"] = n_gram_data.apply(
    lambda x: create_ngram_from_ordered_tokens(x["non_punctuation_tokens"], n=3), axis=1
)

# %%
n_gram_data = n_gram_data.merge(
    doc_info[["sentences", "topic_names", "aggregated_topic_names"]],
    left_on="sentences",
    right_on="sentences",
)
n_gram_data = n_gram_data.merge(
    sentiment_data[["text", "sentiment"]], left_on="sentences", right_on="text"
)

# %%
import math

# %%
for t in n_gram_data["aggregated_topic_names"].unique():
    n_gram_subset = n_gram_data[n_gram_data["aggregated_topic_names"] == t]

    fig, axs = plt.subplots(2, 3, figsize=(20, 15))
    aux = filter_ngrams(
        frequency_ngrams(n_gram_subset, "non_punctuation_tokens")
    ).most_common(10)
    if len(aux) > 0:
        axs[0, 0].bar([x[0] for x in aux], [x[1] for x in aux], color=NESTA_COLOURS[0])
        axs[0, 0].set_xticklabels(
            [x[0] for x in aux], rotation=45, ha="right", fontsize=14
        )

    if len(aux) > 0:
        aux = filter_ngrams(frequency_ngrams(n_gram_subset, "bigrams")).most_common(10)
        axs[0, 1].bar([x[0] for x in aux], [x[1] for x in aux], color=NESTA_COLOURS[0])
        axs[0, 1].set_xticklabels(
            [x[0] for x in aux], rotation=45, ha="right", fontsize=14
        )

    if len(aux) > 0:
        aux = filter_ngrams(frequency_ngrams(n_gram_subset, "trigrams")).most_common(10)
        axs[0, 2].bar([x[0] for x in aux], [x[1] for x in aux], color=NESTA_COLOURS[0])
        axs[0, 2].set_xticklabels(
            [x[0] for x in aux], rotation=45, ha="right", fontsize=14
        )

    n_gram_subset_neg = n_gram_data[
        (n_gram_data["aggregated_topic_names"] == t)
        & (n_gram_data["sentiment"] == "negative")
    ]

    threshold = math.ceil(len(n_gram_subset_neg) * 0.01)

    if len(aux) > 0:
        aux = filter_ngrams(
            frequency_ngrams(n_gram_subset_neg, "non_punctuation_tokens")
        ).most_common(10)
        axs[1, 0].bar([x[0] for x in aux], [x[1] for x in aux], color=NESTA_COLOURS[4])
        axs[1, 0].set_xticklabels(
            [x[0] for x in aux], rotation=45, ha="right", fontsize=14
        )

    if len(aux) > 0:
        aux = filter_ngrams(frequency_ngrams(n_gram_subset_neg, "bigrams")).most_common(
            10
        )
        axs[1, 1].bar([x[0] for x in aux], [x[1] for x in aux], color=NESTA_COLOURS[4])
        axs[1, 1].set_xticklabels(
            [x[0] for x in aux], rotation=45, ha="right", fontsize=14
        )

    if len(aux) > 0:
        aux = filter_ngrams(
            frequency_ngrams(n_gram_subset_neg, "trigrams")
        ).most_common(10)
        axs[1, 2].bar([x[0] for x in aux], [x[1] for x in aux], color=NESTA_COLOURS[4])
        axs[1, 2].set_xticklabels(
            [x[0] for x in aux], rotation=45, ha="right", fontsize=14
        )

    plt.tight_layout()
    plt.suptitle(t + ": " + str(len(n_gram_subset)) + " sentences", fontsize=12)


# %%


# %% [markdown]
# ## Looking at specific topics

# %% [markdown]
# Sentiment for specific topics:

# %%
topic_sentiment = doc_info.merge(
    sentiment_data, how="left", left_on="Document", right_on="text"
)

# %%
topic_sentiment = (
    topic_sentiment.groupby(["topic_names", "sentiment"])
    .nunique()["Document"]
    .unstack()
    .fillna(0)
)
topic_sentiment = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0) * 100

# %%
topics_date = sentences_data.merge(
    doc_info[["Document", "aggregated_topic_names", "topic_names"]],
    left_on="sentences",
    right_on="Document",
)[["sentences", "datetime", "aggregated_topic_names", "topic_names"]]
# topics_date = sentences_data.merge(doc_info[["Document", "aggregated_topic_names", "topic_names"]], left_on="sentences", right_on="Document")[["sentences", "year", "aggregated_topic_names", "topic_names"]]

# %%
topics_date["year_month"] = pd.to_datetime(topics_date["datetime"]).dt.to_period("M")
topics_date["year_month"] = topics_date["year_month"].astype(str)

# %%
topics_date = topics_date.groupby(
    ["aggregated_topic_names", "topic_names", "year_month"]
).count()[["sentences"]]
# topics_date = topics_date.groupby(["aggregated_topic_names", "topic_names", "year"]).count()[["sentences"]]
topics_date.reset_index(inplace=True)


# %%
topics_date = topics_date[topics_date["aggregated_topic_names"] != "Unrelated to HPs"]
topics_date = topics_date[
    topics_date["aggregated_topic_names"] != "-1_heat_pump_heating_air"
]

# %%
year_month_list = pd.period_range(start="2018-01", end="2024-04", freq="M")
year_month_list = year_month_list.strftime("%Y-%m")

# %%
for agg in topics_date.aggregated_topic_names.unique():
    original_topics = topics_info[topics_info["aggregated_topic_names"] == agg][
        "topic_names"
    ].unique()
    topic_sentimet_f = topic_sentiment[topic_sentiment.index.isin(original_topics)]
    if len(topic_sentimet_f) > 0:
        topic_sentimet_f.sort_values("negative", ascending=True, inplace=True)

        counts = (
            topics_info[topics_info["aggregated_topic_names"] == agg]
            .groupby("topic_names")
            .sum()[["updated_count"]]
        )
        counts = counts.reindex(topic_sentimet_f.index)

        time_counts = (
            topics_date[topics_date["aggregated_topic_names"] == agg]
            .drop(columns="aggregated_topic_names")
            .groupby(["topic_names", "year_month"])
            .sum()[["sentences"]]
            .unstack(level=0)
            .fillna(0)
        )
        # time_counts = topics_date[topics_date["aggregated_topic_names"]==agg].drop(columns="aggregated_topic_names").set_index(["topic_names", "year"]).unstack(level=0).fillna(0)
        time_counts.columns = time_counts.columns.droplevel(0)
        time_counts.reset_index(inplace=True)

        time_counts = time_counts.set_index("year_month")
        # time_counts = time_counts.set_index("year")#.rolling(window=1).mean()

        full_range = pd.DataFrame(index=year_month_list)
        # full_range = pd.DataFrame(index = range(2018, 2025,1))
        full_range = full_range.merge(
            time_counts, left_index=True, right_index=True, how="left"
        )
        full_range = full_range.fillna(0)
        # full_range = full_range[full_range.index.month % 3 == 1]

        # Create a figure with two subplots side by side
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(12, 8))
        # Plot topic_sentimet_f on the first axis
        topic_sentimet_f.plot(
            kind="barh",
            stacked=True,
            color=[NESTA_COLOURS[4], NESTA_COLOURS[11], NESTA_COLOURS[1]],
            ax=ax1,
        )
        ax1.set_xlabel("Percentage of sentences")
        ax1.set_ylabel("")  # We don't need a ylabel here, shared with ax2
        ax1.tick_params(axis="y", labelsize=12)
        ax1.legend(
            title="Sentiment",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title_fontsize=10,
            fontsize=10,
        )
        # ax1.set_title('Sentiment Distribution by Topic', fontsize=15)

        # Plot counts on the second axis
        counts.plot(kind="barh", color=NESTA_COLOURS[0], ax=ax2)
        ax2.set_xlabel("Number of sentences")
        ax2.set_ylabel("")  # We don't need a ylabel here, shared with ax1
        ax2.tick_params(axis="y", labelsize=12)
        ax2.legend().remove()
        # ax2.set_title('Count of Sentences by Topic', fontsize=15)

        full_range.plot(
            kind="line", ax=ax3, color=NESTA_COLOURS[: len(full_range.columns)]
        )
        ax3.set_xlabel("")
        ax3.set_ylabel("Number of sentences", fontsize=12)
        ax3.legend(
            title="Topic",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title_fontsize=10,
            fontsize=10,
        )
        ax3.set_xticks(
            [i for i in range(0, len(year_month_list), 6)] + [len(year_month_list) - 1],
            [year_month_list[i] for i in range(0, len(year_month_list), 6)]
            + [year_month_list[len(year_month_list) - 1]],
            fontsize=10,
            rotation=45,
            ha="right",
        )
        # ax3.set_xticks(range(2018,2025,1))

        # Adjust the layout to make sure everything fits
        plt.tight_layout()
    else:
        print("No sentiment data for topic: ", agg)

# %%
# topics_date["topic_names"].replace({"Planning permissions and councils": "Planning permissions", "Planning/development permissions":"Planning permissions"}, inplace=True)

# %%
for agg in topics_date.aggregated_topic_names.unique():
    if len(topic_sentimet_f) > 0:
        time_counts = (
            topics_date[topics_date["aggregated_topic_names"] == agg]
            .drop(columns="aggregated_topic_names")
            .groupby(["topic_names", "year_month"])
            .sum()[["sentences"]]
            .unstack(level=0)
            .fillna(0)
        )
        # time_counts = topics_date[topics_date["aggregated_topic_names"]==agg].drop(columns="aggregated_topic_names").set_index(["topic_names", "year"]).unstack(level=0).fillna(0)
        time_counts.columns = time_counts.columns.droplevel(0)
        time_counts.reset_index(inplace=True)

        time_counts = time_counts.set_index("year_month")
        # time_counts = time_counts.set_index("year")#.rolling(window=1).mean()

        full_range = pd.DataFrame(index=year_month_list)
        # full_range = pd.DataFrame(index = range(2018, 2025,1))
        full_range = full_range.merge(
            time_counts, left_index=True, right_index=True, how="left"
        )
        full_range = full_range.fillna(0)
        # full_range = full_range[full_range.index.month % 3 == 1]

        if full_range.shape[1] > 1:
            full_range.plot(
                kind="line",
                color=NESTA_COLOURS[: len(full_range.columns)],
                figsize=(12, 4),
            )
            plt.legend(
                title="Topic",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                title_fontsize=10,
                fontsize=10,
            )
        else:
            full_range.plot(
                kind="line",
                color=NESTA_COLOURS[: len(full_range.columns)],
                figsize=(8, 4),
            )
            plt.title("Topic: " + agg)
            plt.legend().remove()
        plt.xlabel("")
        plt.ylabel("Number of sentences", fontsize=12)
        plt.xticks(
            [i for i in range(0, len(year_month_list), 6)] + [len(year_month_list) - 1],
            [year_month_list[i] for i in range(0, len(year_month_list), 6)]
            + [year_month_list[len(year_month_list) - 1]],
            fontsize=10,
            rotation=45,
            ha="right",
        )
        # ax3.set_xticks(range(2018,2025,1))

        # Adjust the layout to make sure everything fits
        plt.tight_layout()
    else:
        print("No sentiment data for topic: ", agg)

# %%
for t in doc_info["topic_names"].unique():
    docs_in_topic = doc_info[doc_info["topic_names"] == t]
    docs_in_topic = docs_in_topic.merge(
        sentiment_data, how="left", left_on="Document", right_on="text"
    )
    pos = docs_in_topic[docs_in_topic["sentiment"] == "positive"].sort_values(
        "sentiment", ascending=False
    )
    neg = docs_in_topic[docs_in_topic["sentiment"] == "negative"].sort_values(
        "sentiment", ascending=False
    )

    print("Topic: ", t)
    print("Positive: ")
    if len(pos) > 0:
        print(pos["Document"].iloc[0])
    if len(pos) > 1:
        print(pos["Document"].iloc[1])
        print("\n")

    print("Negative: ")
    if len(neg) > 0:
        print(neg["Document"].iloc[0])
    if len(neg) > 1:
        print(neg["Document"].iloc[1])
        print("\n")

    print("---")

# %%
doc_info["topic_names"].unique()

# %%
docs_in_topic = doc_info[
    doc_info["topic_names"].str.startswith("Energy prices and electricity cost")
]
docs_in_topic = docs_in_topic.merge(
    sentiment_data, how="left", left_on="Document", right_on="text"
)
pos = docs_in_topic[docs_in_topic["sentiment"] == "positive"].sort_values(
    "sentiment", ascending=False
)
neg = docs_in_topic[docs_in_topic["sentiment"] == "negative"].sort_values(
    "sentiment", ascending=False
)
neu = docs_in_topic[docs_in_topic["sentiment"] == "neutral"].sort_values(
    "sentiment", ascending=False
)

# %%
for i in range(len(neg)):
    print(neg.Document.iloc[i])

# %%


# %% [markdown]
# ## Growth vs. size - comparing 2020 to 2023 (full years)

# %%
sentences_year_info = sentences_data.merge(
    doc_info, how="left", left_on="sentences", right_on="sentences"
)[["sentences", "topic_names", "aggregated_topic_names", "year"]]
sentences_year_info.head()

# %%
n_2023 = (
    sentences_year_info[sentences_year_info["year"] == 2023]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)
n_2020 = (
    sentences_year_info[sentences_year_info["year"] == 2020]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)

# %%
sentiment_vs_size = agg_topic_sentiment[["aggregated_topic_names", "negative"]].merge(
    aggregated_topics, on="aggregated_topic_names"
)

# %%
sentiment_vs_size = (
    sentiment_vs_size.merge(n_2023)
    .rename(columns={"sentences": "n_2023"})
    .merge(n_2020)
    .rename(columns={"sentences": "n_2020"})
)

# %%
sentiment_vs_size["growth_2023_2020"] = (
    (sentiment_vs_size["n_2023"] - sentiment_vs_size["n_2020"])
    / sentiment_vs_size["n_2020"]
    * 100
)

# %%
sentiment_vs_size

# %%
ax = sentiment_vs_size.plot(
    kind="scatter",
    x="growth_2023_2020",
    y="negative",
    s="updated_count",
    figsize=(10, 10),
)

# Add text labels on each dot
for i, row in sentiment_vs_size.iterrows():
    ax.text(
        row["growth_2023_2020"],
        row["negative"],
        row["aggregated_topic_names"],
        fontsize=8,
        ha="left",
    )

plt.xscale("log")
plt.yscale("log")

plt.show()

# %%


# %% [markdown]
# ## Growth vs. size - comparing time periods
#
# -  1) May 2019 to April 2020 and 2) May 2023 to April 2024
# and
# -  1) May 2022 to April 2023 and 2) May 2023 to April 2024
#
#
#
# Final plot done in Flourish.

# %%
sentences_data["year_month"] = pd.to_datetime(sentences_data["datetime"]).dt.to_period(
    "M"
)
sentences_data["year_month"] = sentences_data["year_month"].astype(str)

# %%
sentences_year_month_info = sentences_data.merge(
    doc_info, how="left", left_on="sentences", right_on="sentences"
)[["sentences", "topic_names", "aggregated_topic_names", "year_month"]]
sentences_year_month_info.head()

# %%
n_2024 = (
    sentences_year_month_info[
        sentences_year_month_info["year_month"].isin(
            [
                "2023-05",
                "2023-06",
                "2023-07",
                "2023-08",
                "2023-09",
                "2023-10",
                "2023-11",
                "2023-12",
                "2024-01",
                "2024-02",
                "2024-03",
                "2024-04",
            ]
        )
    ]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)
n_2023 = (
    sentences_year_month_info[
        sentences_year_month_info["year_month"].isin(
            [
                "2022-05",
                "2022-06",
                "2022-07",
                "2022-08",
                "2022-09",
                "2022-10",
                "2022-11",
                "2022-12",
                "2023-01",
                "2023-02",
                "2023-03",
                "2023-04",
            ]
        )
    ]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)
n_2020 = (
    sentences_year_month_info[
        sentences_year_month_info["year_month"].isin(
            [
                "2019-05",
                "2019-06",
                "2019-07",
                "2019-08",
                "2019-09",
                "2019-10",
                "2019-11",
                "2019-12",
                "2020-01",
                "2020-02",
                "2020-03",
                "2020-04",
            ]
        )
    ]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)

# %%
sentiment_vs_size = agg_topic_sentiment[["aggregated_topic_names", "negative"]].merge(
    aggregated_topics, on="aggregated_topic_names"
)

# %%
sentiment_vs_size = (
    sentiment_vs_size.merge(n_2024)
    .rename(columns={"sentences": "n_2024"})
    .merge(n_2023)
    .rename(columns={"sentences": "n_2023"})
    .merge(n_2020)
    .rename(columns={"sentences": "n_2020"})
)

# %%
sentiment_vs_size

# %%
sentiment_vs_size["growth_2024_2023"] = (
    (sentiment_vs_size["n_2024"] - sentiment_vs_size["n_2023"])
    / sentiment_vs_size["n_2023"]
    * 100
)
sentiment_vs_size["growth_2024_2020"] = (
    (sentiment_vs_size["n_2024"] - sentiment_vs_size["n_2020"])
    / sentiment_vs_size["n_2020"]
    * 100
)

# %%
sentiment_vs_size[sentiment_vs_size["growth_2024_2020"] > 0]

# %%
ax = sentiment_vs_size.plot(
    kind="scatter",
    x="growth_2024_2020",
    y="negative",
    s="updated_count",
    figsize=(10, 10),
)

# Add text labels on each dot
for i, row in sentiment_vs_size.iterrows():
    ax.text(
        row["growth_2024_2020"],
        row["negative"],
        row["aggregated_topic_names"],
        fontsize=8,
        ha="left",
    )

plt.xscale("log")

plt.show()

# %%
sentiment_vs_size.to_csv("sentiment_vs_size.csv", index=False)

# %% [markdown]
# ## Sentiment over time

# %%
sentiment_over_time = (
    sentences_data[["year", "sentences"]]
    .merge(sentiment_data, left_on="sentences", right_on="text")
    .groupby(["sentiment", "year"])
    .count()[["sentences"]]
    .unstack()
    .fillna(0)
)
sentiment_over_time.columns = sentiment_over_time.columns.droplevel(0)

# %%
sentiment_over_time = sentiment_over_time.div(sentiment_over_time.sum(axis=0))

# %%
sentiment_over_time

# %%
sentiment_over_time.T.plot(kind="line", figsize=(10, 6), color=["red", "grey", "green"])
plt.title("Sentiment over time")

# %%
