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
topics_info.head()

# %%
len(sentiment_data[sentiment_data["score"] > 0.75]) / len(sentiment_data) * 100

# %% [markdown]
# ## Mentions over time

# %%
mentions_df = mse_data.copy()

key_terms = ["heat pump", "boiler", "solar"]
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
# mentions_df.set_index("datetime", inplace=True)

# %%
for col in key_terms:
    column_name = f"mentions_{col.replace(' ', '_')}"
    mentions_df[column_name] = mentions_df[column_name].astype(int)

# %%
mentions_df["year_month"] = mentions_df["datetime"].dt.to_period("M")
mentions_df["year"] = mentions_df["datetime"].dt.year

# %%
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
monthly_prop_rolling = monthly_mentions_rolling_avg[2:].div(
    monthly_mentions_rolling_avg[2:].sum(axis=0)
)
yearly_prop_rolling = yearly_mentions_rolling_avg[2:].div(
    yearly_mentions_rolling_avg[2:].sum(axis=0)
)

# %%
monthly_prop_rolling[169:].plot(
    kind="line", figsize=(10, 6), color=["#97D9E3", "#0F294A", "#D2C9C0"]
)
plt.legend(["Heat Pumps", "Boilers", "Solar panels/PV"], loc="upper left")
plt.xticks(rotation=45)
plt.title("Proportion of mentions of heating technologies in MSE articles")

# %%


# %% [markdown]
# ## Renaming and grouping topics

# %%
topics_info["Name"].unique()

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
        "Underfloor heating and radiators - 1": "13_underfloor_floor_heating_radiators",
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
    "Heat Pump types": {
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
        "Eco4": "42_grant_grants_eco4_government",
    },
    "Numbers and calculations": "31_figures_numbers_calculations_sums",
    "Planning permissions": {
        "Planning permissions - 1": "32_planning_permission_council_ipswich",
        "Planning permissions - 2": "55_permission_planning_development_permitted",
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
        "Energy performance ratings - 1": "43_epc_epcs_rating_property",
        "Eenergy performance ratings - 2": "48_epc_rating_heating_heat",
    },
    "Legionella in domestic hot water systems": "44_legionella_cycle_immersion_bacteria",
    "Complaints": "52_email_complaint_phone_emails",
    "Aircon units": "58_air_units_aircon_split",
    "MCS": "60_certification_microgeneration_scheme_certificate",
    "24/7": "45_hours_247_hour_minutes",
    "Fuse phase": "47_fuse_phase_fuses_3phase",
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
topics_info["new_topic_names"] = topics_info["Name"].apply(map_values)

# %%
aggregated_topics = (
    topics_info[
        (topics_info["Topic"] != -1)
        & (topics_info["new_topic_names"] != "Unrelated to HPs")
    ]
    .groupby("new_topic_names", as_index=False)[["updated_count"]]
    .sum()
)
aggregated_topics.sort_values("updated_count", ascending=True, inplace=True)

# %%
aggregated_topics.plot(
    kind="barh",
    x="new_topic_names",
    y="updated_count",
    figsize=(10, 12),
    color=NESTA_COLOURS[0],
)
plt.legend().remove()
plt.xlabel("Number of sentences")
plt.ylabel("")
plt.yticks(fontsize=12)

# %% [markdown]
# # Visualisations

# %%
doc_info["new_topic_names"] = doc_info["Name"].apply(map_values)

# %%
doc_info = doc_info.merge(sentiment_data, left_on="Document", right_on="text")

# %%
topic_sentimet = (
    doc_info.groupby(["new_topic_names", "sentiment"])
    .nunique()["Document"]
    .unstack()
    .fillna(0)
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
topic_sentimet

# %% [markdown]
# ## Sentiment stacked plot

# %%
topic_sentimet = topic_sentimet[["negative", "neutral", "positive"]]
topic_sentimet = topic_sentimet.div(topic_sentimet.sum(axis=1), axis=0) * 100

# %%
topic_sentimet = topic_sentimet[
    ~topic_sentimet.index.isin(["-1_heat_pump_heating_air", "Unrelated to HPs"])
]

# %%
topic_sentimet.sort_values("negative", ascending=False, inplace=True)

# %%
topic_sentimet.plot(
    kind="barh", stacked=True, color=["red", "grey", "green"], figsize=(9, 20)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.xlabel("Percentage of sentences", fontsize=20)
plt.ylabel("")

# %%


# %% [markdown]
# ## Topic size vs sentiment
#
# In work

# %%
topic_sentimet.reset_index(inplace=True)

# %%
sentiment_vs_size = topic_sentimet[["new_topic_names", "negative"]].merge(
    aggregated_topics, on="new_topic_names"
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
        row["new_topic_names"],
        fontsize=12,
        ha="center",
    )

plt.show()

# %%
