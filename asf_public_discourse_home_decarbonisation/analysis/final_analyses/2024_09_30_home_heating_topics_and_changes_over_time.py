# %% [markdown]
# ## overview of analysis
#
# Analysis of topics of conversation related to home heating:
#
# - Loads topic analysis and sentiment analysis results from S3 (from running topic analysis on MSE home heating forum posts in the past 20 years )
# - Visualises:
#     - topic sizes;
#     - breakdown of sentiment
#     - differences in number of conversations over time;
#     - growth in topics between 2020 and 2024.

# %% [markdown]
# ## package imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    set_plotting_styles,
    NESTA_COLOURS,
)
from asf_public_discourse_home_decarbonisation import S3_BUCKET

set_plotting_styles()

from asf_public_discourse_home_decarbonisation.utils.general_utils import (
    flatten_mapping,
    flatten_mapping_child_key,
    map_values,
)

# %% [markdown]
# ## analysis specific settings

# %%
# data import settings
source = "mse"
analysis_start_date = None  # i.e. all data from the start
analysis_end_date = "2024-05-22"


# %%
# analysis settings
# topic growth analysis
# first period: may 2018 to april 2020
growth_months_period_start = [
    "2018-05",
    "2018-06",
    "2018-07",
    "2018-08",
    "2018-09",
    "2018-10",
    "2018-11",
    "2018-12",
    "2019-02",
    "2019-03",
    "2019-04",
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
# second period: may 2022 to april 2024
growth_months_period_end = [
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


# %% [markdown]
# ## data imports

# %%
path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/titles_analysis_{source}_{analysis_start_date}_{analysis_end_date}"

topics_info = pd.read_csv(
    f"{path_to_data}_titles_topics_info.csv",
)
doc_info = pd.read_csv(
    f"{path_to_data}_titles_docs_info.csv",
)
forum_data = pd.read_csv(
    f"{path_to_data}_forum_title_data.csv",
)

# %%
path_to_save_data = f"s3://{S3_BUCKET}/data/{source}/outputs/final_analyses"


# %% [markdown]
# ## Renaming and grouping topics

# %%
renaming_and_grouping_topics = {
    "Unrelated to home heating": {
        "Outliers cluster": "-1_options_eon_electric_discount"
    },
    "Energy suppliers and tariffs": {
        "British gas": "1_gas_british_fuel_options",
        "EDF": "12_edf_online_options_blue",
        "NPower": "9_npower_options_online_sign",
        "Switching options": "5_switch_switching_help_options",
        "Scottish Power": "6_scottish_power_scottishpower_hydro",
        "Octopus & avro": "14_octopus_avro_agile_energy",
        "Fixed tariffs and Eon": "15_tariff_tariffs_fixed_eon",
        "Switching suppliers": "16_supplier_suppliers_changing_switching",
        "Eon": "17_eon_rep_fixonline_fix",
        "Ovo & SSE": "20_ovo_ombudsman_energy_sse",
        "Economy 7 and economy 10": "25_economy_10_e7_times",
        "Tariffs": "35_tarrif_tarriff_tarrifs_tarriffs",
    },
    "Bills, charges & utilities": {
        "Bills & credit": "10_bills_credit_eon_final",
        "Fixed price deals": "19_price_fixed_deal_rates",
        "Direct debits": "18_direct_debit_debits_iresa",
        "Standing charges": "26_standing_charge_charges_daily",
        "Utilities": "32_utility_utilities_firstutility_buyer",
        "Monthly direct debits": "34_dd_monthly_increase_dds",
    },
    "Home heating & heating systems": {
        "Energy and electricity": "0_energy_electricity_club_mse",
        "Central heating": "13_heating_central_heat_electric",
        "Stove & wood burner": "11_stove_wood_chimney_burner",
        "Boilers": "4_boiler_combi_new_boilers",
        "Storage heaters": "23_storage_heaters_heater_night",
        "Heat pumps": "28_pump_heat_source_air",
        "Solar panels/PV": "8_solar_panels_pv_panel",
        "Oil heating": "24_oil_heating_tank_central",
        "Radiators": "29_radiator_radiators_towel_valves",
        "LPG": "31_lpg_flogas_tank_bulk",
        "Hot water heating and cylinders": "30_water_hot_heating_cylinder",
        "Insulation": "33_insulation_cavity_wall_loft",
    },
    "Smart meters and readings": "2_meter_smart_meters_readings",
    "Other": {
        "Green miscellaneous": "7_green_recycling_ethical_washing",
        "Moving houses": "21_house_moving_home_new",
        "Lightbulbs": "22_bulb_bulbs_light_led",
        "Cashback": "27_cashback_quidco_cash_switch",
        "Quotes": "3_quote_kitchen_bathroom_roof",
    },
}

# %%
# Create a flat mapping dictionary i.e a 1 to 1 mapping between the old name and new name
flat_mapping_agg_topic_names = flatten_mapping({}, renaming_and_grouping_topics)
flat_mapping_new_topic_names = flatten_mapping_child_key(
    {}, renaming_and_grouping_topics
)

# %%
# Creating new columns with the aggregated and non-aggregated topic names in topics_info and doc_info dataframes
topics_info["aggregated_topic_names"] = topics_info.apply(
    lambda x: map_values(flat_mapping_agg_topic_names, x["Name"]), axis=1
)
topics_info["topic_names"] = topics_info.apply(
    lambda x: map_values(flat_mapping_new_topic_names, x["Name"]), axis=1
)

doc_info["aggregated_topic_names"] = doc_info.apply(
    lambda x: map_values(flat_mapping_agg_topic_names, x["Name"]), axis=1
)
doc_info["topic_names"] = doc_info.apply(
    lambda x: map_values(flat_mapping_new_topic_names, x["Name"]), axis=1
)

# %%
# Setting the aggregated topics to remove from the analysis
aggregated_topics_to_remove = ["Unrelated to home heating", "Other"]

# %% [markdown]
# ## Analysis results

# %% [markdown]
# ### 1. Counts of aggregated topics

# %%
aggregated_topics = (
    topics_info[
        ~topics_info["aggregated_topic_names"].isin(aggregated_topics_to_remove)
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

# %%
non_aggregated_topics = (
    topics_info[
        ~topics_info["aggregated_topic_names"].isin(aggregated_topics_to_remove)
    ]
    .groupby(["topic_names", "aggregated_topic_names"], as_index=False)[
        ["updated_count"]
    ]
    .sum()
)
non_aggregated_topics.sort_values("updated_count", ascending=True, inplace=True)
non_aggregated_topics.plot(
    kind="barh",
    x="topic_names",
    y="updated_count",
    figsize=(9, 20),
    color=NESTA_COLOURS[0],
)
plt.legend().remove()
plt.xlabel("Number of titles")
plt.ylabel("")
plt.yticks(fontsize=22)

# %%
aggregated_topics

# %%
non_aggregated_topics

# %%


# %% [markdown]
# ### 4. Growth vs. size - comparing time periods
#
# 1) May 2019 to April 2020 and 2) May 2023 to April 2024

# %%
forum_data["year_month"] = pd.to_datetime(forum_data["datetime"]).dt.to_period("M")
forum_data["year_month"] = forum_data["year_month"].astype(str)


# %%
sentences_year_month_info = forum_data.merge(
    doc_info, how="left", left_on="sentences", right_on="sentences"
)[["sentences", "topic_names", "aggregated_topic_names", "year_month"]]
sentences_year_month_info.head()

# %%
n_2024 = (
    sentences_year_month_info[
        sentences_year_month_info["year_month"].isin(growth_months_period_end)
    ]
    .groupby("topic_names", as_index=False)
    .count()[["topic_names", "sentences"]]
)
n_2020 = (
    sentences_year_month_info[
        sentences_year_month_info["year_month"].isin(growth_months_period_start)
    ]
    .groupby("topic_names", as_index=False)
    .count()[["topic_names", "sentences"]]
)

# %%
growth = (
    n_2024.rename(columns={"sentences": "n_2024"})
    .merge(n_2020)
    .rename(columns={"sentences": "n_2020"})
)

growth["growth_2024_2020"] = (
    (growth["n_2024"] - growth["n_2020"]) / growth["n_2020"] * 100
)

# %%
growth[growth["growth_2024_2020"] > 0]

# %%


# %% [markdown]
# ### 5. Topic changes over time

# %%
topics_date = forum_data.merge(
    doc_info[["Document", "aggregated_topic_names", "topic_names"]],
    left_on="sentences",
    right_on="Document",
)[["sentences", "datetime", "aggregated_topic_names", "topic_names"]]
topics_date["year_month"] = pd.to_datetime(topics_date["datetime"]).dt.to_period("M")
topics_date["year_month"] = topics_date["year_month"].astype(str)
topics_date["year"] = pd.to_datetime(topics_date["datetime"]).dt.year
topics_date = topics_date.groupby(
    ["aggregated_topic_names", "topic_names", "year_month"]
).count()[["sentences"]]
topics_date.reset_index(inplace=True)
topics_date = topics_date[
    ~topics_date["aggregated_topic_names"].isin(aggregated_topics_to_remove)
]


# %%


# %%
year_month_list = pd.period_range(start="2004-01", end="2024-04", freq="M")
year_month_list = year_month_list.strftime("%Y-%m")


# %% [markdown]
# breakdown for each topic

# %%
for agg in topics_date.aggregated_topic_names.unique():
    time_counts = (
        topics_date[topics_date["aggregated_topic_names"] == agg]
        .drop(columns="aggregated_topic_names")
        .groupby(["topic_names", "year_month"])
        .sum()[["sentences"]]
        .unstack(level=0)
        .fillna(0)
    )
    time_counts.columns = time_counts.columns.droplevel(0)
    time_counts.reset_index(inplace=True)

    time_counts = time_counts.set_index("year_month")

    full_range = pd.DataFrame(index=year_month_list)
    full_range = full_range.merge(
        time_counts, left_index=True, right_index=True, how="left"
    )
    full_range = full_range.fillna(0)

    if full_range.shape[1] > 1:
        full_range.plot(
            kind="line", color=NESTA_COLOURS[: len(full_range.columns)], figsize=(12, 4)
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
            kind="line", color=NESTA_COLOURS[: len(full_range.columns)], figsize=(8, 4)
        )
        plt.legend().remove()
    plt.title("Topic: " + agg)
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

    # Adjust the layout to make sure everything fits
    plt.tight_layout()

# %%


# %% [markdown]
# breakdown for each aggregated topic:

# %%
counts_to_save = pd.DataFrame()

# %%
for agg in topics_date.topic_names.unique():
    time_counts = (
        topics_date[topics_date["topic_names"] == agg]
        .groupby(["year_month"])
        .sum()[["sentences"]]
    )

    full_range = pd.DataFrame(index=year_month_list)
    full_range = full_range.merge(
        time_counts, left_index=True, right_index=True, how="left"
    )
    full_range = full_range.fillna(0)

    if len(counts_to_save) == 0:
        counts_to_save = full_range.copy()
        counts_to_save.rename(columns={"sentences": agg}, inplace=True)
    else:
        counts_to_save[agg] = full_range["sentences"]

    if full_range.shape[1] > 1:
        full_range.plot(
            kind="line", color=NESTA_COLOURS[: len(full_range.columns)], figsize=(12, 4)
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
            kind="line", color=NESTA_COLOURS[: len(full_range.columns)], figsize=(8, 4)
        )
        plt.legend().remove()
    plt.title("Topic: " + agg)
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

    # Adjust the layout to make sure everything fits
    plt.tight_layout()


# %%
topics_year = forum_data.merge(
    doc_info[["Document", "aggregated_topic_names", "topic_names"]],
    left_on="sentences",
    right_on="Document",
)[["sentences", "datetime", "aggregated_topic_names", "topic_names"]]
topics_year["year"] = pd.to_datetime(topics_year["datetime"]).dt.year

topics_year = topics_year.groupby(
    ["aggregated_topic_names", "topic_names", "year"]
).count()[["sentences"]]
topics_year.reset_index(inplace=True)
topics_year = topics_year[
    ~topics_year["aggregated_topic_names"].isin(aggregated_topics_to_remove)
]

# %%
counts_to_save = pd.DataFrame()

# %%
for agg in topics_year.topic_names.unique():
    time_counts = (
        topics_year[topics_year["topic_names"] == agg]
        .groupby(["year"])
        .sum()[["sentences"]]
    )

    full_range = pd.DataFrame(index=range(2003, 2024))
    full_range = full_range.merge(
        time_counts, left_index=True, right_index=True, how="left"
    )
    full_range = full_range.fillna(0)

    if len(counts_to_save) == 0:
        counts_to_save = full_range.copy()
        counts_to_save.rename(columns={"sentences": agg}, inplace=True)
    else:
        counts_to_save[agg] = full_range["sentences"]


# %%
counts_to_save.T.to_csv(f"{path_to_save_data}/counts_over_time_home_heating_T.csv")

# %%
counts_to_save.to_csv(f"{path_to_save_data}/counts_over_time_home_heating.csv")

# %% [markdown]
# ## Organising data for Flourish

# %% [markdown]
# ### 1. Barchat with topic sizes

# %%
non_aggregated_topics.rename(
    columns={
        "aggregated_topic_names": "Category",
        "topic_names": "Topic name",
        "updated_count": "Number of posts",
    }
).reset_index(drop=True)

# %%
non_aggregated_topics[
    non_aggregated_topics["aggregated_topic_names"] == "Home heating & heating systems"
].rename(
    columns={
        "aggregated_topic_names": "Category",
        "topic_names": "Topic name",
        "updated_count": "Number of posts",
    }
).reset_index(
    drop=True
)

# %%


# %% [markdown]
# ### 3. Growth versus proportion of negative

# %%
growth["growth_2024_2020"].min(), growth["growth_2024_2020"].max()

# %%


# %%
growth.merge(
    non_aggregated_topics[["topic_names", "aggregated_topic_names", "updated_count"]],
    left_on="topic_names",
    right_on="topic_names",
)[
    [
        "growth_2024_2020",
        "updated_count",
        "topic_names",
        "aggregated_topic_names",
        "n_2024",
        "n_2020",
    ]
].rename(
    columns={
        "topic_names": "Topic name",
        "aggregated_topic_names": "Category",
        "updated_count": "Number number of sentences",
        "negative": "Proportion of negative sentences",
        "n_2024": "Number of sentences in 2024",
        "n_2020": "Number of sentences in 2020",
        "growth_2024_2020": "Growth in number of sentences (%) between 2020 and 2024",
    }
).reset_index(
    drop=True
)

# %%
