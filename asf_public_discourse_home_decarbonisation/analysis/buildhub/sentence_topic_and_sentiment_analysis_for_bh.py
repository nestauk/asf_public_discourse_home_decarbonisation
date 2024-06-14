#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    plot_post_distribution_over_time,
    plot_mentions_line_chart,
)
import pandas as pd
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_abbreviations,
)
from asf_public_discourse_home_decarbonisation.utils.preprocessing_utils import (
    preprocess_data_for_linechart_over_time,
    resample_and_calculate_averages_for_linechart,
)


# In[ ]:


bh_data = get_bh_data(category="all")
print("Date Range:", bh_data["date"].min(), "to", bh_data["date"].max())


# In[ ]:


key_terms = ["heat pump", "boiler"]
# Preprocess data
bh_data = preprocess_data_for_linechart_over_time(bh_data, key_terms=key_terms)
# Resample and calculate averages
bh_data_monthly = resample_and_calculate_averages_for_linechart(
    bh_data, key_terms=key_terms, cadence_of_aggregation="M", window=12
)


# In[ ]:


key_terms_colour = {"heat pump": "#97D9E3", "boiler": "#0F294A"}
plot_mentions_line_chart(bh_data_monthly, key_terms_colour, plot_type="both")


# In[12]:


categories_dict = {
    "Cost": [
        "16_price_prices_pay_money",
        "30_10k_price_cost_paid",
        "145_cost_month_year_electricity",
        "53_cost_pump_source_air",
        "134_oil_litre_year_price",
        "97_vat_reclaim_plus_zero",
        "139_quote_quotes_quoting_got",
    ],
    "Power": [
        "7_kwh_kw_12kw_power",
        "18_kw_pump_kwh_source",
        "38_kw_kwh_average_day",
        "138_standby_consumption_power_ecodan",
    ],
    "Heating Systems": {
        "Heat Pumps": [
            "1_source_air_pump_heat",
            "31_pumps_heat_pump_work",
            "45_ground_source_pump_borehole",
            "126_energy_consumption_efficiency_rating",
            "144_ecodan_ecodans_r32_know",
            "68_split_monoblock_monobloc_unit",
        ],
        "Water Heating": [
            "6_hot_domestic_water_heating",
            "109_cylinder_reheat_unvented_cylinders",
            "146_willis_heater_heaters_immersion",
            "75_legionella_cycle_legionnaires_bacteria",
            "81_domestic_hot_kwh_water",
        ],
        "Floor Heating": [
            "9_floor_heating_underfloor_floors",
            "14_floor_heating_pump_underfloor",
            "88_buffer_floor_heating_tank",
            "90_radiators_floor_heating_underfloor",
            "142_rads_floor_heating_upstairs",
        ],
        "Boilers and Stoves": [
            "3_boiler_boilers_gas_combi",
            "41_boiler_gas_oil_source",
            "44_stove_wood_burner_log",
            "125_lpg_gas_mains_bulk",
        ],
        "Hydrogen": [
            "103_hydrogen_green_natural_gas",
        ],
        "General Heating": [
            "23_heating_systems_resistance_design",
            "28_slab_concrete_floor_cooling",
            "42_defrost_defrosting_defrosts_cycle",
            "79_degrees_temperature_temp_set",
        ],
    },
    "Components": {
        "Pipes and Valves": [
            "8_pipe_pipes_pipework_copper",
            "37_valve_valves_bypass_port",
        ],
        "Tanks and Cylinders": [
            "22_buffer_tank_pump_heat",
            "26_cylinder_unvented_cylinders_telford",
            "57_tank_tanks_loft_litre",
            "64_cylinder_domestic_hot_water",
            "74_buffer_tank_tanks_size",
            "107_tank_temperature_temp_degrees",
            "129_immersion_immersions_3kw_solic",
        ],
        "Pumps and Circulators": [
            "15_pump_pumps_speed_circulation",
            "147_ftc6_ftc_ftc5_controller",
        ],
        "Thermostats and Controllers": [
            "36_thermostat_thermostats_room_control",
            "55_controller_controls_control_controllers",
            "91_settings_setting_parameters_change",
            "141_ecodan_controller_ftc6_cooling",
        ],
        "Radiators and Manifolds": [
            "19_radiators_radiator_output_temperature",
            "34_manifold_manifolds_mixer_blending",
            "40_rads_rad_warm_temp",
        ],
        "Compressors and Refrigerants": [
            "43_compressor_compressors_scroll_crankcase",
            "89_refrigerant_fridge_refrigeration_refrigerants",
            "133_glycol_ethylene_exchanger_propylene",
        ],
        "Fans and Coils": [
            "77_floor_tiles_flooring_floors",
            "86_inverter_driven_inverters_drive",
        ],
        "Miscellaneous Components": [
            "49_flow_rate_rates_meter",
            "50_wiring_cable_wires_wire",
            "65_zones_zone_zoning_single",
            "121_phase_single_3phase_transformer",
            "127_blinds_external_windows_velux",
            "135_warranty_year_manufacturer_item",
            "137_trvs_trv_open_rads",
            "143_batteries_battery_storage_life",
        ],
    },
    "Noise": [
        "4_noise_sound_noisy_quiet",
    ],
    "Installation": [
        "10_installer_installers_install_installation",
        "25_plumber_plumbers_plumbing_electrician",
        "94_vaillant_daikin_installer_schematics",
        "118_manual_manuals_instructions_copy",
    ],
    "Temperature and Control": [
        "13_flow_temp_temperature_temps",
        "36_thermostat_thermostats_room_control",
        "55_controller_controls_control_controllers",
        "79_degrees_temperature_temp_set",
        "91_settings_setting_parameters_change",
        "147_ftc6_ftc_ftc5_controller",
    ],
    "Certification and Incentives": [
        "21_certification_microgeneration_scheme_certificate",
        "51_grant_bus_grants_5k",
        "96_epc_incentive_renewable_rating",
        "136_certification_microgeneration_scheme_certified",
    ],
    "Energy and Efficiency": [
        "52_carbon_co2_emissions_fossil",
        "62_octopus_tariff_cosy_agile",
        "115_efficiency_efficient_inefficient_penalty",
        "108_tariff_tariffs_economy_tou",
        "126_energy_consumption_efficiency_rating",
    ],
    "Design and Planning": [
        "67_planning_lpa_pd_permission",
        "69_size_sizing_oversizing_fit",
        "111_design_designers_designer_professional",
    ],
    "Weather and Climate": [
        "32_winter_weather_cold_rain",
        "85_house_warm_cold_days",
        "110_climate_change_global_scientists",
    ],
    "Calculation and Data": [
        "46_loss_calculations_calcs_spreadsheet",
        "112_spreadsheet_excel_chart_data",
        "56_numbers_figures_maths_calculations",
    ],
    "Water Systems": [
        "57_tank_tanks_loft_litre",
        "64_cylinder_domestic_hot_water",
        "75_legionella_cycle_legionnaires_bacteria",
        "81_domestic_hot_kwh_water",
        "107_tank_temperature_temp_degrees",
        "109_cylinder_reheat_unvented_cylinders",
        "129_immersion_immersions_3kw_solic",
    ],
    "Miscellaneous": [
        "2_thread_post_question_forum",
        "63_build_self_builders_builder",
        "66_garden_kitchen_room_bedroom",
        "104_loops_loop_100m_loopcad",
        "105_2023by_2022by_editeddecember_2021by",
        "119_trees_wood_tree_timber",
        "128_200mm_150mm_100mm_spacing",
        "130_photos_pics_photo_picture",
        "132_problem_issue_issues_problems",
        "124_e7_e10_domestic_boost",
    ],
}


# In[6]:


# import boto3
# import pandas as pd
# from io import StringIO

# s3 = boto3.client('s3', region_name='us-east-1')  # or your preferred region
# bucket = 'asf-public-discourse-home-decarbonisation'
# key = 'data/buildhub/outputs/topic_analysis/buildhub_heat pump_sentence_topics_info.csv'

# obj = s3.get_object(Bucket=bucket, Key=key)
# data = obj['Body'].read().decode('utf-8')
# df = pd.read_csv(StringIO(data))
file_path = "/Users/aidan.kelly/nesta/ASF/asf_public_discourse_home_decarbonisation/asf_public_discourse_home_decarbonisation/analysis/buildhub/files/buildhub_heat pump_sentence_topics_info.csv"
df = pd.read_csv(file_path)
df.head()


# In[2]:


new_dict = {}

for category, topics in categories_dict.items():
    if isinstance(topics, dict):
        new_dict[category] = {}
        for subcategory, subtopics in topics.items():
            new_dict[category][subcategory] = {}
            for topic in subtopics:
                count = df.loc[df["Name"] == topic, "updated_count"].values[0]
                new_dict[category][subcategory][topic] = count
    else:
        new_dict[category] = {}
        for topic in topics:
            count = df.loc[df["Name"] == topic, "updated_count"].values[0]
            new_dict[category][topic] = count


# In[ ]:


print(new_dict)


# In[ ]:


import matplotlib.pyplot as plt

aggregated_data = {
    "Cost": sum(new_dict["Cost"].values()),
    "Power": sum(new_dict["Power"].values()),
    "Heat Pumps": sum(new_dict["Heating Systems"]["Heat Pumps"].values()),
    "Water Heating": sum(new_dict["Heating Systems"]["Water Heating"].values()),
    "Floor Heating": sum(new_dict["Heating Systems"]["Floor Heating"].values()),
    "Boilers and Stoves": sum(
        new_dict["Heating Systems"]["Boilers and Stoves"].values()
    ),
    "Hydrogen": sum(new_dict["Heating Systems"]["Hydrogen"].values()),
    "General Heating": sum(new_dict["Heating Systems"]["General Heating"].values()),
    "Pipes and Valves": sum(new_dict["Components"]["Pipes and Valves"].values()),
    "Tanks and Cylinders": sum(new_dict["Components"]["Tanks and Cylinders"].values()),
    "Pumps and Circulators": sum(
        new_dict["Components"]["Pumps and Circulators"].values()
    ),
    "Thermostats and Controllers": sum(
        new_dict["Components"]["Thermostats and Controllers"].values()
    ),
    "Radiators and Manifolds": sum(
        new_dict["Components"]["Radiators and Manifolds"].values()
    ),
    "Compressors and Refrigerants": sum(
        new_dict["Components"]["Compressors and Refrigerants"].values()
    ),
    "Fans and Coils": sum(new_dict["Components"]["Fans and Coils"].values()),
    "Noise": sum(new_dict["Components"]["Noise"].values()),
    "Miscellaneous Components": sum(
        new_dict["Components"]["Miscellaneous Components"].values()
    ),
    "Installation": sum(new_dict["Installation"].values()),
    "Temperature and Control": sum(new_dict["Temperature and Control"].values()),
    "Certification and Incentives": sum(
        new_dict["Certification and Incentives"].values()
    ),
    "Energy and Efficiency": sum(new_dict["Energy and Efficiency"].values()),
    "Design and Planning": sum(new_dict["Design and Planning"].values()),
    "Weather and Climate": sum(new_dict["Weather and Climate"].values()),
    "Calculation and Data": sum(new_dict["Calculation and Data"].values()),
    "Water Systems": sum(new_dict["Water Systems"].values()),
    "Miscellaneous": sum(new_dict["Miscellaneous"].values()),
}
# Sort the data by values for better visualization
sorted_aggregated_data = dict(
    sorted(aggregated_data.items(), key=lambda item: item[1], reverse=True)
)

# Plot the aggregated data
plt.figure(figsize=(12, 8))
plt.barh(
    list(sorted_aggregated_data.keys()),
    list(sorted_aggregated_data.values()),
    color="#0000FF",
)
plt.ylabel("Categories")
plt.xlabel("Number of Sentences")
plt.title("Comparison of categories based on the number of sentences")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[31]:


import pandas as pd

file_path = "./files/buildhub_heat pump_sentence_docs_info.csv"
df = pd.read_csv(file_path)
cluster = categories_dict["Noise"]
filtered_df = df[df["Name"].isin(cluster)]


print(filtered_df["sentences"])


with open("mcs_installation_sentences.txt", "w") as f:
    for sentence in filtered_df["sentences"]:
        if "microgeneration certification" in sentence:
            f.write(sentence + "\n-----\n")


# In[32]:


from asf_public_discourse_home_decarbonisation.utils.preprocessing_utils import (
    preprocess_text,
    calculate_ngram_threshold,
    process_ngrams,
)

from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    plot_word_cloud,
    plot_top_ngrams,
)
from nltk import FreqDist

import pandas as pd

file_path = "./files/buildhub_heat pump_sentence_docs_info.csv"
df = pd.read_csv(file_path)

new_stopwords = [
    "would",
    "hours",
    "hour",
    "minute",
    "minutes",
    "ago",
    "dan",
    "(presumably",
    "looks",
    "like",
    "need",
    "ap50",
    ".3page",
    "fraser",
    "lamont",
    "got",
    "bit",
    "sure",
    "steamytea",
    "could",
    "get",
    "still",
    "october",
    "6",
    "2013",
    "january",
    "2016",
    "moderator",
    "thisreport",
    "pretty",
    "&",
    "-",
]


def preprocess_for_tokens_and_word_freq_dist(df, category_name, new_stopwords):
    cluster = categories_dict[category_name]
    filtered_df = df[df["Name"].isin(cluster)]
    filtered_df.rename(columns={"sentences": "text"}, inplace=True)
    filtered_df["text"] = filtered_df["text"].str.rstrip(".!?")
    filtered_df["text"] = filtered_df["text"].str.replace(",", "")
    filtered_tokens = preprocess_text(filtered_df, new_stopwords)
    word_freq_dist = FreqDist(filtered_tokens)
    return filtered_tokens, word_freq_dist


categories_of_interest = ["Weather and Climate", "Cost", "Noise"]

tokens = {}
freq_dist = {}
for category in categories_of_interest:
    tokens[category], freq_dist[category] = preprocess_for_tokens_and_word_freq_dist(
        df, category, new_stopwords
    )
    plot_word_cloud(freq_dist[category], "./", threshold=75)
    raw_bigram_freq_dist, bigram_threshold = calculate_ngram_threshold(
        tokens[category], 2, 0.0008
    )
    raw_trigram_freq_dist, trigram_threshold = calculate_ngram_threshold(
        tokens[category], 3, 0.0004
    )
    weather_bigram_freq_dist = process_ngrams(raw_bigram_freq_dist, bigram_threshold)
    weather_trigram_freq_dist = process_ngrams(raw_trigram_freq_dist, trigram_threshold)
    plot_top_ngrams(
        raw_bigram_freq_dist,
        n=20,
        ngram_type="Bigram",
        color=NESTA_COLOURS[0],
        output_path="./",
    )
    plot_top_ngrams(
        raw_trigram_freq_dist,
        n=20,
        ngram_type="Trigram",
        color=NESTA_COLOURS[0],
        output_path="./",
    )


# print(noise_freq_dist)
# plot_word_cloud(noise_freq_dist, "./", threshold=75)
