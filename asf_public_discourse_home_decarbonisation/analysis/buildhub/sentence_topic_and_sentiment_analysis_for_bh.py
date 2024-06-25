#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


bh_data = get_bh_data(category="all", collection_date="24_05_21")
print("Date Range:", bh_data["date"].min(), "to", bh_data["date"].max())


# In[3]:


key_terms = ["heat pump", "boiler"]
# Preprocess data
bh_data = preprocess_data_for_linechart_over_time(bh_data, key_terms=key_terms)
# Resample and calculate averages
# bh_data.head()
bh_data_monthly = resample_and_calculate_averages_for_linechart(
    bh_data, key_terms=key_terms, cadence_of_aggregation="M", window=12
)


# In[4]:


bh_data_original_post = bh_data[bh_data["is_original_post"] == 1]
bh_data_original_post["mentions_heat_pump"].sum()


# In[5]:


key_terms_colour = {"heat pump": "#97D9E3", "boiler": "#0F294A"}
plot_mentions_line_chart(bh_data_monthly, key_terms_colour, plot_type="discrete")


# In[6]:


categories_dict = {
    "Cost": {
        "Heat Pump Cost": {
            "HP cost - 1": "16_price_prices_pay_money",
            "HP cost - 2": "30_10k_price_cost_paid",
            "HP cost - ASHP": "53_cost_pump_source_air",
        },
        "Heat Pump Purchase Benefits": {
            "HP cost - VAT reclaim": "97_vat_reclaim_plus_zero",
            "HP cost - Credit Card": "61_hp_hps_card_cheaper",
        },
        "Heat Pump Quote": {"HP cost - Quotes": "139_quote_quotes_quoting_got"},
        "Running Costs": {
            "Running costs - electricity": "145_cost_month_year_electricity",
        },
        "Oil Cost": {
            "Oil Cost": "134_oil_litre_year_price",
        },
    },
    "PV and Solar Panels": {"PV and Solar Panels": "0_pv_solar_panels_excess"},
    "Insulation": {
        "Insulation": "5_insulation_walls_insulated_cavity",
        "Glazed Windows": "93_windows_glazed_window_glazing",
        "External Blinds and Window Coverings": "127_blinds_external_windows_velux",
    },
    "Power": {
        "Power Usage - 1": "7_kwh_kw_12kw_power",
        "Power Usage - 2": "18_kw_pump_kwh_source",
        "Power Usage - Average": "38_kw_kwh_average_day",
        "Standby Power Consumption": "138_standby_consumption_power_ecodan",
    },
    "Heat Pump Types": {
        "Air Source Heat Pump": {
            "Air Source Heat Pump": "1_source_air_pump_heat",
        },
        "Ground Source Heat Pump": {
            "Ground Source Heat Pump": "45_ground_source_pump_borehole",
        },
        "General Heat Pump": {
            "Heat Pump Operation": "31_pumps_heat_pump_work",
            "Monobloc Units": "68_split_monoblock_monobloc_unit",
            "Pump Flow": "29_pump_flow_source_heat",
        },
        "Heat Pump Brands": {
            "Ecodan Specifics": "144_ecodan_ecodans_r32_know",
            "Brands of Heat Pumps": "33_mitsubishi_samsung_lg_panasonic",
        },
    },
    "Water Heating": {
        "Domestic Water Heating": "6_hot_domestic_water_heating",
        "Hot Water Cylinders": "109_cylinder_reheat_unvented_cylinders",
        "Willis Immersion Heaters": "146_willis_heater_heaters_immersion",
        "Legionella Prevention": "75_legionella_cycle_legionnaires_bacteria",
        "Domestic Hot Water": "76_hot_domestic_water_source",
        "Hot Water Usage": "81_domestic_hot_kwh_water",
        "Annual Checks": "131_g3_inspection_unvented_annual",
        "Immersion Heaters - 1": "129_immersion_immersions_3kw_solic",
        "Immersion Heaters - 2": "84_immersion_heater_heaters_switch",
    },
    "Underfloor Heating & Radiators": {
        "Underfloor Heating": {
            "Underfloor Heating - System": "9_floor_heating_underfloor_floors",
            "Underfloor Heating - Pump": "14_floor_heating_pump_underfloor",
            "Underfloor Heating - Buffer Tank": "88_buffer_floor_heating_tank",
            "Underfloor Heating - Tiles": "77_floor_tiles_flooring_floors",
        },
        "Underfloor Heating & Radiators": {
            "Underfloor Heating & Radiators - 1": "90_radiators_floor_heating_underfloor",
            "Underfloor Heating & Radiators - 2": "142_rads_floor_heating_upstairs",
        },
    },
    "Boilers and Stoves": {
        "Gas Boilers": "3_boiler_boilers_gas_combi",
        "Oil and Gas Boilers": "41_boiler_gas_oil_source",
        "Wood Burning Stoves": "44_stove_wood_burner_log",
        "LPG": "125_lpg_gas_mains_bulk",
        "Gas Mains": "54_gas_mains_fuel_standing",
    },
    "Hydrogen": {"Hydrogen": "103_hydrogen_green_natural_gas"},
    "General Heating": {
        "Heating Systems Design": "23_heating_systems_resistance_design",
        "Concrete Floor Cooling": "28_slab_concrete_floor_cooling",
        "Defrost Cycles": "42_defrost_defrosting_defrosts_cycle",
        "Heating Different Bedooms": "117_upstairs_downstairs_heating_bedrooms",
    },
    "Pipes and Valves": {
        "Pipework": "8_pipe_pipes_pipework_copper",
        "Valves": "37_valve_valves_bypass_port",
        "Thermostatic Radiator Valves": "137_trvs_trv_open_rads",
    },
    "Tanks and Cylinders": {
        "Buffer Tanks - 1": "22_buffer_tank_pump_heat",
        "Buffer Tanks - 2": "72_buffer_buffers_volume_volumiser",
        "Buffer Tanks - 3": "74_buffer_tank_tanks_size",
        "Unvented Cylinders": "26_cylinder_unvented_cylinders_telford",
        "Loft Tanks": "57_tank_tanks_loft_litre",
        "Domestic Hot Water Cylinders": "64_cylinder_domestic_hot_water",
        "Tank Temperatures": "107_tank_temperature_temp_degrees",
    },
    "Pumps and Circulators": {
        "Circulation Pumps": "15_pump_pumps_speed_circulation",
    },
    "Radiators and Manifolds": {
        "Radiators": {
            "Radiator Output": "19_radiators_radiator_output_temperature",
            "Warm Radiators": "40_rads_rad_warm_temp",
            "Radiators for Heat Pumps": "106_radiators_radiator_pump_source",
        },
        "Manifolds": {
            "Manifold Systems": "34_manifold_manifolds_mixer_blending",
        },
    },
    "Compressors and Refrigerants": {
        "Compressors": "43_compressor_compressors_scroll_crankcase",
        "Propane Refrigerants": "82_r290_r32_propane_refrigerant",
        "Fridge Refrigerants": "89_refrigerant_fridge_refrigeration_refrigerants",
        "Glycol Usage": "133_glycol_ethylene_exchanger_propylene",
    },
    "Fans and Coils": {
        "Fan coils": "78_fan_coil_coils_fans",
        "Coil": "80_coil_coils_cylinder_3m2",
        "Inverter Drives": "86_inverter_driven_inverters_drive",
    },
    "Miscellaneous Components": {
        "Flow Meters": "49_flow_rate_rates_meter",
        "Wiring": "50_wiring_cable_wires_wire",
        "Zoning Systems": "65_zones_zone_zoning_single",
        "Transformers": "121_phase_single_3phase_transformer",
        "Warranties": "135_warranty_year_manufacturer_item",
    },
    "Battery Storage": {
        "Battery Storage": "143_batteries_battery_storage_life",
        "Sunamp": "35_sunamp_sunamps_charge_amp",
    },
    "Noise": {"Noise Levels": "4_noise_sound_noisy_quiet"},
    "Installation and Installers": {
        "Installers": "10_installer_installers_install_installation",
        "Plumbing and Electrical": "25_plumber_plumbers_plumbing_electrician",
        "Engineer Training": "92_engineer_engineering_training_engineers",
        "Installer Schematics": "94_vaillant_daikin_installer_schematics",
        "Manuals and Instructions": "118_manual_manuals_instructions_copy",
    },
    "Temperature and Control": {
        "Flow Temperature": "13_flow_temp_temperature_temps",
        "Room Thermostats": "36_thermostat_thermostats_room_control",
        "Control Systems": "55_controller_controls_control_controllers",
        "Settings and Parameters": "91_settings_setting_parameters_change",
        "Ecodan Controllers": "141_ecodan_controller_ftc6_cooling",
        "Heat Pump Temperatures": "70_hp_hps_temp_temps",
        "Control Systems": "55_controller_controls_control_controllers",
        "Temperature Settings": "79_degrees_temperature_temp_set",
        "Settings and Parameters": "91_settings_setting_parameters_change",
        "FTC Controllers": "147_ftc6_ftc_ftc5_controller",
    },
    "Certification and Incentives": {
        "EPC Ratings": {
            "EPC Ratings": "96_epc_incentive_renewable_rating",
            "SAP Ratings": "71_sap_rating_report_score",
        },
        "Microgeneration Certification Scheme": {
            "Microgeneration Certification Scheme - 1": "21_certification_microgeneration_scheme_certificate",
            "Microgeneration Certification Scheme - 2": "136_certification_microgeneration_scheme_certified",
        },
        "BUS Grants": {
            "BUS Grants": "51_grant_bus_grants_5k",
        },
        "Domestic RHI": {
            "Domestic RHI": "27_incentive_renewable_scheme_payments",
        },
    },
    "Carbon Emissions": {"Carbon Emissions": "52_carbon_co2_emissions_fossil"},
    "Energy Efficiency": {
        "Efficiency Penalty": {
            "Efficiency Penalty": "115_efficiency_efficient_inefficient_penalty",
            "Cycling": "20_cycling_short_hours_minutes",
        },
        "COP and efficiency": {
            "Energy Consumption": "126_energy_consumption_efficiency_rating",
            "SCOP": "12_cop_scop_35_flow",
            "Weather Compensation": "48_compensation_weather_comp_curve",
        },
    },
    "Tariffs": {
        "Energy Tariffs": "62_octopus_tariff_cosy_agile",
        "Economy Tariffs": "108_tariff_tariffs_economy_tou",
        "Domestic Boost": "124_e7_e10_domestic_boost",
        "Cheap Rates": "140_e7_e10_rate_cheap",
    },
    "Smart Meters": {"Smart Meters": "39_meter_smart_meters_metering"},
    "Planning": {
        "Planning Permission": "67_planning_lpa_pd_permission",
    },
    "Design and Sizing": {
        "System Sizing": "69_size_sizing_oversizing_fit",
        "ASHP Sizing": "123_sizing_size_source_air",
        "Design Professionals": "111_design_designers_designer_professional",
    },
    "Cold Weather": {
        "Winter Weather": "32_winter_weather_cold_rain",
        "House Warmth": "85_house_warm_cold_days",
    },
    "Calculation and Data": {
        "Heat Loss Calculations": "46_loss_calculations_calcs_spreadsheet",
        "Data and Charts": "112_spreadsheet_excel_chart_data",
        "Building Loss Calculation": "114_loss_calculation_spreadsheet_building",
        "Maths and Figures": "56_numbers_figures_maths_calculations",
        "Units": "122_units_unit_si_unitower",
    },
    "Cooling": {
        "A/C": "47_air_aircon_a2w_units",
        "Cooling": "98_cooling_mode_active_cool",
        "Cooler Rooms": "100_rooms_room_bedrooms_cooler",
        "Cooling MVHR": "101_mvhr_ventilation_cooling_fresh",
    },
    "Other": {
        "Shower and Bath": "17_shower_showers_bath_baths",
        "Duct": "58_duct_ducts_ducting_ducted",
        "Decisions": "59_options_decision_doing_idea",
        "Concrete": "60_screed_slab_concrete_pir",
        "Property Renovation": "11_house_houses_property_renovation",
        "Forum Threads": "2_thread_post_question_forum",
        "Self Builders": "63_build_self_builders_builder",
        "Systems set-up": "24_systems_setup_replace_replacement",
        "Room Usage": "66_garden_kitchen_room_bedroom",
        "Loop Systems": "104_loops_loop_100m_loopcad",
        "Edited Posts": "105_2023by_2022by_editeddecember_2021by",
        "Trees and Timber": "119_trees_wood_tree_timber",
        "Spacing Measurements": "128_200mm_150mm_100mm_spacing",
        "Photos": "130_photos_pics_photo_picture",
        "Problems and Issues": "132_problem_issue_issues_problems",
        "Climate Change": "110_climate_change_global_scientists",
        "Electricity on the Grid": "73_electricity_energy_grid_electric",
        "England and Scotland": "83_scotland_cornwall_england_north",
        "Science": "87_science_physics_scientific_scientist",
        "Condensation": "95_condensate_condensation_drain_condensing",
        "Frost Protection": "102_frost_frosting_protection_mode",
        "Towel Rail": "113_towel_rails_towels_rail",
        "Customer Sales": "116_customer_company_sales_customers",
        "Anti-freeze": "120_antifreeze_inhibitor_freeze_anti",
        "Roofing": "99_roof_flat_roofing_facing",
    },
}


# In[7]:


# import boto3
import pandas as pd

# from io import StringIO

# s3 = boto3.client('s3', region_name='us-east-1')  # or your preferred region
# bucket = 'asf-public-discourse-home-decarbonisation'
# key = 'data/buildhub/outputs/topic_analysis/buildhub_heat pump_sentence_topics_info.csv'

# obj = s3.get_object(Bucket=bucket, Key=key)
# data = obj['Body'].read().decode('utf-8')
# df = pd.read_csv(StringIO(data))
topics_info_file_path = "./files/buildhub_heat pump_sentence_topics_info.csv"
topics_info = pd.read_csv(topics_info_file_path)
topics_info.head()


# In[8]:


dict_with_counts = {}

list_of_dicts_with_subsubcategories = [
    "Cost",
    "Heat Pump Types",
    "Underfloor Heating & Radiators",
    "Radiators and Manifolds",
    "Certification and Incentives",
    "Energy Efficiency",
]


for category, subcategories in categories_dict.items():
    print(f"Category: {category}")
    dict_with_counts[category] = {}
    if category in list_of_dicts_with_subsubcategories:
        for subcategory, topics in subcategories.items():
            print(f"  Subcategory: {subcategory}")
            dict_with_counts[category][subcategory] = {}
            for topic, value in topics.items():
                print(f"  Topic: {topic}, Value: {value}")
                count = topics_info.loc[
                    topics_info["Name"] == value, "updated_count"
                ].values[0]
                dict_with_counts[category][subcategory][topic] = count
    else:
        for topic, value in subcategories.items():
            # print(f"  Topic: {topic}, Value: {value}")
            count = topics_info.loc[
                topics_info["Name"] == value, "updated_count"
            ].values[0]
            dict_with_counts[category][topic] = count


# In[9]:


print(dict_with_counts)


# In[10]:


import matplotlib.pyplot as plt


aggregated_data = {}

for category, subcategories in dict_with_counts.items():
    if isinstance(subcategories, dict):
        category_sum = 0
        for subcategory, topics in subcategories.items():
            if isinstance(topics, dict):
                subcategory_sum = sum(topics.values())
                category_sum += subcategory_sum
            else:
                category_sum += topics
        aggregated_data[category] = category_sum
    else:
        aggregated_data[category] = subcategories

print(aggregated_data)


# Sort the data by values for better visualization
sorted_aggregated_data = dict(
    sorted(aggregated_data.items(), key=lambda item: item[1], reverse=False)
)


# Get rid of the other category
del sorted_aggregated_data["Other"]
# Plot the aggregated data
plt.figure(figsize=(10, 12))
plt.barh(
    list(sorted_aggregated_data.keys()),
    list(sorted_aggregated_data.values()),
    color="#0000FF",
)
# plt.ylabel('Categories')
plt.xlabel("Number of Sentences")
# plt.title('Comparison of categories based on the number of sentences')
plt.xticks(rotation=45, ha="right")
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# In[11]:


granular_category_data = {}


print(dict_with_counts)
for category, subcategories in dict_with_counts.items():
    if category not in list_of_dicts_with_subsubcategories:
        category_sum = sum(
            [
                sum(topics.values()) if isinstance(topics, dict) else topics
                for topics in subcategories.values()
            ]
        )
        granular_category_data[category] = category_sum
    else:
        print(f"Category: {category}")
        for subcategory, topics in subcategories.items():
            # print(f"Subcategory: {subcategory}")
            # print(f"Topics: {topics}")
            if isinstance(topics, dict):
                subcategory_sum = sum(topics.values())
                granular_category_data[subcategory] = subcategory_sum
            else:
                granular_category_data[subcategory] = topics

print(granular_category_data)
sorted_granular_category_data = dict(
    sorted(granular_category_data.items(), key=lambda item: item[1], reverse=False)
)
# Get rid of the other category
del sorted_granular_category_data["Other"]

selected_items = {
    key: sorted_granular_category_data[key]
    for key in [
        "Microgeneration Certification Scheme",
        "EPC Ratings",
        "BUS Grants",
        "Domestic RHI",
    ]
}
# Order the selected_items dictionary by its values
ordered_selected_items = dict(sorted(selected_items.items(), key=lambda item: item[1]))

# Plot the aggregated data
plt.figure(figsize=(14, 8))
plt.barh(
    list(ordered_selected_items.keys()),
    list(ordered_selected_items.values()),
    color="#0000FF",
)
plt.ylabel("Categories")
plt.xlabel("Number of Sentences")
# plt.title('Comparison of categories based on the number of sentences')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# In[12]:


import pandas as pd

file_path = "./files/buildhub_heat pump_sentence_docs_info.csv"
docs_info = pd.read_csv(file_path)
cluster = categories_dict["Noise"].values()
filtered_docs_info = docs_info[docs_info["Name"].isin(cluster)]


# filtered_list_of_dicts =

with open("Noise_cluster.txt", "w") as f:
    for sentence in filtered_docs_info["sentences"]:
        # if "microgeneration certification" in sentence:
        f.write(sentence + "\n-----\n")
# with open('mcs_installation_sentences.txt', 'w') as f:
#    for sentence in filtered_docs_info['sentences']:
#        if "microgeneration certification" in sentence:
#            f.write(sentence + "\n-----\n")


# In[13]:


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

from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    NESTA_COLOURS,
)

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


def preprocess_for_tokens_and_word_freq_dist(docs_info, category_name, new_stopwords):
    if category_name in list_of_dicts_with_subsubcategories:
        cluster = flatten_dict_values(category_name)
    else:
        cluster = categories_dict[category_name].values()
    filtered_docs_info = docs_info[docs_info["Name"].isin(cluster)]
    filtered_docs_info.rename(columns={"sentences": "text"}, inplace=True)
    filtered_docs_info["text"] = filtered_docs_info["text"].str.rstrip(".!?")
    filtered_docs_info["text"] = filtered_docs_info["text"].str.replace(",", "")
    filtered_tokens = preprocess_text(filtered_docs_info, new_stopwords)
    word_freq_dist = FreqDist(filtered_tokens)
    return filtered_tokens, word_freq_dist


categories_of_interest = ["Cold Weather", "Cost", "Noise"]


def flatten_dict_values(category_name):
    flattened_values = []
    for dict in categories_dict[category_name].values():
        for value in dict.values():
            flattened_values.append(value)
    return flattened_values


tokens = {}
freq_dist = {}
for category in categories_of_interest:
    tokens[category], freq_dist[category] = preprocess_for_tokens_and_word_freq_dist(
        docs_info, category, new_stopwords
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


# In[14]:


sentiment_data = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/sentiment/buildhub_heat pump_sentence_topics_sentiment.csv"
)
sentences_data = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/topic_analysis/buildhub_heat pump_sentences_data.csv"
)


# In[15]:


len(sentiment_data[sentiment_data["score"] > 0.75]) / len(sentiment_data) * 100


# In[15]:


docs_info = docs_info.merge(sentiment_data, left_on="Document", right_on="text")


# In[16]:


docs_info


# In[17]:


topic_sentiment = (
    docs_info.groupby(["Name", "sentiment"]).nunique()["Document"].unstack().fillna(0)
)


# In[18]:


topic_sentiment.head()


# In[19]:


topic_sentiment["proportion_negative"] = topic_sentiment["negative"] / (
    topic_sentiment["negative"]
    + topic_sentiment["positive"]
    + topic_sentiment["neutral"]
)


# In[20]:


topic_sentiment["ratio_to_negative"] = (
    topic_sentiment["positive"] + topic_sentiment["neutral"]
) / (topic_sentiment["negative"])


# In[21]:


renamed_topic_sentiment = topic_sentiment.copy()
index_mapping = {}
# Iterate over each row in topic_sentiment

# Create a set of all topics
all_topics = set(renamed_topic_sentiment.index)

index_mapping = {}
# Iterate over each row in topic_sentiment
for index, row in renamed_topic_sentiment.iterrows():
    for key, value in categories_dict.items():
        if key not in list_of_dicts_with_subsubcategories:
            if index in value.values():
                for key, value in value.items():
                    if value == index:
                        # Rename the index
                        index_mapping[index] = key
                        # Remove the topic from all_topics
                        # print(index)
                        all_topics.remove(index)
        else:
            for subcategory, topics in value.items():
                if index in topics.values():
                    for key, value in topics.items():
                        if value == index:
                            # Rename the index
                            index_mapping[index] = key
                            # Remove the topic from all_topics
                            # print(index)
                            all_topics.remove(index)


# Print out the topics that are not mapped
for topic in all_topics:
    print(topic)


renamed_topic_sentiment.rename(index=index_mapping, inplace=True)

# Filter the DataFrame
values = index_mapping.values()
# print(values)
filtered_renamed_topic_sentiment = renamed_topic_sentiment[
    renamed_topic_sentiment.index.isin(values)
]
# filtered_renamed_topic_sentiment
renamed_topic_sentiment


# In[22]:


filtered_renamed_topic_sentiment.sort_values(by="proportion_negative", ascending=False)
filtered_renamed_topic_sentiment


# In[23]:


filtered_renamed_topic_sentiment = filtered_renamed_topic_sentiment[
    ["negative", "neutral", "positive"]
]
filtered_renamed_topic_sentiment = (
    filtered_renamed_topic_sentiment.div(
        filtered_renamed_topic_sentiment.sum(axis=1), axis=0
    )
    * 100
)


# In[24]:


filtered_renamed_topic_sentiment.sort_values("negative", ascending=False, inplace=True)


# Function to replace numbers in the index with blank spaces
def replace_numbers_with_blank(index):
    return [
        "".join([" " if char.isdigit() else char for char in str(idx)]) for idx in index
    ]


# Apply the function to the index
# topic_sentiment.index = replace_numbers_with_blank(topic_sentiment.index)
# topic_sentiment.index = topic_sentiment.index.str.replace('_', ' ')
top_30_negative = filtered_renamed_topic_sentiment.head(30)


# In[25]:


import matplotlib.pyplot as plt

top_30_negative.plot(
    kind="barh", stacked=True, color=["red", "grey", "green"], figsize=(20, 16)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")


# In[26]:


# Sort by positive sentiment and take the top 30
top_30_positive = filtered_renamed_topic_sentiment.sort_values(
    by="positive", ascending=False
).head(30)

# Rearrange the columns
top_30_positive = top_30_positive[["positive", "neutral", "negative"]]

# Plot the top 30 most positive sentiments with green on the left and red on the right
top_30_positive.plot(
    kind="barh", stacked=True, color=["green", "grey", "red"], figsize=(20, 16)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")

# Get the 'Name' values of the top 30 positive and negative topics
positive_topics = set(top_30_positive.index)
negative_topics = set(top_30_negative.index)

# Find the common topics
common_topics = positive_topics & negative_topics
common_topics = list(common_topics)


# In[27]:


# Get the 'neutral' values of the common topics
polarised_topics = filtered_renamed_topic_sentiment.loc[common_topics]

polarised_topics = polarised_topics.sort_values(by="neutral", ascending=False)
print(polarised_topics)
polarised_topics.plot(
    kind="barh", stacked=True, color=["red", "grey", "green"], figsize=(12, 8)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")


# In[ ]:


# In[42]:


rows_list = []
for category, subcategories in categories_dict.items():
    if category in list_of_dicts_with_subsubcategories:
        for subcategory, topics in subcategories.items():
            for topic, value in topics.items():
                rows = docs_info[docs_info["Name"] == value].copy()
                rows["Name"] = category
                print(rows)
                rows_list.append(rows)
    else:
        for topic, value in subcategories.items():
            rows = docs_info[docs_info["Name"] == value].copy()
            rows["Name"] = category
            rows_list.append(rows)

broad_docs_info = pd.concat(rows_list)
broad_topic_sentiment = (
    broad_docs_info.groupby(["Name", "sentiment"])
    .nunique()["Document"]
    .unstack()
    .fillna(0)
)
broad_topic_sentiment["proportion_negative"] = broad_topic_sentiment["negative"] / (
    broad_topic_sentiment["negative"]
    + broad_topic_sentiment["positive"]
    + broad_topic_sentiment["neutral"]
)
broad_topic_sentiment["ratio_to_negative"] = (
    broad_topic_sentiment["positive"] + broad_topic_sentiment["neutral"]
) / (broad_topic_sentiment["negative"])

# broad_topic_sentiment.sort_values(by='proportion_negative', ascending=True)
broad_topic_sentiment
broad_topic_sentiment = broad_topic_sentiment[["negative", "neutral", "positive"]]
broad_topic_sentiment = (
    broad_topic_sentiment.div(broad_topic_sentiment.sum(axis=1), axis=0) * 100
)

broad_topic_sentiment.sort_values("negative", ascending=True, inplace=True)
# Sort the DataFrame in descending order by the "negative" column and select the first 20 rows
# Sort the DataFrame in descending order by the "negative" column, select the first 20 rows, then sort in ascending order
top_20_negative = (
    broad_topic_sentiment.sort_values("negative", ascending=False)
    .head(20)
    .sort_values("negative")
)
# Plot the DataFrame
top_20_negative.plot(
    kind="barh", stacked=True, color=["red", "grey", "green"], figsize=(12, 8)
)
plt.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")


# In[ ]:


# ### SENTIMENT OVER TIME

# In[68]:


sentences_data["year"] = pd.to_datetime(sentences_data["date"]).dt.to_period("Y")
sentences_data


# In[69]:


sentiment_over_time = sentences_data[["sentences", "id", "year", "month"]]
sentiment_over_time


# In[70]:


sentiment_over_time = sentiment_over_time.merge(
    sentiment_data, left_on="sentences", right_on="text"
)
sentiment_over_time


# In[71]:


sentiment_over_time["sentiment_number"] = sentiment_over_time["sentiment"].map(
    {"negative": -1, "neutral": 0, "positive": 1}
)


# In[72]:


aggregated_docs_info = docs_info.copy()
for category, subcategories in categories_dict.items():
    if category in list_of_dicts_with_subsubcategories:
        for subcategory, topics in subcategories.items():
            for topic, value in topics.items():
                aggregated_docs_info.loc[
                    aggregated_docs_info["Name"] == value, "Name"
                ] = category
    else:
        for topic, value in subcategories.items():
            aggregated_docs_info.loc[aggregated_docs_info["Name"] == value, "Name"] = (
                category
            )


aggregated_docs_info
# aggregated_docs_info.to_csv("aggregated_docs_info.csv", index=False)


# In[73]:


sentiment_over_time = sentiment_over_time.merge(
    aggregated_docs_info[["Document", "Topic", "Name"]],
    left_on="sentences",
    right_on="Document",
)
sentiment_over_time


# In[74]:


# Filter out rows where 'Name' is not the specific name
sentiment_over_time_noise = sentiment_over_time[sentiment_over_time["Name"] == "Noise"]
sentiment_over_time_noise_2020 = sentiment_over_time_noise[
    sentiment_over_time_noise["month"] == "2020-08"
]
sentiment_over_time_noise_2020_negative = sentiment_over_time_noise_2020[
    sentiment_over_time_noise_2020["sentiment"] == "negative"
]
sentiment_over_time_noise_2020_negative_sentences = (
    sentiment_over_time_noise_2020_negative["sentences"]
)
sentiment_over_time_noise_2020_negative_sentences
covid_keywords = ["covid", "pandemic", "lockdown", "quarantine", "virus", "coronavirus"]

with open("Noise_2020_august_cluster.txt", "w") as f:
    for sentence in sentiment_over_time_noise_2020_negative_sentences:
        # if "microgeneration certification" in sentence:
        f.write(sentence + "\n-----\n")


# In[75]:


# print(sentiment_over_time_grants.head())
sentiment_over_time = (
    sentiment_over_time.groupby(["Name", "year"])["sentiment"].value_counts().unstack()
)

# Calculate the total sentiments
sentiment_over_time["total"] = (
    sentiment_over_time["positive"]
    + sentiment_over_time["neutral"]
    + sentiment_over_time["negative"]
)

# Calculate the proportion of negative sentiments
sentiment_over_time["proportion_negative"] = (
    sentiment_over_time["negative"] / sentiment_over_time["total"]
)


# In[77]:


sentiment_over_time
# Calculate the mean of 'proportion_negative' for each year
mean_proportion_negative = sentiment_over_time.groupby("year").mean().reset_index()

mean_proportion_negative["Name"] = "All Topics"

# Reorder columns if necessary
mean_proportion_negative = mean_proportion_negative[
    ["Name", "year", "negative", "neutral", "positive", "total", "proportion_negative"]
]
# Append the mean values to the original DataFrame
mean_proportion_negative

# Create a new DataFrame for 'All Topics'
# all_topics = pd.DataFrame({'Name': 'All Topics', 'Year': mean_proportion_negative.index, 'proportion_negative': mean_proportion_negative.values})


# sentiment_over_time


# In[78]:


# sentiment_over_time
sentiment_over_time.reset_index(inplace=True)
sentiment_over_time = pd.concat([sentiment_over_time, mean_proportion_negative])
sentiment_over_time


# In[79]:


topics_to_select = [
    "Noise",
    "Boilers and Stoves",
    "Cost",
    "All Topics",
    "Certification and Incentives",
]
selected_topics = sentiment_over_time[
    sentiment_over_time["Name"].isin(topics_to_select)
]
# Convert 'year' to integer type
# Convert 'year' to datetime type
selected_topics["year"] = selected_topics["year"].dt.to_timestamp()

# Filter out rows where 'year' is before 2020
selected_topics = selected_topics[selected_topics["year"].dt.year >= 2020]
selected_topics_noise = selected_topics[selected_topics["Name"] == "Noise"]
selected_topics_noise.to_csv("selected_topics_noise.csv", index=False)

import matplotlib.pyplot as plt

# Create a figure and a set of subplots
# fig, ax = plt.subplots()

fig, ax = plt.subplots(figsize=(10, 6))
# Loop through each topic and plot 'proportion_negative' over the years
color_dict = {
    "Cost": "pink",
    "Noise": "grey",
    "Boilers and Stoves": "plum",
    "All Topics": "Black",
    "Certification and Incentives": "gold",
}
for name, group in selected_topics.groupby("Name"):
    # group['year'] = group['year'].dt.to_timestamp()
    ax.plot(group["year"], group["total"], label=name, color=color_dict[name])

# Add a legend
ax.legend(fontsize="small")

# Add labels and a title
ax.set_xlabel("Date")
ax.set_ylabel("Number of sentences")
ax.grid(True, axis="y", alpha=0.3)
# Set y-axis to start at 0
ax.set_ylim(bottom=0)

# Make x-axis labels vertical
plt.xticks(rotation="vertical")

plt.show()


# In[80]:


import matplotlib.pyplot as plt

# Create a figure and a set of subplots
# fig, ax = plt.subplots()

fig, ax = plt.subplots(figsize=(10, 6))
# Loop through each topic and plot 'proportion_negative' over the years
color_dict = {
    "Cost": "pink",
    "Noise": "grey",
    "Boilers and Stoves": "plum",
    "All Topics": "Black",
    "Certification and Incentives": "gold",
}
for name, group in selected_topics.groupby("Name"):
    # group['year'] = group['year'].dt.to_timestamp()
    ax.plot(
        group["year"], group["proportion_negative"], label=name, color=color_dict[name]
    )

# Add a legend
ax.legend(fontsize="small")

# Add labels and a title
ax.set_xlabel("Date")
ax.set_ylabel("Proportion of Negative Sentiment")
ax.grid(True, axis="y", alpha=0.3)
# Set y-axis to start at 0
ax.set_ylim(bottom=0)

# Make x-axis labels vertical
plt.xticks(rotation="vertical")

plt.show()


# In[ ]:


# In[61]:


# Define a dictionary that maps topics to colors
# Remove the name of the index

color_dict = {
    "Cost": "pink",
    "Noise": "grey",
    "Cold Weather": "gold",
    "Boilers and Stoves": "plum",
    "All Topics": "Black",
}  # Replace with your actual topics and colors

# Create a list of colors for the topics in selected_topics
colors = [color_dict[topic] for topic in selected_topics.index]
# selected_topics.T.plot(kind="line", figsize=(14, 8), color=NESTA_COLOURS)
transpose_selected_topics = selected_topics.T
transpose_selected_topics.index = transpose_selected_topics.index.droplevel(0)
transpose_selected_topics.plot(kind="line", figsize=(14, 8), color=colors)
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylabel("Sentiment Score")
plt.xlabel("Year")
plt.title("Sentiment over time per topic of conversation about heat pumps")
plt.grid(True, axis="y", alpha=0.3)


# In[122]:


sentences_data["month"] = pd.to_datetime(sentences_data["date"]).dt.to_period("M")
sentiment_over_time_grants = sentences_data[["sentences", "id", "month", "year"]]
sentiment_over_time_grants = sentiment_over_time_grants.merge(
    sentiment_data, left_on="sentences", right_on="text"
)

# print(sentiment_over_time_grants.head())
sentiment_over_time_grants["sentiment_number"] = sentiment_over_time_grants[
    "sentiment"
].map({"negative": -1, "neutral": 0, "positive": 1})


aggregated_docs_info_grants = docs_info.copy()
for category, subcategories in categories_dict.items():
    if category in list_of_dicts_with_subsubcategories:
        for subcategory, topics in subcategories.items():
            for topic, value in topics.items():
                aggregated_docs_info_grants.loc[
                    aggregated_docs_info_grants["Name"] == value, "Name"
                ] = subcategory
    else:
        for topic, value in subcategories.items():
            aggregated_docs_info_grants.loc[
                aggregated_docs_info_grants["Name"] == value, "Name"
            ] = category


sentiment_over_time_grants = sentiment_over_time_grants.merge(
    aggregated_docs_info_grants[["Document", "Topic", "Name"]],
    left_on="sentences",
    right_on="Document",
)

# print(sentiment_over_time_grants.head())
sentiment_over_time_grants = (
    sentiment_over_time_grants.groupby(["Name", "year"])["sentiment"]
    .value_counts()
    .unstack()
)

# Calculate the total sentiments
sentiment_over_time_grants["total"] = (
    sentiment_over_time_grants["positive"]
    + sentiment_over_time_grants["neutral"]
    + sentiment_over_time_grants["negative"]
)

# Calculate the proportion of negative sentiments
sentiment_over_time_grants["proportion_negative"] = (
    sentiment_over_time_grants["negative"] / sentiment_over_time_grants["total"]
)
# sentiment_over_time_grants
topics_to_select = [
    "Microgeneration Certification Scheme",
    "EPC Ratings",
    "BUS Grants",
    "Domestic RHI",
]

selected_topics = sentiment_over_time_grants.loc[topics_to_select]

import matplotlib.pyplot as plt
import seaborn as sns

# Create a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=selected_topics,
    x="year",
    y="proportion_negative",
    hue="Name",
    palette=NESTA_COLOURS,
)

# Set the title and labels
# plt.title('Proportion of Negative Sentiments Over Time')
plt.xlabel("Year")
plt.ylabel("Proportion of Negative Sentiment")

# Show the plot
plt.show()

# Select only 'Name', 'year', and 'proportion_negative'
# sentiment_over_time_grants = sentiment_over_time_grants[['Name', 'year', 'proportion_negative']]
"""
topics_to_select = ["Microgeneration Certification Scheme", "EPC Ratings", "BUS Grants", "Domestic RHI"]
selected_topics = sentiment_over_time_grants.loc[topics_to_select]
#selected_topics
 # Replace with your actual topics and colors

# Create a list of colors for the topics in selected_topics
#colors = [color_dict[topic] for topic in selected_topics.index]
#selected_topics.T.plot(kind="line", figsize=(14, 8), color=NESTA_COLOURS)
transpose_selected_topics = selected_topics.T
#transpose_selected_topics.index = transpose_selected_topics.index.droplevel(0)
transpose_selected_topics.plot(kind="line", figsize=(14, 8), color=NESTA_COLOURS)
plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.ylabel("Sentiment Score")
plt.xlabel("Year")
plt.title("Sentiment over time per topic of conversation about heat pumps")
plt.grid(True, axis = 'y', alpha = 0.3)
"""


# In[118]:


sentiment_over_time_grants


# In[ ]:
