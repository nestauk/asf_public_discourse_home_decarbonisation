# %% [markdown]
# ## overview of analysis
#
#
#

# %% [markdown]
# ## package imports

# %%
import matplotlib.pyplot as plt
import pandas as pd
from asf_public_discourse_home_decarbonisation.utils.plotting_utils import (
    set_plotting_styles,
    NESTA_COLOURS,
)

set_plotting_styles()

from asf_public_discourse_home_decarbonisation.utils.general_utils import (
    flatten_mapping,
    flatten_mapping_child_key,
    map_values,
)
from asf_public_discourse_home_decarbonisation.getters.topic_analysis_getters import (
    get_docs_info,
    get_topics_info,
    get_sentence_data,
)
from asf_public_discourse_home_decarbonisation.getters.sentiment_getters import (
    get_sentence_sentiment,
)

# %% [markdown]
# ## analysis specific settings

# %%
# data import settings
source = "mse"  # either "mse" or "buildhub"
filter_by = "heat pump"
analysis_start_date = "2016-01-01"
analysis_end_date = "2024-05-22"


# %%
# analysis settings
first_complete_month = "2016-01"
last_complete_month = "2024-04"

# topic growth analysis
# first period: may 2019 to april 2020
growth_months_period_start = [
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
# second period: may 2023 to april 2024
growth_months_period_end = [
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
sentiment_data = get_sentence_sentiment(
    source, filter_by, analysis_start_date, analysis_end_date
)
sentences_data = get_sentence_data(
    source, filter_by, analysis_start_date, analysis_end_date
)
doc_info = get_docs_info(source, filter_by, analysis_start_date, analysis_end_date)
topics_info = get_topics_info(source, filter_by, analysis_start_date, analysis_end_date)

# %%
# we know a portion of the documents tagged as outliers actually mention heat pumps
len(
    doc_info[
        (doc_info["Topic"] == -1)
        & (doc_info["Document"].str.contains("heat pump", case=False))
    ]
) / len(doc_info) * 100

# %%
# proportion of documents not tagged as outliers that mention heat pumps
len(
    doc_info[
        (doc_info["Topic"] != -1)
        & (doc_info["Document"].str.contains("heat pump", case=False))
    ]
) / len(doc_info) * 100

# %%


# %% [markdown]
# ## Renaming and grouping topics

# %%
if source == "buildhub":
    renaming_and_grouping_topics = {
        "Solar panels and solar PV": "0_pv_solar_panels_excess",
        "Heat pump types": {
            # Heat pump types
            "Air source heat pumps": "4_source_air_pump_heat",
            "Heat pumps and people": "35_pumps_heat_pump_people",
            "Ground source heat pumps and boreholes": "36_ground_borehole_source_boreholes",
            "A2A, A2W and aircon units": "46_air_aircon_a2w_units",
            "Monoblock vs split heat pumps": "79_split_monoblock_monobloc_splits",
            # Specific brands
            "Mitsubishi Ecodan power": "86_mitsubishi_ecodan_tech_112kw",
            "Ecodan": "146_ecodan_ecodans_r32_know",
            "Samsung Gen6": "149_samsung_gen6_gen_ehs",
        },
        "Boilers and other heating systems": {
            "Boilers, gas and oil": "5_boiler_boilers_gas_oil",
            "Gas and oil": "23_gas_oil_mains_fuel",
            "Gas and oil boilers": "45_boiler_gas_oil_source",
            "Combi boilers": "130_combi_combis_gas_hot",
            "LPG, oil and mains gas": "118_lpg_gas_mains_oil",
            "Stoves and wood burners": "40_stove_wood_burner_log",
            "Central heating systems": "34_heating_systems_central_design",
        },
        "Hydrogen": "102_hydrogen_green_natural_gas",
        "Noise": "2_noise_sound_noisy_quiet",
        "Hot water heating": {
            "Domestic hot water heating": "3_hot_domestic_water_heating",
            "Domestic hot water": "53_hot_domestic_water_source",
            # Immersion heater for hot water heating
            "Willis immersion heater": "140_willis_heater_heaters_immersion",
            "Immersion": "136_immersion_immersions_3kw_solic",
            "Immersion heater": "91_immersion_heater_switch_iboost",
            # Unvented cylinders
            "Unvented cylinders": "110_cylinder_reheat_unvented_cylinders",
            "Unvented cylinder Telford": "27_cylinder_unvented_cylinders_telford",
            "Domestic hot water unvented cylinder": "65_cylinder_domestic_hot_unvented",
            # Thermal stores
            "Thermal stores:": "128_store_thermal_stores_dump",
        },
        "Underfloor heating and radiators": {
            "Undefloor heating": "8_floor_heating_underfloor_floors",
            "Underfloor heating and heat pumps": "13_floor_heating_pump_source",
            "Radiators 1": "19_rads_rad_upstairs_floor",
            "Radiators 2": "24_radiators_radiator_output_assisted",
            "Radiators 3": "106_radiators_radiator_pump_source",
            "Underfloor heating and radiators": "90_radiators_floor_underfloor_heating",
        },
        "Installations and installers": {
            "Installations and installers": "9_installer_installers_install_installation",
            "Engineers and training": "92_engineer_engineering_training_engineers",
            "Plumbers and electricians": "26_plumber_plumbers_plumbing_electrician",
            "Self-building": "69_build_self_architect_builders",
            "Replacement and repairs": "143_replace_replacement_repairable_repair",
        },
        "Insulation": {
            "Cavity wall insulation": "6_insulation_insulated_cavity_walls",
            "Insulation": "124_upstairs_downstairs_heating_insulated",
            "Glazed windows": "94_windows_glazed_window_glazing",
            "Windows and blinds": "132_blinds_external_velux_windows",
        },
        "Cold weather": {
            "Cold weather": "11_winter_degrees_cold_weather",
            "Cold and warm house": "121_house_warm_cold_cool",
        },
        "Flow temperature": "14_flow_temp_temperature_temps",
        "Temperature controls": {
            "Weather compensation curve": "48_compensation_weather_curve_comp",
            "Thermostast": "49_thermostat_thermostats_room_control",
            "Heat pump temperature": "80_hp_temp_hps_temps",
            "Heat pump cycling": "20_cycling_short_minutes_hours",
        },
        "EPCs and efficiency": {
            "SAP score": "76_sap_report_rating_score",
            "EPC rating and RHI": "100_epc_rating_incentive_renewable",
            "Efficiency": "120_efficiency_efficient_inefficient_inefficiency",
        },
        "Heat pump efficiency": "10_cop_scop_35_better",
        "Heat loss calculations": {
            "Heat loss calculations spreadsheet": "52_loss_calculations_calcs_spreadsheet",
            "Building heat loss calculation": "150_loss_calculation_losses_building",
        },
        "Planning permissions": "75_planning_lpa_pd_permission",
        "MCS": {
            "MCS scheme 1": "25_certification_microgeneration_scheme_certificate",
            "MCS scheme 2": "142_certification_microgeneration_scheme_certified",
        },
        "Grants and incentives": {
            "RHI scheme": "31_incentive_renewable_payments_scheme",
            "Cost incentives": "47_cost_source_pump_incentive",
            "BUS grant": "61_grant_bus_grants_5k",
        },
        "Money and costs": {
            "Prices": "30_price_money_prices_pay",
            "Costs": "41_10k_pay_cost_price",
            "Air source heat pump costs": "67_source_air_cost_costs",
            "Electricity prices": "88_electricity_energy_prices_grid",
            "Running costs": "151_costs_running_run_cost",
            "Heat pump VAT reclaim": "101_vat_reclaim_zero_ex",
            "Heat pump cost - credit card": "57_hp_hps_card_rules",
            "Quotes": "138_quote_quotes_quoting_text",
        },
        "Energy consumption and power": {
            # Readings and consumption
            "Smart meter readings": "44_meter_smart_meters_readings",
            "Average daily consumption": "37_kw_kwh_day_average",
            "Daily consumption": "74_kwh_kw_kwhday_usage",
            "Ecodan standby consumption": "133_standby_consumption_power_ecodan",
            "Domestic hot water consumption": "83_hot_domestic_water_kwh",
            # Power
            "Heat pump power": "21_kw_pump_kwh_source",
            "Power": "42_10kw_7kw_12kw_5kw",
        },
        "Tariffs": {
            "Octupus agile and cosy": "64_octopus_tariff_agile_cosy",
            "Tariffs - export and economy": "98_tariff_tariffs_export_economy",
            "E7 and E10 tariffs": "129_e7_e10_rate_cheap",
            "E7 and E10 tariffs - overnight boost": "126_e7_e10_boost_overnight",
        },
        "Heat pump sizing": {
            "Sizing and overfitting": "66_size_sizing_fit_oversizing",
            "Heat pump sizing": "107_sizing_size_source_pump",
        },
        "Batteries and EV charging": "104_batteries_battery_ev_charging",
        "Tanks and storage heaters": {
            "Sunamp": "32_sunamp_sunamps_charge_amp",
            "Tank": "59_tank_tanks_loft_litre",
            "Tank temperature": "119_tank_temperature_temp_degrees",
            "Heat buffer tank": "28_buffer_tank_pump_heat",
            "Buffer tank litres": "71_buffer_tank_tanks_litres",
            "Buffer tank heating": "96_buffer_floor_heating_tank",
        },
        "Suppliers and companies": "62_supplier_brands_company_suppliers",
        "Property": {
            "Renovation": "12_house_property_houses_renovation",
            "Showers and baths": "16_shower_showers_bath_baths",
            "Tiles and floors": "70_tiles_floor_flooring_floors",
            "Rooms and bedrooms": "97_rooms_room_bedrooms_cooler",
            "Roof": "109_roof_flat_facing_pitched",
            "Kitchen and rooms": "113_kitchen_bedroom_room_rooms",
            "Towel rails": "115_towel_rails_rail_towels",
        },
        "Technical": {
            # pipework
            "Pipes and pipework": "7_pipe_pipes_pipework_copper",
            "Pipes and heating": "148_pipes_floor_pipe_heating",
            # pump flow rate
            "pump flow rate": "15_pump_flow_source_heat",
            "flow rates": "54_flow_rate_rates_meter",
            # valves
            "valves": "29_valve_valves_bypass_pressure",
            # manifold systems
            "manifolds": "39_manifold_manifolds_blending_mixer",
            # circulation pumps
            "speed circulation": "17_pump_pumps_speed_circulation",
            # concrete slab and cooling
            "concrete slab": "18_screed_slab_concrete_100mm",
            "concrete lab, floor, cooling": "22_slab_concrete_floor_cooling",
            "cooling": "112_cooling_active_mode_newtons",
            # heat pump defrost cycle
            "defrost cycle": "43_defrost_defrosting_defrosts_cycle",
            # heat pump controllers
            "controllers": "50_controller_controls_control_controllers",
            "ecodan controllers": "135_ecodan_controller_85kw_mitsubishi",
            "ftc controllers": "147_ftc6_ftc_ftc5_controller",
            # compressors
            "compressors": "51_compressor_compressors_scroll_speed",
            # wiring, ducting
            "wiring": "56_wiring_cable_wires_wire",
            "ducting": "60_duct_ducts_ducting_ducted",
            # warranty
            "smasung gen warranty": "141_samsung_gen_samsungs_warranty",
            "manufacturer warranty": "134_warranty_year_manufacturer_item",
            # inspection
            "annual inspection": "144_g3_inspection_unvented_annual",
            # sensors
            "sensors": "145_sensor_sensors_remote_cable",
            # zoning system
            "zones": "63_zones_zone_zoning_single",
            # fans and coils
            "coils": "68_coil_coils_cylinder_3m2",
            "fans": "81_fan_coil_coils_fans",
            # ventilation
            "ventillation": "72_mvhr_ventilation_fresh_rates",
            # anti freeze soluations and refrigerants
            "glycol": "73_glycol_antifreeze_inhibitor_freeze",
            "propane refrigerant": "77_r290_r32_propane_refrigerant",
            "refrigerant": "85_refrigerant_fridge_refrigeration_refrigerants",
            "frost protect": "117_frost_frosting_protection_protect",
            # buffer tank volumiser
            "buffer volumiser and cycling": "82_buffer_buffers_volumiser_cycling",
            # settings
            "settings": "87_settings_parameters_setting_change",
            # inverter
            "iverter": "89_inverter_driven_inverters_drive",
            # condensation
            "condensation": "99_condensate_condensation_drain_condensing",
            # schematics and mannuals
            "schematics": "108_vaillant_installer_schematics_arotherm",
            "manuals": "125_manual_manuals_instructions_download",
            # Other
            "loops": "111_loops_loop_100m_kitchen",
            "phase": "127_phase_single_3phase_transformer",
        },
        "Other": {  # Other topics
            "Approaches, options etc": "33_approach_options_idea_solution",
            "Numbers and calculations": "55_numbers_figures_calculations_maths",
            "Spreadhsheet data": "93_spreadsheet_data_excel_chart",
            "Units": "123_units_unit_unitower_si",
            "Legionella in domestic hot water systems": "78_legionella_cycle_legionnaires_bacteria",
            "Location/areas": "84_scotland_cornwall_england_wales",
            "Dates by": "103_2023by_2022by_editeddecember_2021by",
            "Carbon emissions": "58_carbon_co2_emissions_fossil",
            "Climate change and scientists": "105_climate_change_global_scientists",
            "Energy general": "131_energy_cool_protons_saving",
            "Trees, wood and timber": "122_trees_wood_tree_timber",
            "Systems design": "38_systems_setup_designed_design",
            "Design": "114_design_designers_designer_designing",
            "Wall etc": "116_wall_plant_source_location",
            "relay": "137_relay_relays_contacts_ssr",
        },
        "Unrelated to HPs": {
            "Outliers cluster": "-1_heat_heating_pump_water",
            "General sentences": "1_thread_question_forum_post",
            "Photos": "139_photos_post_photo_picture",
            "Science and physics": "95_science_physics_scientific_scientist",
        },
    }
elif source == "mse":
    renaming_and_grouping_topics = {
        "Solar panels and solar PV": "0_solar_panels_battery_pv",
        "Boilers and other heating systems": {
            "Old gas boilers": "3_boiler_boilers_gas_old",
            "Gas and fossil fuels": "16_gas_mains_fossil_fuels",
            "Heat Pumps vs. boilers": "21_boiler_pump_oil_gas",
            "LPG and oil": "49_lpg_oil_bulk_tank",
            "Oil": "51_oil_oils_club_christmas",
            "Wood burner & stove": "30_wood_burner_stove_log",
        },
        "Heat pump types": {
            "Air source heat pumps": "14_air_source_pump_heat",
            "General heat pumps": "17_pumps_heat_pump_cantor",
            "Ground source heat pumps": "36_ground_source_pump_heat",
            "Split heat pumps and aircon units": "60_air_aircon_units_split",
        },
        "Hydrogen": "56_hydrogen_h2_hvo_gas",
        "Insulation": "4_insulation_loft_wall_insulated",
        "Underfloor heating and radiators": {
            "Radiators": "5_radiators_radiator_rads_flow",
            "Underfloor heating and radiators": "15_underfloor_floor_heating_radiators",
        },
        "Property": "7_house_property_bed_bungalow",
        "Money and costs": {
            "Money savings & suppliers": "12_money_savings_price_suppliers",
            "Credit cost": "22_month_year_credit_cost",
            "Heating systems and costs": "28_heating_systems_costs_electric",
            "Heat pump cost": "35_pump_cost_source_air",
            "Energy prices and electricity cost": "61_electricity_prices_energy_costs",
            "Quotes": "58_quote_quotes_ive_quoting",
        },
        "Noise": "11_noise_noisy_quiet_microgeneration",
        "Tanks and storage heaters": {
            "Storage heaters": "26_storage_heaters_heater_night",
            "Tanks": "41_tank_tanks_bunded_500l",
            "Sunamp": "53_sunamp_phase_sunamps_change",
        },
        "Domestic hot water": "9_water_hot_cylinder_domestic",
        "Showers and baths": "25_shower_showers_bath_electric",
        "Smart meters and readings": "10_meter_smart_meters_readings",
        "Electricity and gas consumption": {
            "Average daily consumption": "13_kwh_day_average_kw",
            "Heating hot water consumption": "31_kwh_hot_heating_water",
            "Heat pump consumption": "23_pump_heat_kw_kwh",
            "Energy usage consumption": "57_energy_use_consumed_usage",
            "Electricity consumption": "37_year_electricity_kwh_month",
        },
        "Installations and installers": "18_installer_installers_installation_install",
        "Pipework and plumbing": "27_pipes_pipe_pipework_plumbing",
        "Heat pump performance": "19_cop_scop_flow_cops",
        "Tariffs": {
            "Octopus agile": "8_octopus_agile_tariff_flux",
            "Economy7 tariff": "24_e7_economy_tariff_rate",
            "Tariffs and rates": "44_tariff_tariffs_rate_offer",
        },
        "Flow temperature": "39_flow_temp_temperature_degrees",
        "Temperature controls": {
            "Setting thermostast temperature": "2_thermostat_degrees_cold_set",
            "Weather compensation": "32_compensation_weather_curve_temperature",
            "Heat pump temperature": "33_pump_heat_temperature_source",
            "Defrost": "55_defrost_defrosting_freezing_ice",
        },
        "Settings and controls": "20_settings_controls_control_tweaking",
        "Grants": {
            "BUS and government grants": "45_grant_grants_government_bus",
            "Renewable heat incentive": "29_renewable_incentive_payments_heat",
        },
        "Planning permissions": {
            "Planning permissions and councils": "38_planning_permission_council_ipswich",
            "Planning/development permissions": "52_planning_permission_development_permitted",
        },
        "EPCs and energy performance": {
            "EPCs and properties": "42_epc_epcs_rating_property",
            "EPCs and heating": "50_epc_rating_heat_epcs",
        },
        "MCS": "54_certification_microgeneration_scheme_certificate",
        "Technical": {
            "Fuse phase": "48_fuse_phase_fuses_100a",
            "Valves and pressure": "59_valve_valves_pressure_bypass",
            "Phase change materials": "40_phase_change_pcm_liquid",
        },
        "Other": {
            "Green planet and environment": "47_green_planet_greener_environmental",
            "Numbers and calculations": "34_numbers_figures_calculations_sums",
            "Time of use": "43_hours_247_hour_minutes",
            "Legionella in domestic hot water systems": "46_legionella_cycle_immersion_bacteria",
        },
        "Unrelated to HPs": {
            "Outliers cluster": "-1_heat_pump_water_air",
            "General sentences": "1_thread_im_post_think",
            "Tumble dryers": "6_dryer_tumble_dry_washing",
        },
    }
else:
    raise ValueError("source must be 'buildhub' or 'mse'")

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
aggregated_topics_to_remove = ["Unrelated to HPs", "Other"]

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
aggregated_topics

# %% [markdown]
# ### 2. Sentiment breakdown (neg, pos, neu) of aggregated topics

# %%
agg_topic_sentiment = doc_info.merge(
    sentiment_data, how="left", left_on="Document", right_on="text"
)
agg_topic_sentiment = (
    agg_topic_sentiment.groupby(["aggregated_topic_names", "sentiment"])
    .nunique()["Document"]
    .unstack()
    .fillna(0)
)
agg_topic_sentiment = (
    agg_topic_sentiment.div(agg_topic_sentiment.sum(axis=1), axis=0) * 100
)

agg_topic_sentiment = agg_topic_sentiment[
    ~agg_topic_sentiment.index.isin(aggregated_topics_to_remove)
]

agg_topic_sentiment.sort_values("negative", ascending=True, inplace=True)

# %%
agg_topic_sentiment

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
# ### 3. Aggregated topics: total count vs sentiment
#

# %%
agg_topic_sentiment.reset_index(inplace=True)
sentiment_vs_size = agg_topic_sentiment[["aggregated_topic_names", "negative"]].merge(
    aggregated_topics, on="aggregated_topic_names"
)

# %%
agg_topic_sentiment

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
# ### 4. Growth vs. size - comparing time periods
#
# May 2019 to April 2020 and 2) May 2023 to April 2024

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
        sentences_year_month_info["year_month"].isin(growth_months_period_end)
    ]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)
n_2020 = (
    sentences_year_month_info[
        sentences_year_month_info["year_month"].isin(growth_months_period_start)
    ]
    .groupby("aggregated_topic_names", as_index=False)
    .count()[["aggregated_topic_names", "sentences"]]
)

# %%
sentiment_vs_size = agg_topic_sentiment[["aggregated_topic_names", "negative"]].merge(
    aggregated_topics, on="aggregated_topic_names"
)
sentiment_vs_size = (
    sentiment_vs_size.merge(n_2024)
    .rename(columns={"sentences": "n_2024"})
    .merge(n_2020)
    .rename(columns={"sentences": "n_2020"})
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

# %% [markdown]
# ### 5. Topic changes over time

# %%
topics_date = sentences_data.merge(
    doc_info[["Document", "aggregated_topic_names", "topic_names"]],
    left_on="sentences",
    right_on="Document",
)[["sentences", "datetime", "aggregated_topic_names", "topic_names"]]
topics_date["year_month"] = pd.to_datetime(topics_date["datetime"]).dt.to_period("M")
topics_date["year_month"] = topics_date["year_month"].astype(str)
topics_date = topics_date.groupby(
    ["aggregated_topic_names", "topic_names", "year_month"]
).count()[["sentences"]]
topics_date.reset_index(inplace=True)
topics_date = topics_date[topics_date["aggregated_topic_names"] != "Unrelated to HPs"]


# %%


# %%
year_month_list = pd.period_range(
    start=first_complete_month, end=last_complete_month, freq="M"
)
year_month_list = year_month_list.strftime("%Y-%m")


# %%


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
# ### 6. Example quotes with neg/pos sentiment for each topic

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

    print("***Topic: ", t)
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


# %%
