# Dictionary for mapping heating technology terms to specific categories.
# This ruleset is used to categorise and tag text data based on the occurrence of certain keywords.
# Each entry in the list is a dictionary with two keys:
# - 'value': a regular expression pattern that matches specific terms or phrases related to heating technologies.
# - 'tag': a string representing the category or tag to be associated with the matched terms.
# This ruleset is designed to be used in text processing functions to classify and analyse forum posts
# or other text data based on the mention of different heating technologies.
# Example usage includes keyword frequency analysis, categorisation of discussion topics, etc.

heating_technologies_ruleset_twitter = [
    {
        "value": "heat pump|heat pumps",
        "tag": "general_heat_pumps",
    },
    {
        "value": "air source hp|air source hps|air-source hp|air-source hps",
        "tag": "ashp",
    },
    {
        "value": "ground source hp|ground source hps|ground-source hp|ground-source hps",
        "tag": "gshp",
    },
    {
        "value": "water source hp| water source hps|water-source hp|water-source hps",
        "tag": "wshp",
    },
    {
        "value": "air to air hp|air to air hps|air-to-air hp|air-to-air hps|air2air hp|air2air hps|a2a hp|a2a hps",
        "tag": "atahp",
    },
    {
        "value": "air to water hp|air to water hps|air-to-water hp|air-to-water hps| air2water hps|air2water hp|a2w hp|a2w hps",
        "tag": "atwhp",
    },
    {
        "value": "hybrid hp|hybrid hps|bivalent hp|bivalent hps|warm air hp|warm air hps",
        "tag": "heat_pump_others",
    },
    {
        "value": "boiler|boilers|#boiler|#boilers",
        "tag": "general_boilers",
    },
    {
        "value": "combi boiler | combi boilers | combi-boiler | combi-boilers ",
        "tag": "combi_boilers",
    },
    {
        "value": " gas boiler | gas boilers ",
        "tag": "gas_boilers",
    },
    {
        "value": " oil boiler | oil boilers ",
        "tag": "oil_boilers",
    },
    {
        "value": "hydrogen boiler | hydrogen boilers ",
        "tag": "hydrogen_boilers",
    },
    {
        "value": "hydrogen ready boiler|hydrogen ready boilers|hydrogen-ready boiler|hydrogen-ready boilers",
        "tag": "hydrogen_ready_boilers",
    },
    {
        "value": "electric boiler|electric boilers",
        "tag": "electric_boilers",
    },
    {
        "value": "biomass boiler|biomass boilers",
        "tag": "biomass_boilers",
    },
    {
        "value": "electric heating|electric radiator|electric radiators|central heating",
        "tag": "heating_others",
    },
    {"value": "solar thermal|solar water heating", "tag": "solar_thermal"},
    {"value": "heating system|heating systems", "tag": "heating_system"},
    {
        "value": "district heating|heat network|heat networks",
        "tag": "district_heating",
    },
    {"value": "hybrid heating", "tag": "hybrid_heating"},
    {
        "value": "hp installation|hp installations|hybrid installation|hybrid installations|hybrid heating installation|hybrid heating installations",
        "tag": "installations",
    },
    {
        "value": "hp installer|hp installers|hp engineer|hp engineers|heating engineer|heating engineers|boiler engineer|boiler engineers",
        "tag": "installers_and_engineers",
    },
    {
        "value": "retrofit installer|retrofit installers|renewables installer|renewables installers",
        "tag": "retrofit_or_renewable_installers",
    },
    {
        "value": "low carbon heating|low-carbon heating|#lowcarbonheating|home decarbonisation",
        "tag": "low_carbon_heating_and_home_decarbonisation",
    },
    {
        "value": "bus scheme|boiler upgrade scheme|renewable heat incentive|domestic rhi|clean heat grant|home energy scotland grant|home energy scotland loan|home energy scotland scheme",
        "tag": "government_grants",
    },
    {
        "value": "microgeneration certification scheme|mcs certified|mcs certification|mcs installation|mcs installations",
        "tag": "mcs",
    },
    {
        "value": "hp cost estimator|hp cost calculator",
        "tag": "nesta_cost_estimator_tool",
    },
    {
        "value": "MSBC|Money Saving Boiler Challenge",
        "tag": "MSBC",
    },
]
