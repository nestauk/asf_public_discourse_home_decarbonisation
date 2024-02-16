data_source_parms = [
    {
        "data_source": "mse",
        "part": [
            {
                "category": "all",
                "keywords": "heat_pump_keywords",
            },
            {
                "category": "all",
                "keywords": "boiler_keywords",
            },
            {
                "category": "green-ethical-moneysaving",
            },
            {
                "category": "lpg-heating-oil-solid-other-fuels",
            },
            {
                "category": "is-this-quote-fair",
            },
            {
                "category": "energy",
            },
            {
                "category": "all",
            },
        ],
    },
    {
        "data_source": "bh",
        "part": [
            {
                "category": "all",
                "keywords": "heat_pump_keywords",
            },
            {
                "category": "all",
                "keywords": "boiler_keywords",
            },
            {
                "category": "all",
            },
            {
                "category": "119_air_source_heat_pumps_ashp",
            },
            {
                "category": "120_ground_source_heat_pumps_gshp",  # too small to generate topics
            },
            {
                "category": "125_general_alternative_energy_issues",  # too small to generate topics
            },
            {
                "category": "136_underfloor_heating",
            },
            {
                "category": "137_central_heating_radiators",
            },
            {
                "category": "139_boilers_hot_water_tanks",
            },
            {"category": "140_other_heating_systems"},
        ],
    },
]

from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

model_and_additional_params = [
    {"model_name": "Basic Model", "text_column": "title", "filter": "posts"},
    {
        "model_name": "`Auto` number of topics",
        "nr_topics": "auto",
        "text_column": "title",
        "filter": "posts",
    },
    {
        "model_name": "KeyBERTInspired",
        "representation_model": KeyBERTInspired(),
        "text_column": "title",
        "filter": "posts",
    },
    {
        "model_name": "Cluster size>15",
        "min_topic_size": 15,
        "text_column": "title",
        "filter": "posts",
    },
    {
        "model_name": "Cluster size>30",
        "min_topic_size": 30,
        "text_column": "title",
        "filter": "posts",
    },
    {
        "model_name": "CountVectorizer",
        "vectorizer_model": CountVectorizer(stop_words="english"),
        "text_column": "title",
        "filter": "posts",
    },
]