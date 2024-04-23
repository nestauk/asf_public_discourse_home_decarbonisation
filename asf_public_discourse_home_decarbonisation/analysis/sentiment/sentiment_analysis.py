# Package imports
import argparse
import nltk

nltk.download("punkt")  # Download the Punkt tokenizer models if not already downloaded
from nltk.tokenize import sent_tokenize
import pandas as pd

# Local imports
from asf_public_discourse_home_decarbonisation.utils.sentiment_analysis_utils import (
    compute_sentiment_with_flair,
)
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    read_public_discourse_data,
)
from asf_public_discourse_home_decarbonisation.utils.text_cleaning_utils import (
    process_abbreviations,
    remove_urls,
    remove_username_pattern,
    remove_introduction_patterns,
)

# Setup logging
import logging

logger = logging.getLogger(__name__)


def argparser() -> argparse.Namespace:
    """
    Argparser function to parse arguments from the command line: n_runs, path_to_config_file, path_to_data
    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source",
        type=str,
        help='"mse", "buildhub" or path to data source (local or S3)',
    )
    args = parser.parse_args()
    return args


def prep_data_for_sentiment_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data for sentiment analysis.
    Args:
        data (pd.DataFrame): the data to preprocess
    Returns:
        pd.DataFrame: the preprocessed data
    """
    # Remove URLs
    data["text"] = data["text"].apply(remove_urls)

    # Process abbreviations
    data = process_abbreviations(data)

    # Convert text to lowercase
    data["text"] = data["text"].str.lower()

    # Remove username patterns
    data["text"] = data["text"].apply(remove_username_pattern)

    # Remove introduction patterns
    data["text"] = data["text"].apply(remove_introduction_patterns)

    return data


def sentence_sentiment(data):
    data["sentences"] = data["text"].apply(sent_tokenize)

    data = (
        data[["id", "datetime", "date", "sentences", "is_original_post"]]
        .explode("sentences")
        .reset_index(drop=True)
    )

    data = data[data["sentences"].str.contains("heat pump")]

    data["sentiment_flair"] = data["sentences"].apply(compute_sentiment_with_flair)

    return data


if __name__ == "__main__":
    # Get the data
    data = read_public_discourse_data(source="data_source")

    data = prep_data_for_sentiment_analysis(data)

    # Filter data for posts containing "heat pump" & compute sentence sentiment
    hp_data = data[data["text"].str.contains("heat pump")]
    hp_data = sentence_sentiment(hp_data)

    hp_sentiment_per_year = (
        hp_data.groupby(["year", "sentiment_flair"])["id"].count().unstack()
    )

    # Filter data for posts containing "boiler" & compute sentence sentiment
    boiler_data = data[data["text"].str.contains("boiler")]
    boiler_data = sentence_sentiment(boiler_data)

    boiler_sentiment_per_year = (
        boiler_data.groupby(["year", "sentiment_flair"])["id"].count().unstack()
    )
