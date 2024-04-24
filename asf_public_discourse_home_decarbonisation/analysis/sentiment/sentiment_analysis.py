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
from asf_public_discourse_home_decarbonisation import S3_BUCKET
import logging


# Setup logging
logger = logging.getLogger(__name__)


def argparser() -> argparse.Namespace:
    """
    Argparser function to parse arguments from the command line: data_source

    data_source takes either `mse` for Money Saving Expert data, `buildhub` for BuildHub data or a path to a local or S3 data source.
    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_source",
        type=str,
        help='"mse", "buildhub" or name of a different source',
    )
    parser.add_argument(
        "--source_path",
        type=str,
        help="path to data source (local or S3) if different from MSE and Buildhub",
    )
    args = parser.parse_args()
    return args


def prep_data_for_sentiment_analysis(
    data: pd.DataFrame, data_source: str
) -> pd.DataFrame:
    """
    Preprocess the data for sentiment analysis.
    Args:
        data (pd.DataFrame): the data to preprocess
        data_source (str): data source name
    Returns:
        pd.DataFrame: the preprocessed data
    """
    # Remove URLs
    data["text"] = data["text"].apply(remove_urls)

    # Convert text to lowercase
    data["text"] = data["text"].str.lower()

    # Process abbreviations
    data = process_abbreviations(data)

    # Remove username patterns
    data["text"] = data["text"].apply(remove_username_pattern)

    # Remove introduction patterns
    data["text"] = data["text"].apply(remove_introduction_patterns)

    if data_source == "buildhub":
        data.rename(columns={"url": "id"}, inplace=True)
        data.rename(columns={"date": "datetime"}, inplace=True)
        data["datetime"] = pd.to_datetime(data["datetime"])

    data["year"] = data["datetime"].dt.year

    return data


def compute_sentence_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sentiment for each sentence in the data.

    Args:
        data (pd.DataFrame): dataframe with columns "id", "year", "text", "is_original_post"

    Returns:
        pd.DataFrame: dataframe with one additional column, the sentiment column
    """
    data["sentences"] = data["text"].apply(sent_tokenize)

    data = (
        data[["id", "year", "sentences", "is_original_post"]]
        .explode("sentences")
        .reset_index(drop=True)
    )

    data = data[data["sentences"].str.contains("heat pump")]

    data["sentiment"] = data["sentences"].apply(compute_sentiment_with_flair)

    return data


def compute_average_and_yearly_stats(data: pd.DataFrame, filter_term: str):
    """
    Computes average sentiment and yearly sentiment breakdown
    (i.e. number of sentences with POS/NEG sentiment per year).

    Args:
        data (pd.DataFrame): Data with id and sentiment columns

    Returns:
        pd.DataFrame: _description_
    """
    avg_sentiment = data["sentiment"].mean()
    logger.info(f"Average sentiment for `{filter_term}`: {avg_sentiment}")
    breakdown_per_year = data.groupby(["year", "sentiment"])["id"].count().unstack()

    logger.info(
        f"Yearly sentiment breakdown for `{filter_term}`:\n{breakdown_per_year}"
    )

    return breakdown_per_year


if __name__ == "__main__":
    args = argparser()
    data_source = args.data_source
    source_path = args.source_path

    # filter terms
    filter_terms = ["heat pump", "boiler"]

    # Get the data
    if source_path is None:
        data = read_public_discourse_data(source=data_source)
    else:
        data = read_public_discourse_data(source=source_path)

    # preparing data for sentiment analysis
    data = prep_data_for_sentiment_analysis(data)

    # Compute sentiment for sentences containing specific terms
    for term in filter_terms:
        filtered_data = data[data["text"].str.contains(term)]
        filtered_data = compute_sentence_sentiment(filtered_data)
        yearly_stats = compute_average_and_yearly_stats(filtered_data, term)

        yearly_stats.to_csv(
            f"s3://{S3_BUCKET}/data/{data_source}/outputs/sentiment/yearly_sentiment_stats_source_{data_source}_term_{term}.csv",
            index=False,
        )
