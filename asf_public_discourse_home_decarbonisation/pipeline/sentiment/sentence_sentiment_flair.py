"""
Script to compute sentence level sentiment analysis in posts mentioning certain terms (such as "heat pump" or "boiler") using [Flair](https://flairnlp.github.io/docs/tutorial-basics/tagging-sentiment).

The script:
- It starts by reading in data from the specified source (MSE, BuildHub or another) and preprocesses it for sentiment analysis;
- For each of the terms ("heat pump" and "boiler"), it then computes the sentiment of each sentence mentioning the term (positive, negative);
- After that, the average sentiment and a yearly sentiment breakdown (number of negative/positive sentences per year) are also computed.

To run this script:
python asf_public_discourse_home_decarbonisation/analysis/sentiment/sentence_sentiment_flair.py --data_source DATA_SOURCE --source_path SOURCE_PATH

where:
- DATA_SOURCE (required): the data source name, e.g. "mse" or "buildhub" or the name of another source of data
- SOURCE_PATH (optional): if data source is different from "mse"/"buildhub" then provide the path to the data source (local or S3).
"""

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
    Argparser function to parse arguments from the command line: data_source and source_path.

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
        required=False,  # making source_path optional
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
    if data_source == "buildhub":
        data["text"] = data["text"].astype(str)
        data.rename(columns={"url": "id"}, inplace=True)
        data.rename(columns={"date": "datetime"}, inplace=True)

    # Remove URLs
    data["text"] = data["text"].apply(remove_urls)

    # Convert text to lowercase
    data["text"] = data["text"].str.lower()

    # Process abbreviations
    data["text"] = data["text"].apply(process_abbreviations)

    # Remove username patterns
    data["text"] = data["text"].apply(remove_username_pattern)

    # Remove introduction patterns
    data["text"] = data["text"].apply(remove_introduction_patterns)

    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year

    return data


def compute_sentence_sentiment(data: pd.DataFrame, term: str) -> pd.DataFrame:
    """
    Computes sentiment for each sentence in the data containing the term.

    Args:
        data (pd.DataFrame): dataframe with columns "id", "year", "text", "is_original_post"
        term (str): term that needs to be present in sentence data
    Returns:
        pd.DataFrame: dataframe with one additional column, the sentiment column
    """
    data["sentences"] = data["text"].apply(sent_tokenize)

    data = (
        data[["id", "year", "sentences", "is_original_post"]]
        .explode("sentences")
        .reset_index(drop=True)
    )

    data = data[data["sentences"].str.contains(term)]

    data["sentiment"] = data["sentences"].apply(compute_sentiment_with_flair)

    return data


def compute_average_and_yearly_stats(data: pd.DataFrame, filter_term: str):
    """
    Computes average sentiment and yearly sentiment breakdown
    (i.e. number of sentences with POS/NEG sentiment per year).

    Args:
        data (pd.DataFrame): Data with id and sentiment columns

    Returns:
        pd.DataFrame: a dataframe with number of POSITIVE and NEGATIVE sentences per year
    """
    breakdown_per_year = (
        data.groupby(["year", "sentiment"])["id"].count().unstack(fill_value=0)
    )
    logger.info(
        f"Yearly sentiment breakdown for `{filter_term}` (# of sentences):\n{breakdown_per_year}"
    )

    breakdown_per_year_percent = (
        breakdown_per_year.div(breakdown_per_year.sum(axis=1), axis=0) * 100
    )
    logger.info(
        f"Yearly sentiment breakdown for `{filter_term}` (% of sentences):\n{breakdown_per_year_percent}"
    )

    avg_neg = len(data[data["sentiment"] == "NEGATIVE"]) / len(data) * 100
    logger.info(f"Average % of negative sentences (all time):\n{avg_neg}")

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
    data = prep_data_for_sentiment_analysis(data, data_source)

    # Compute sentiment for sentences containing specific terms
    for term in filter_terms:
        logger.info(f"Computing sentiment for term: `{term}`")
        filtered_data = data[data["text"].str.contains(term)]

        filtered_data = compute_sentence_sentiment(filtered_data, term)

        path_to_save_prefix = f"s3://{S3_BUCKET}/data/{data_source}/outputs/sentiment"
        filtered_data.to_csv(
            f"{path_to_save_prefix}/flair_sentence_sentiment_source_{data_source}_term_{term}.csv",
            index=False,
        )
        logger.info(
            f"Sentences and respective sentiment saved to `s3://{S3_BUCKET}/data/{data_source}/outputs/sentiment/`"
        )

        yearly_stats = compute_average_and_yearly_stats(filtered_data, term)

        yearly_stats.to_csv(
            f"{path_to_save_prefix}/flair_yearly_sentiment_stats_source_{data_source}_term_{term}.csv",
            index=False,
        )
        logger.info(
            f"Yearly sentiment stats saved to `s3://{S3_BUCKET}/data/{data_source}/outputs/sentiment/`"
        )
