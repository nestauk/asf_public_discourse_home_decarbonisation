"""
Script to compare sentiment for different technologies (heat pumps, solar panels and  boilers) in MSE data.
It computes the sentiment for sentences containing mentions of the technologies and saves the results to S3.

To run the script, use the following command:
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment_technologies.py --start_date "YYYY-MM-DD" --end_date "YYYY-MM-DD"

[optional] add --test to run in test mode

Example usage:
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment_technologies.py --start_date "2016-01-01" --end_date "2024-05-23"
"""

# Package imports
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
import logging

logger = logging.getLogger(__name__)

# Local imports
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    save_to_s3,
)
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment import (
    SentenceBasedSentiment,
)
from asf_public_discourse_home_decarbonisation.utils.general_utils import list_chunks
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_text_prep_utils import (
    prepping_data_for_topic_analysis,
)
from asf_public_discourse_home_decarbonisation import config


def parse_arguments(parser):
    parser.add_argument(
        "--start_date",
        help="Analysis start date in the format YYYY-MM-DD. Default to None (all data)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--end_date",
        help="Analysis end date in the format YYYY-MM-DD. Defaults to None (all data)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--test",
        help="Run in test mode",
        action="store_true",
    )
    return parser.parse_args()


def compute_sentiment(sentences_data: list, chunk_size: int) -> pd.DataFrame:
    """
    Calls the SentenceBasedSentiment class to compute sentiment for a list of sentences.
    The sentences are processed in chunks of size chunk_size.

    Args:
        sentences_data (list): sentences data
        chunk_size (int): chunk size

    Returns:
        pd.DataFrame: dataframe with sentiment original sentences, sentiment and score
    """
    sentiment_model = SentenceBasedSentiment()
    all_sentiment = []
    for text in tqdm(list_chunks(sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    return all_sentiment


def prep_data_for_sentiment_model(
    mse_data: pd.DataFrame, tech: str, start_date: str, end_date: str, test: bool
) -> list:
    """
    Prepares the data for sentiment analysis by extracting sentences containing mentions of the technology
    and removing duplicates.

    Args:
        mse_data (pd.DataFrame): dataframe with mse data
        tech (str): technology e.g. "heat pump"
        start_date (str): start date of the analysis in the format YYYY-MM-DD
        end_date (str): end date of the analysis in the format YYYY-MM-DD
        test (bool): True for test mode

    Returns:
        list: list of sentences
    """
    sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        tech,
        start_date,
        end_date,
        only_keep_sentences_with_expression=True,
    )

    if test:
        sentences_data = sentences_data.head(50)

    sentences_data = list(sentences_data.drop_duplicates("sentences")["sentences"])

    logger.info(f"Number of unique sentences for {tech} tech: {len(sentences_data)}")

    return sentences_data


if __name__ == "__main__":
    chunk_size = 100
    parser = ArgumentParser()
    args = parse_arguments(parser)

    mse_data = get_mse_data(
        category="all", collection_date=config["latest_data_collection_date"]["mse"]
    )

    path_to_save_prefix = f"data/mse/outputs/sentiment/comparing_technologies/{args.start_date}_{args.end_date}"

    technologies = ["heat pump", "solar panel", "boiler"]

    for tech in technologies:
        logger.info(f"Computing sentiment for {tech} sentences.")
        sentences_data = prep_data_for_sentiment_model(
            mse_data=mse_data,
            tech=tech,
            start_date=args.start_date,
            end_date=args.end_date,
            test=args.test,
        )

        all_sentiment = compute_sentiment(sentences_data, chunk_size)

        tech_filename = tech.replace(" ", "_")
        output_name = (
            f"{path_to_save_prefix}_mse_{tech_filename}_sentences_sentiment.csv"
        )

        if not args.test:
            save_to_s3(
                S3_BUCKET,
                all_sentiment,
                output_name,
            )
        logger.info(f"Sentiment for {tech} sentences completed.")
