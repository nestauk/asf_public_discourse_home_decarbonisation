"""
Script to compare sentiment for different technologies (heat pumps, solar panels and  boilers) in MSE data.
It computes the sentiment for sentences containing mentions of the technologies and saves the results to S3.

To run the script, use the following command:
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment_technologies.py --start_date "YYYY-MM-DD" --end_date "YYYY-MM-DD"

Example usage:
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment_technologies.py --start_date "2016-01-01" --end_date "2024-05-22"
"""

# Package imports
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser

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
    return parser.parse_args()


if __name__ == "__main__":
    chunk_size = 100
    parser = ArgumentParser()
    args = parse_arguments(parser)

    mse_data = get_mse_data(
        category="all", collection_date=config["latest_data_collection_date"]["mse"]
    )

    hp_sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        "heat pump",
        args.start_date,
        args.end_date,
        only_keep_sentences_with_expression=True,
    )

    solar_sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        "solar panel",
        args.start_date,
        args.end_date,
        only_keep_sentences_with_expression=True,
    )

    boilers_sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        "boiler",
        args.start_date,
        args.end_date,
        only_keep_sentences_with_expression=True,
    )

    hp_sentences_data = list(
        hp_sentences_data.drop_duplicates("sentences")["sentences"]
    )
    solar_sentences_data = list(
        solar_sentences_data.drop_duplicates("sentences")["sentences"]
    )
    boilers_sentences_data = list(
        boilers_sentences_data.drop_duplicates("sentences")["sentences"]
    )

    sentiment_model = SentenceBasedSentiment()

    # BOILERS
    all_sentiment = []
    for text in tqdm(list_chunks(boilers_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    path_to_save_prefix = f"data/mse/outputs/sentiment/comparing_technologies/{args.start_date}_{args.end_date}"
    output_name = f"{path_to_save_prefix}_mse_boiler_sentences_sentiment.csv"

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )

    # SOLAR PANELS
    all_sentiment = []
    for text in tqdm(list_chunks(solar_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    output_name = f"{path_to_save_prefix}_mse_solar_panel_sentences_sentiment.csv"

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )

    # Heat pumps
    all_sentiment = []
    for text in tqdm(list_chunks(hp_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    output_name = f"{path_to_save_prefix}_mse_heat_pump_sentences_sentiment.csv"

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )
