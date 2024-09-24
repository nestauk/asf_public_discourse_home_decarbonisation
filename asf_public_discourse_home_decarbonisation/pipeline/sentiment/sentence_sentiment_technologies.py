"""
"""

# Package imports
from tqdm import tqdm
import pandas as pd

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


if __name__ == "__main__":

    chunk_size = 100

    mse_data = get_mse_data(category="all", collection_date="2024_06_03")

    hp_sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        "heat pump",
        "2018-01-01",
        "2024-05-22",
        only_keep_sentences_with_expression=True,
    )

    solar_sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        "solar panel",
        "2018-01-01",
        "2024-05-22",
        only_keep_sentences_with_expression=True,
    )

    boilers_sentences_data = prepping_data_for_topic_analysis(
        mse_data,
        "boiler",
        "2018-01-01",
        "2024-05-22",
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

    output_name = f"data/mse/outputs/sentiment/comparing_technologies/mse_boiler_sentences_sentiment.csv"

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

    output_name = f"data/mse/outputs/sentiment/comparing_technologies/mse_solar_panel_sentences_sentiment.csv"

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )

    # HPS
    all_sentiment = []
    for text in tqdm(list_chunks(hp_sentences_data, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    output_name = f"data/mse/outputs/sentiment/comparing_technologies/mse_heat_pump_sentences_sentiment.csv"

    save_to_s3(
        S3_BUCKET,
        all_sentiment,
        output_name,
    )
