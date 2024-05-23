"""
Apply a sentiment model to sentences and save scores.
Collect and save sentiment for sentences by running:

`python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source "mse"`

For other uses, the class can be used with:

from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment import SentenceBasedSentiment
sentiment_model = SentenceBasedSentiment(process_data=False)
texts = ["This is a really great sentence", "This sentence is awful", "Cat"]
sentiment_scores = sentiment_model.get_sentence_sentiment(texts)
>> [("This is a really great sentence", 'positive', 0.97741115), ("This sentence is awful", 'negative', 0.9255473), ("Cat", 'neutral', 0.6470574)]
"""

# Package imports
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import os
from argparse import ArgumentParser
from typing import Union
import pandas as pd

# Local imports
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    save_to_s3,
    load_s3_data,
)
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.utils.text_cleaning_utils import (
    preprocess,
)


class SentenceBasedSentiment(object):
    """Find sentiment scores for sentences"""

    def __init__(
        self,
        model_name="cardiffnlp/twitter-roberta-base-sentiment-latest",
        process_data=False,
    ):
        """
        Args:
            model_name (str, optional): Model name. Defaults to "cardiffnlp/twitter-roberta-base-sentiment-latest".
            process_data (bool, optional): True to process data, if not processed already. Defaults to False.
        """
        self.model_name = model_name
        self.process_data = process_data
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, model_max_length=512
        )
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def get_sentence_sentiment(self, texts: Union[list, str]) -> list:
        """
        Get the sentiment for a list of sentences using a pretrained sentiment model.
        Returns a list of tuples with sentence, sentiment and the score:
        e.g. ['sentence AAA', 'neutral', 0.6559918), ('sentence BBB', 'negative', 0.6319174)]
        """

        if isinstance(texts, str):
            texts = [texts]

        if self.process_data:
            texts = [preprocess(text) for text in texts]

        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        output = self.model(**encoded_input)

        scores = np.array([softmax(i) for i in output[0].detach().numpy()])

        results = list(
            zip(
                texts,
                [self.config.id2label[i] for i in np.argmax(scores, axis=1)],
                np.max(scores, axis=1),
            )
        )

        return results


def list_chunks(orig_list: list, chunk_size: int = 100):
    """Chunks list into batches of a specified chunk_size."""
    for i in range(0, len(orig_list), chunk_size):
        yield orig_list[i : i + chunk_size]


def parse_arguments(parser):
    parser.add_argument(
        "--source",
        help="`mse` or `buildhub`",
        required=True,
    )
    parser.add_argument(
        "--process_data",
        help="True to process data, if not processed already. Defaults to False.",
        default=False,
    )
    parser.add_argument(
        "--relevant_clusters",
        help="Relevant clusters e.g. '1,2,10'. Defaults to None (all clusters).",
        default=None,
    )
    return parser.parse_args()


if __name__ == "__main__":

    parser = ArgumentParser()
    args = parse_arguments(parser)

    chunk_size = 100

    source = args.source
    input_path = (
        f"/data/{source}/outputs/topic_analysis/{source}_sentence_docs_info.csv"
    )

    sentences = load_s3_data(
        S3_BUCKET,
        input_path,
    )

    if args.relevant_clusters is not None:
        relevant_clusters = [int(i) for i in args.relevant_clusters.split(",")]
        sentences = sentences[sentences["Topic"].isin(relevant_clusters)]

    output_name = (
        f"/data/{source}/outputs/sentiment/{source}_sentence_topics_sentiment.csv"
    )

    sentences_texts = list(sentences["Document"].unique())

    sentiment_model = SentenceBasedSentiment()

    print(
        f"Finding sentiment for {len(sentences_texts)} sentences in {len(sentences_texts)/chunk_size} chunks"
    )

    all_sentiment = []
    for text in tqdm(list_chunks(sentences_texts, chunk_size=chunk_size)):
        sentiment_scores = sentiment_model.get_sentence_sentiment(text)
        all_sentiment += sentiment_scores

    all_sentiment = pd.DataFrame(all_sentiment, columns=["text", "sentiment", "score"])

    print(all_sentiment.head())

    # save_to_s3(
    #     S3_BUCKET,
    #     all_sentiment,
    #     output_name,
    # )
