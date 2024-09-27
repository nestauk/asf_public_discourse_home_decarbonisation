"""
Utility functions for computing sentiment.
"""

from flair.nn import Classifier
from flair.data import Sentence


sentiment_model = "sentiment-fast"
tagger = Classifier.load(sentiment_model)


def compute_sentiment_with_flair(sentence: str) -> str:
    """
    Computes the sentiment of a sentence using the Flair library.
    Args:
        sentence (str): the sentence to compute the sentiment for
    Returns:
        str: a POSITIVE or NEGATIVE sentiment label
    """
    sentence = Sentence(sentence)
    tagger.predict(sentence)
    return sentence.labels[0].value
