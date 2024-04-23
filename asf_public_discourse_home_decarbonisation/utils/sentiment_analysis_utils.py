from flair.nn import Classifier
from flair.data import Sentence


def compute_sentiment_with_flair(sentence, sentiment_model="sentiment-fast"):
    tagger = Classifier.load(sentiment_model)

    sentence = Sentence(sentence)
    tagger.predict(sentence)
    return sentence.labels[0].value
