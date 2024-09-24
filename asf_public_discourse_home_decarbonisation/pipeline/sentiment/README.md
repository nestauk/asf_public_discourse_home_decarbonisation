# 😄 Sentiment analysis

Scripts for assessing sentiment (negative, positive and neutral) of public discourse data.

## 👍👎 Sentence level sentiment (using Twitter-roBERTa-base for Sentiment Analysis), `sentence_sentiment.py`

You can use the `SentenceBasedSentiment` class defined in the script to get sentiment and respective probabilities for a list of sentences. The model used is [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). See the example below:

```
from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment import SentenceBasedSentiment
sentiment_model = SentenceBasedSentiment()
texts = ["This is a really great sentence", "This sentence is awful", "Cat"]
sentiment_scores = sentiment_model.get_sentence_sentiment(texts)
>> [("This is a really great sentence", 'positive', 0.97741115), ("This sentence is awful", 'negative', 0.9255473), ("Cat", 'neutral', 0.6470574)]
```

The output is a list of tuples where each tuple contains the sentence, sentiment label and the probability of the sentiment label.

Alternatively you can also run the script from the command line to compute the sentiment of sentences used in the topic analysis:

```
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source SOURCE --filter_by_expression FILTER_BY_EXPRESSION --start_date START_DATE --end_date END_DATE --relevant_clusters RELEVANT_CLUSTERS --irrelevant_clusters IRRELEVANT_CLUSTERS
```

where

- `SOURCE` is the source of the data e.g. "mse" or "buildhub"
- [optional] `FILTER_BY_EXPRESSION` is the expression to filter by e.g. "heat pump"
- [optional] `START_DATE` is the analysis start date in the format YYYY-MM-DD. Default to None (i.e. all data)
- [optional] `END_DATE` is the analysis end date in the format YYYY-MM-DD. Defaults to None (i.e. all data)
- [optional] `RELEVANT_CLUSTERS` is the clusters to keep e.g. "1,2,10"
- [optional] `IRRELEVANT_CLUSTERS` is the clusters to remove e.g. "1,2,10"

## 🔌 Sentiment for different technologies, `sentence_sentiment_technologies.py`

Compares sentiment for different technologies (heat pumps, solar panels and boilers) in MSE data. It computes the sentiment for sentences containing mentions of the technologies and saves the results to S3.

To run the script, use the following command:

```
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment_technologies.py --start_date "YYYY-MM-DD" --end_date "YYYY-MM-DD"
```

## 👍👎 Sentence level sentiment and changes over time (using Flair), `sentence_sentiment_flair.py` [legacy script]

Computes sentiment at the sentence level for sentences containing specific terms such as "heat pump" and "boiler". It then computes the number of positive and negative sentences per year for each specific search term.

To run the script,
`python asf_public_discourse_home_decarbonisation/analysis/sentiment/sentence_sentiment_flair.py --data_source DATA_SOURCE --source_path SOURCE_PATH`
where

- `DATA_SOURCE`: the data source name, e.g. "mse" or "buildhub" or the name of another source of data
- [optional] `SOURCE_PATH`: if data source is different from "mse"/"buildhub" then provide the path to the data source (local or S3).

_Example:_
`python asf_public_discourse_home_decarbonisation/analysis/sentiment/sentiment_analysis.py --data_source "mse"` which will output and save two dataframes:

- sentiment label (NEGATIVE/POSITIVE) for each sentence matching the search terms ("heat pump" and "boiler")
- the number of NEGATIVE and POSITIVE sentences per year for each of the search terms ("heat pump" and "boiler").
