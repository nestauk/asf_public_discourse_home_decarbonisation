# 😄 Sentiment analysis

Pipeline for assessing sentiment (negative, positive and neutral) in public discourse data.

## 👍👎 Sentence level sentiment (using Twitter-roBERTa-base for Sentiment Analysis)

Using `sentence_sentiment.py`

You can use the `SentenceBasedSentiment` class defined in the script to get sentiment and respective probabilities for a list of sentences. The model used is [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest). See the example below:

```
from asf_public_discourse_home_decarbonisation.pipeline.sentiment.sentence_sentiment import SentenceBasedSentiment
sentiment_model = SentenceBasedSentiment(process_data=False)
texts = ["This is a really great sentence", "This sentence is awful", "Cat"]
sentiment_scores = sentiment_model.get_sentence_sentiment(texts)
>> [("This is a really great sentence", 'positive', 0.97741115), ("This sentence is awful", 'negative', 0.9255473), ("Cat", 'neutral', 0.6470574)]
```

The output is a list of tuples where each tuple contains the sentence, sentiment label and the probability of the sentiment label.

Alternatively yu can also run the script from the command line to compute the sentiment of sentences used in the topic analysis:

```
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source SOURCE --filter_by_expression FILTER_BY_EXPRESSION --start_date START_DATE --end_date END_DATE --process_data PROCESS_DATA --relevant_clusters RELEVANT_CLUSTERS --irrelevant_clusters IRRELEVANT_CLUSTERS
where
- SOURCE is the source of the data e.g. "mse" or "buildhub"
- [optional] FILTER_BY_EXPRESSION is the expression to filter by e.g. "heat pump"
- [optional] START_DATE is the analysis start date in the format YYYY-MM-DD. Default to None (i.e. all data)
- [optional] END_DATE is the analysis end date in the format YYYY-MM-DD. Defaults to None (i.e. all data)
- [optional] RELEVANT_CLUSTERS is the clusters to keep e.g. "1,2,10"
- [optional] IRRELEVANT_CLUSTERS is the clusters to remove e.g. "1,2,10"
```

## 👍👎 Sentence level sentiment (using Flair)
