# 📚 Topic analysis

This folder contains scripts to perform topic analysis on text data, using [BERTopic](https://maartengr.github.io/BERTopic/index.html), and evaluate the results.

## Identifying topics of conversation in sentences from forum conversations

Using `sentence_topic_analysis.py`

This script performs topic analysis on sentences extracted from forum text using BERTopic. To run the script:

`python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source SOURCE --start_date START_DATE --end_date END_DATE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --filter_by_expression FILTER_BY_EXPRESSION --min_topic_size MIN_TOPIC_SIZE`

where:

- SOURCE is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] FILTER_BY_EXPRESSION is the expression to filter by. Defaults to 'heat pump'.
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 100.

## Evaluating the results of difference models and hyperparamters

`evaluate_bertopic_results.py`
This script evaluates the results of the topic analysis on different metrics.

`bert_params_config.py`
File with parameters to test BERTopic in `evaluate_bertopic_results.py`
