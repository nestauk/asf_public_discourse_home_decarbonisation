# 📚 Topic analysis identification and evaluation

This folder contains scripts to perform topic analysis on text data, using [BERTopic](https://maartengr.github.io/BERTopic/index.html).

BERTopic is a topic modelling technique used to cluster text into easily interpretable topics. BERTopic finds hidden semantic patterns in text documents and assigns topics to them, without the need of a list of topics beforehand as input - the algorithm will define them automatically.

## Identify topics of conversation

There are two scripts to identify topics of conversation in forum conversations: one using sentences as input (from posts and replies) and another using posts' titles.

### 1. Identifying topics of conversation in sentences from forum conversations

Using `sentence_topic_analysis.py`

This script performs topic analysis on sentences extracted from forum text (original posts and respective replies). To run the script:

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source SOURCE --start_date START_DATE --end_date END_DATE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --filter_by_expression FILTER_BY_EXPRESSION --min_topic_size MIN_TOPIC_SIZE
```

where:

- `SOURCE` is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] FILTER_BY_EXPRESSION is the expression to filter by. Defaults to 'heat pump'.
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 100.

### 2. Identifying topics of conversation in forum post titles

Using `title_topic_analysis.py`

This script performs topic analysis using titles from posts as input. To run the script:

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/title_topic_analysis.py --source SOURCE --start_date START_DATE --end_date END_DATE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --min_topic_size MIN_TOPIC_SIZE
```

where:

- `SOURCE` is the source of the data: `mse` or `buildhub`
- [optional] `START_DATE` is the start date of the analysis in the format YYYY-MM-DD
- [optional] `END_DATE` is the end date of the analysis in the format YYYY-MM-DD
- [optional] `REDUCE_OUTLIERS_TO_ZERO` is `True` to reduce outliers to zero. Defaults to `False`
- [optional] `MIN_TOPIC_SIZE` is the minimum size of a topic. Defaults to 100.

## Evaluating the results of difference models and hyperparamters

Using `evaluate_bertopic_results.py`

This script evaluates the results of the topic analysis on different metrics (percentage of outliers and average probablity of belonging to a non-outlier topic) by changing BERTopic model specification such as:

- number of topics
- representation model
- minimum topic size
- vectorizer model
- UMAP model
- HDBSCAN model
- embedding model

To run the script:

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/evaluate_bertopic_results.py --n_runs N_RUNS --path_to_config_file CONFIG_FILE_PATH --path_to_data_file PATH_DATA
```

where

- `N_RUNS` is the number of times to run model -[optional] `CONFIG_FILE_PATH` is location of your configuration file in the repository structure, defaults to `"asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.bert_params_config"`
  Note: you don't need the file extension .py at the end. -[optional] `PATH_DATA` if not reading standard forum data (e.g. if reading questions data). Defaults to `None`. If `None`, then `source_name` as set in the `CONFIG_FILE_PATH` will be used instead to read the data.

The file `bert_params_config.py` is the configuration file with the hyperparameters to evaluate.
