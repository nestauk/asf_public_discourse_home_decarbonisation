# 🧰 Utility functions

This folder contains scripts with utils for procesing text data, running topic analysis and plotting results.

## General utils, `general_utils.py`

General utils used across different pipelines.

## Text processing utils, `text_processing_utils.py`

Forum analysis specific text processing utils: such as removing URLs, processing abbreviations and checking if a sentence ends with punctuation.

## Preparing data for topic analysis utils, `topic_analysis_text_prep_utils.py`

Utils for preparing forum data for topic analysis: cleaning text, creating a dataset with sentences from full posts and removing small sentences from datasets. The main function in this script is `prepping_data_for_topic_analysis()`.

## Topic analysis utils, `topic_analysis_utils.py`

Topic analysis utils (specific for [BERTopic](https://maartengr.github.io/BERTopic/index.html)) such as: topic model definition and updating topics and docs with duplicate numbers\*.

\*duplicate numbers: topic analysis uses unique documents has input. Then we update the number of documents in each topic with the number of duplicates in the original dataset.

## Sentiment analysis utils, `sentiment_analysis_utils.py`

Utils for computing sentiment using [Flair library](https://flairnlp.github.io/docs/tutorial-basics/tagging-sentiment).

## Plotting utils, `plotting_utils.py`

Plotting utils for visualising the results from exploratory data analysis and initial text analysis (frequencies of words and n-grams) including functions for: visualising n-gram frequencies, creating wordclouds, distributions of posts and users.

## N-gram utils, `ngram_utils.py`

Utils for dealing with words/tokens and n-grams.

## Processing text data and n-grams, `preprocessing_utils.py` [legacy script]

Utils for preprocessing text data and dealing with n-grams.
