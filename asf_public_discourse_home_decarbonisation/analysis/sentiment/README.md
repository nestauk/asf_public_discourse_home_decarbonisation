# 😄🫤 Sentiment analysis

This folder contains scripts for assessing sentiment in public discourse data.

## Sentence level sentiment analysis & changes in sentiment over time

`sentiment_analysis.py` computes sentiment at the sentence level for sentences containing specific terms such as "heat pump" and "boiler". It then computes the number of positive and negative sentences per year for each specific search term.

To run the script,
`python asf_public_discourse_home_decarbonisation/analysis/sentiment/sentiment_analysis.py --data_source DATA_SOURCE --source_path SOURCE_PATH`
where

- DATA_SOURCE (required): the data source name, e.g. "mse" or "buildhub" or the name of another source of data
- SOURCE_PATH (optional): if data source is different from "mse"/"buildhub" then provide the path to the data source (local or S3).

_Example:_ `python asf_public_discourse_home_decarbonisation/analysis/sentiment/sentiment_analysis.py --data_source "mse"` which will output and save two dataframes:

- sentiment label (NEGATIVE/POSITIVE) for each sentence matching the search terms ("heat pump" and "boiler")
- the number of NEGATIVE and POSITIVE sentences per year for each of the search terms ("heat pump" and "boiler").
