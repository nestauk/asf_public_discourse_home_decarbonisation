# 🤔 Frequently asked questions (FAQ) identification

Folder with scripts to identify frequently asked questions from questions asked in forums as input data, making use of [BERTopic](https://maartengr.github.io/BERTopic/index.html) to group similar questions together.

## ❓ Identifying FAQs (using sentences as input)

Using `faq_identification.py`

This script performs topic analysis on questions data, to identify groups of frequently asked questions.
It uses sentences in forum discussions as input data, so it requires sentences data to be available on S3.
It saves the topics and documents information in the S3 bucket (with the option of having the LLM generated representative question as a column). 
To run the script:
`python asf_public_discourse_home_decarbonisation/pipeline/faqs_identification/faq_identification.py --source SOURCE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --filter_by_expression FILTER_BY_EXPRESSION --start_date START_DATE --end_date END_DATE --min_topic_size MIN_TOPIC_SIZE --openai_key OPENAI_KEY`

where:

- SOURCE is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] FILTER_BY_EXPRESSION is the expression to filter by. Defaults to 'heat pump'
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 75.
- OPENAI_KEY is the OpenAI key in the format "sk-YOUR_API_KEY"

## ❔ Identifying FAQs from scratch (input data: forum discussions)

[To be added in a separate PR]
