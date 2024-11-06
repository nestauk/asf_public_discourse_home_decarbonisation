# Pipeline

Main public discourse analysis pipelines are available in this folder.

Explore the folder structure below and, for more detail, read the README's in the relevant sections:

```
asf_public_discourse_home_decarbonisation/pipeline/
├───bert_topic_analysis/ - topic analysis identification and evaluation
│    ├─sentence_topic_analysis.py - identifying topics of conversation from sentences in forum conversations
│    ├─title_topic_analysis.py - identifying topics of conversation in forum post titles
│    ├─evaluate_bertopic_results.py - evaluating BERTopic results
│    ├─bert_params_config.py - script with hyper-parameters for BERTopic evaluation
├───sentiment/ - scripts to compute sentiment
|    |-sentence_sentiment.py - computing sentiment of sentences using cardiffnlp/twitter-roberta-base-sentiment-latest
|    |-sentence_sentiment_flair.py - computing sentiment of sentences using Flair
|    |-sentence_sentiment_technologies.py - computing sentiment of sentences mentioning different technologies (heat pumps, solar panels and  boilers) using cardiffnlp/twitter-roberta-base-sentiment-latest
├───faqs_identification/ - frequently asked questions identification [WIP]
|    |-extract_questions.py - extracting questions from forum conversations
├───stats/ - scripts to compute stats
|    |-data_source_stats.py - producing stats from public discourse data (number of posts, replies, etc)
├───data_processing_flows/ - scripts for processing text data from forums (not always required, only for certain tasks)
│    ├─processing_text_data.py - processing text data from forums
│    ├─text_processing_utils.py - utility functions for processing text data
│    ├─flow_requirements.txt - requirements for running processing_text_data.py
```
