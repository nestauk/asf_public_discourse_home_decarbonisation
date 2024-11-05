# 🗣️🌿 Understanding Public Discourse on Home Decarbonisation 🌿🗣️

The `asf_public_discourse_home_decarbonisation` repository contains code to **analyse public discourse data from online forums** to identify:

- **Topics of conversation**: We use [BERTopic](https://maartengr.github.io/BERTopic/index.html) (a topic modelling technique) to identify topics of conversation in online forum threads. We apply this to sentences from posts and replies and to forum post titles. We then manually rename these topics.

- **Frequently asked questions posted in the forum**: We collect questions asked in forum posts or replies, and apply BERTopic to identify similar groups of questions, leveraging [OpenAI's large language models](https://openai.com/) to create the final topic representations (i.e. our frequently asked questions).

- **Sentiment analysis of topics of conversation**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) sentiment model is used to identify sentiment of sentences and topics (by aggregating the sentiment of sentences in specific topics). This allows us to identify issues raised by those posting in the forums, as well as identifying positive experiences.

- **How all the above change over time**: all sentences and posts are associated with a date/time so we can track how topics, sentiment and questions change over time.

For the purpose of this project, the analyses are focused on conversations about **home heating** and **home decarbonisation**. However, _the pipelines created here can be applied to any domain area_.

In the context of the 🌿 **[sustainable future mission](https://www.nesta.org.uk/sustainable-future/)** 🌿 work, the analyses in this codebase allow us to:

- understand the current barriers to installing low-carbon heating technologies, such as regulatory constraints or installation challenges with heat pumps, and identify the skills/training needed by heating engineers to help them transition to low carbon heating;
- identify unknown issues faced by homeowners when installing low carbon heating tech in their home and questions they typically have;
- identify the common misconceptions around low-carbon heating tech;
- identify frequently asked questions by engineers installing low carbon heating tech;
- track the dominant narratives about heat pumps (and other heating technologies) across online forums.

You can read more about this project [here](https://www.nesta.org.uk/project/understanding-public-discourse-on-home-decarbonisation/). For a written overview of the technical work see this technical appendix (coming soon!).

## 🗂️ Repository structure

Below we have the repository structure and we highlight a few of the key folders and scripts:

```
asf_public_discourse_home_decarbonisation
├───analysis/
│    Analysis scripts
│    ├─ final_analyses/ - folder with publishable analyses
├───config/
│    Configuration scripts
│    ├─ base.yaml - should be updated with latest data collection date
├───getters/
│    Scripts with functions to load data from S3
│    ├─ getter_utils.py - general getter utils
│    ├─ public_discourse_getters.py - getter functions for any public discourse data
├───notebooks/
│    Notebooks with prototype code for pipeline
├───pipeline/
│    Subdirs with scripts to process data and produce outputs
│    ├─ data_processing_flows/ - processing text data from forums
│    ├─ bert_topic_analysis/ - topic analysis identification
|    |    |- sentence_topic_analysis.py - identifying topics of conversation from sentences in forum conversations
|    |    |- title_topic_analysis.py - identifying topics of conversation in forum post titles
│    ├─ faqs_identification/ - frequently asked questions identification
|    |    |- faq_identification.py - identifying frequently asked questions in forum conversations
│    ├─ sentiment/ - scripts to compute sentiment
|    |    |- sentence_sentiment.py - computing sentiment of sentences
│    ├─ stats/ - scripts to compute stats
|    |    |- data_source_stats.py - computing data source stats
│    ├─ README.md - instructions to run the different pipelines
├───utils/
│    Utils scripts for cleaning and visualising text data, applying topic analysis and sentiment analysis.
```

## 🆕 Latest data

❗📢 When new data is available please open a new issue/PR and update the configs [here](https://github.com/nestauk/asf_public_discourse_home_decarbonisation/blob/dev/asf_public_discourse_home_decarbonisation/config/base.yaml).

Latest data collection:

- Money Saving Expert: 2024/06/03
- Buildhub: 2024/05/23

Data collection scripts are available in [this GitHub repository](https://github.com/nestauk/asf_public_discourse_web_scraping) for Nesta employees. The data collection code will not be shared publicly.

## 🆕 Latest analyses

This section highlights the latest analyses conducted that will feature in publishable outputs.

### ♨️ Heat pump topic & sentiment analysis: identifying topics of conversation in heat pump conversations and respective sentiment

#### Topic analysis identification

To identify topics of conversation in heat pump Money Saving Expert forum threads, we run:

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_date "2016-01-01" --end_date "2024-05-22" --filter_by_expression "heat pump"
```

To identify topics of conversation in heat pump Buildhub forum threads, we run:

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "buildhub" --start_date "2016-01-01" --end_date "2024-05-22" --filter_by_expression "heat pump"
```

The above will run the pipelines for topic analysis on sentence data, and the results will be saved to S3.

#### Sentiment analysis computation

To compute sentiment for sentences in the above topics, we run the following commands. These will save the sentiment results to S3.

For Money Saving Expert:

```
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source "mse" --filter_by_expression "heat pump" --start_date "2016-01-01" --end_date "2024-05-22"
```

For Buildhub:

```
python asf_public_discourse_home_decarbonisation/pipeline/sentiment/sentence_sentiment.py --source "buildhub" --filter_by_expression "heat pump" --start_date "2016-01-01" --end_date "2024-05-22"
```

#### Analysis results: topics & sentiment

To visualise the results run the following command to explore the notebook:

```
pip install jupytext
jupytext --to notebook asf_public_discourse_home_decarbonisation/analysis/final_analyses/2024_09_30_topics_and_sentiment.py
```

### 🏘 Home heating topic analysis: identifying size and growth of topics of conversation in the home heating domain

To identify topics of conversation in the home heating domain, we run the following

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/title_topic_analysis.py --source "mse" --end_date "2024-05-22" --min_topic_size 300
```

This will take all post titles since 2003 and identify topics of conversation.

To visualise the results run the following command to explore the notebook:

```
pip install jupytext
jupytext --to notebook asf_public_discourse_home_decarbonisation/analysis/final_analyses/2024_09_30_home_heating_topics_and_changes_over_time.py
```

### 🤔💭 Heat Pump FAQ analysis: Identifying frequently asked questions from home heating public discourse

[coming soon...]

## 🗞 Publications

- [Understand public discourse on home decarbonisation - Project page information](https://www.nesta.org.uk/project/understanding-public-discourse-on-home-decarbonisation/)
- [Navigating heat pump adoption: insights from homeowner discussions online](https://www.nesta.org.uk/data-visualisation-and-interactive/navigating-heat-pump-adoption-insights-from-homeowner-discussions-online/)
- [Smart prepayment customers’ experience of the Demand Flexibility Service](https://www.nesta.org.uk/report/smart-prepayment-customers-experience-of-the-demand-flexibility-service/) - featuring analysis of MSE forum posts, as a complement to interviews
- [coming soon...] Frequently asked questions by homeowners
- [coming soon...] Medium blog on BERTopic for topic analysis and use cases
- [coming soon...] Medium blog on identifying frequently asked questions using BERTopic

## ⚙️ Setup

- Meet the data science cookiecutter [requirements](http://nestauk.github.io/ds-cookiecutter/quickstart), in brief:
  - Install: `direnv` and `conda`
- Run `make install` to configure the development environment:
  - Setup the conda environment
  - Configure `pre-commit`
- Download and install the [Averta font](https://github.com/deblynprado/neon/blob/master/fonts/averta/Averta-Regular.ttf)
- Download spacy model: `python -m spacy download en_core_web_sm`

## 📢 Contributor guidelines

[Technical and working style guidelines](https://github.com/nestauk/ds-cookiecutter/blob/master/GUIDELINES.md)

---

<small><p>Project based on <a target="_blank" href="https://github.com/nestauk/ds-cookiecutter">Nesta's data science project template</a>
(<a href="http://nestauk.github.io/ds-cookiecutter">Read the docs here</a>).
</small>
