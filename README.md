# 🗣️🌿 Understanding Public Discourse on Home Decarbonisation 🌿🗣️

The `asf_public_discourse_home_decarbonisation` repository contains code to **analyse public discourse data from online forums** to identify:
- **Topics of conversation**: We use [BERTopic](https://maartengr.github.io/BERTopic/index.html) (a topic modelling technique) to identify topics of conversation in online forum threads. We apply this to sentences from posts and replies and to forum post titles. We then manually rename these topics.

- **Frequently asked questions posted in the forum**: We collect questions asked in forum posts or replies, and apply BERTopic to identify similar groups of questions, leveraging [OpenAI's large language models](https://openai.com/) to create the final topic representations (i.e. our frequently asked questions).

- **Sentiment analysis of topics of conversation**: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) sentiment model is used to identify sentiment of sentences and topics (by aggregating the sentiment of sentences in specific topics). This allows us to identify issues raised by those posting in the forums, as well as identify positive experiences.

- **How all the above change over time**: all sentences and posts are associated with a date/time so we can track how topics, sentiment and questions changes over time.

For the purpose of this project, the analyses are focused on conversations about **home heating** and **home decarbonisation**. However, *the pipelines created here can be applied to any domain area*.

In the context of the 🌿 **[sustainable future mission](https://www.nesta.org.uk/sustainable-future/)** 🌿 work, the analyses in this codebase allow us to:
- understand existing difficulties around installing low-carbon heating tech, such as heat pumps, and identify the skills/training needed by heating engineers to help them transition to low carbon heating;
- identify unknown issues faced by homeowners when installing low carbon heating tech in their home and questions they typically have;
- identify the common misconceptions around low-carbon heating tech;
- identify frequently asked questions by engineers installing low carbon heating tech;
- track the dominant narratives about heat pumps (and other heating technologies) across online forums.

You can read more about this project [here](https://www.nesta.org.uk/project/understanding-public-discourse-on-home-decarbonisation/). For a written overview of the technical work see this technical appendix (coming soon!).

## 🗂️ Repository structure
[to be continued]

Below we have the repository structure and we highlight a few of the key scripts:
```
asf_public_discourse_home_decarbonisation
├───analysis/
│    Analysis scripts
├───config/
│    Configuration scripts
│    ├─ base.yaml - should be updated with latest data collection date
├───getters/
│    Scripts with functions to load data
│    ├─ getter_utils.py - general getter utils
│    ├─ mse_getters.py - Money Saving Expert getters
│    ├─ bh_getters.py - Buildhub getters
├───notebooks/
│    Notebooks with prototype code for pipeline
├───pipeline/
│    Subdirs with scripts to process data and produce outputs
│    ├─ data_processing_flows/ - processing text data from forums
│    ├─ bert_topic_analysis/ - topic analysis identification
|    |    |- sentence_topic_analysis.py - identifying topics of conversation from sentences in forum conversations
│    ├─ faqs_identification/ - frequently asked questions identification
|    |    |- faq_identification.py - identifying frequently asked questions in forum conversations
│    ├─ sentiment/ - scripts to compute sentiment
|    |    |- sentence_sentiment.py - computing sentiment of sentences
│    ├─ README.md - instructions to run the different pipelines
├───utils/
│    Utils scripts for cleaning and visualising text data, applying topic analysis and sentiment analysis.
```

## 🆕 Latest analyses
[to be completed...]

❗📢 When new data is available please open a new issue/PR and update the configs [here](https://github.com/nestauk/asf_public_discourse_home_decarbonisation/blob/dev/asf_public_discourse_home_decarbonisation/config/base.yaml).

Latest data collection:
- Money Saving Expert: DD-MM-YYY
- Buildhub: DD-MM-YY

### Heat pump topic analysis: identifying topics of conversation in heat pump conversations

[missing some context into what happens in each analysis

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_date "" --end_date "" --filter_by_expression "heat pump"
```

```
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "buildhub" --start_date "" --end_date "" --filter_by_expression "heat pump"
```


### Heat pump sentiment analysis: identifying negative, positive and neutral topics of conversation

### Home heating topic analysis: identifying size and growth of topics of conversation in the home heating domain


## 🗞 Publications
- [Project page information](https://www.nesta.org.uk/project/understanding-public-discourse-on-home-decarbonisation/)
- [coming soon...] Navigating heat pump adoption: Insights from homeowner discussions online
- [coming soon...] Insights from online forum home heating conversations
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
