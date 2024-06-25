#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

# sentences_data = pd.read_csv(
#    f"s3://asf-public-discourse-home-decarbonisation/data/buildhub/outputs/topic_analysis/buildhub_heat pump_sentences_data.csv"
# )

sentences_data = pd.read_csv(
    f"s3://asf-public-discourse-home-decarbonisation/data/mse/outputs/topic_analysis/mse_heat pump_sentences_data.csv"
)


# In[3]:


# Apply the function to extract questions
sentences_data["is_question"] = sentences_data["sentences"].str.contains("\?")
questions_data = sentences_data[sentences_data["is_question"]]


# In[4]:


sentences_data


# In[5]:


questions_data


# In[6]:


from asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.sentence_topic_analysis import (
    update_topics_with_duplicates,
    update_docs_with_duplicates,
    topic_model_definition,
    get_outputs_from_topic_model,
)
import logging
import openai
from bertopic.representation import OpenAI
from asf_public_discourse_home_decarbonisation import S3_BUCKET


logger = logging.getLogger(__name__)

prompt = """
In this topic, the following documents are a small but representative subset of all documents in the topic:
[DOCUMENTS]
The topic is described by the following keywords: [KEYWORDS]

Based on the information above, extract a short (less than 11 words) representative question in the following format:
topic: <Representative Question>
"""

client = "<INSERT OPENAI KEY>"
# representation_model = OpenAI(client, model="gpt-3.5-turbo", delay_in_seconds=10, chat=True)

representation_model = OpenAI(
    client,
    model="gpt-3.5-turbo",
    chat=True,
    prompt=prompt,
    nr_docs=10,
    delay_in_seconds=3,
)

print(type(representation_model))

docs = list(questions_data.drop_duplicates("sentences")["sentences"])
dates = list(questions_data.drop_duplicates("sentences")["date"])
min_topic_size = 75

topic_model = topic_model_definition(
    min_topic_size, representation_model=representation_model
)
topics, probs = topic_model.fit_transform(docs)
topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

logger.info(f"Number of topics: {len(topics_info) - 1}")
logger.info(f"% of outliers: {topics_info[topics_info['Topic'] == -1]['%'].values[0]}")
topics_info = update_topics_with_duplicates(topics_info, doc_info, questions_data)
doc_info = update_docs_with_duplicates(doc_info, questions_data)
source = "mse"
filter_by_expression = "heat pump"
topics_info.to_csv(
    f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_FAQ_topics_info.csv",
    index=False,
)
doc_info.to_csv(
    f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_FAQ_docs_info.csv",
    index=False,
)
