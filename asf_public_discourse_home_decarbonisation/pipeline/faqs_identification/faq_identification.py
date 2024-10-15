"""
Script for identifying frequently asked questions (FAQs) when directly reading from existing sentences data.
It requires the dataset with sentences to be already available in the S3 bucket.

The pipeline:
- Gets sentences data
- Extracts questions
- Identifies questions topics using BERTopic
    1) and a the default bag-of-words representation model
    2) and OpenAI's representation model to create a representative question for each FAQ topic
- Saves the topics and documents information in the S3 bucket

To run this script:
python asf_public_discourse_home_decarbonisation/pipeline/faqs_identification/faq_identification.py --source SOURCE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --filter_by_expression FILTER_BY_EXPRESSION --start_date START_DATE --end_date END_DATE --min_topic_size MIN_TOPIC_SIZE --openai_key OPENAI_KEY

where:
- SOURCE is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] FILTER_BY_EXPRESSION is the expression to filter by. Defaults to 'heat pump'
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 75.
- OPENAI_KEY is the OpenAI key in the format "sk-YOUR_API_KEY"

Example for MSE:
python asf_public_discourse_home_decarbonisation/pipeline/faqs_identification/faq_identification.py --source "mse" --start_date "2016-01-01" --end_date "2024-05-23"
"""

import pandas as pd
import math
import argparse
import logging
import openai
from bertopic.representation import OpenAI
import os
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.pipeline.faqs_identification.openai_prompt import (
    representative_question_prompt,
)
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    update_topics_with_duplicates,
    update_docs_with_duplicates,
    topic_model_definition,
    get_outputs_from_topic_model,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        help="`mse` or `buildhub`",
        required=True,
    )
    parser.add_argument(
        "--reduce_outliers_to_zero",
        help="True to reduce outliers to zero. Defaults to False",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--filter_by_expression",
        help="Expression to filter by. Defaults to 'heat pump'",
        default="heat pump",
    )
    parser.add_argument(
        "--start_date",
        help="Analysis start date in the format YYYY-MM-DD. Default to None (all data)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--end_date",
        help="Analysis end date in the format YYYY-MM-DD. Defaults to None (all data)",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--min_topic_size",
        help="Minimum topic size. Defaults to 75",
        default=75,
        type=int,
    )
    parser.add_argument(
        "--openai_key", help="OpenAI key", default=os.environ.get("OPENAI_KEY")
    )
    return parser.parse_args()


def run_faq_identification(
    min_topic_size: int,
    docs: list,
    questions_data: pd.DataFrame,
    reduce_outliers_to_zero: bool,
    representation_model=None,
) -> tuple:
    """
    Identifies frequently asked questions (FAQs) using BERTopic.

    Args:
        min_topic_size (int): minimum size of topics
        docs (list): the list of documents to group i.e. questions
        questions_data (pd.DataFrame): dataframe with the questions data
        reduce_outliers_to_zero (bool): True to reduce outliers to zero.
        representation_model (OpenAI, optional): OpenAI representation model. Defaults to None.

    Returns:
        (pd.DataFrame, pd.DataFrame): topics_info and doc_info
    """
    if representation_model is None:
        topic_model = topic_model_definition(min_topic_size)
    else:
        topic_model = topic_model_definition(min_topic_size, representation_model)
    topics, probs = topic_model.fit_transform(docs)
    topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

    logger.info(f"Number of topics: {len(topics_info) - 1}")
    logger.info(
        f"% of outliers: {topics_info[topics_info['Topic'] == -1]['%'].values[0]}"
    )

    if reduce_outliers_to_zero:
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
        topic_model.update_topics(docs, topics=new_topics)
        topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)
        topics_info.sort_values("Count", ascending=False)

        logger.info(f"Number of topics: {len(topics_info) - 1}")
        logger.info(
            f"% of outliers: {topics_info[topics_info['Topic'] == -1]['%'].values[0]}"
        )

    topics_info = update_topics_with_duplicates(topics_info, doc_info, questions_data)
    doc_info = update_docs_with_duplicates(doc_info, questions_data)

    return topics_info, doc_info


if __name__ == "__main__":
    # Dealing with user defined arguments
    args = parse_arguments()
    source = args.source
    reduce_outliers_to_zero = args.reduce_outliers_to_zero
    filter_by_expression = args.filter_by_expression
    start_date = args.start_date
    end_date = args.end_date
    min_topic_size = args.min_topic_size
    openai_key = args.openai_key

    sentences_data = pd.read_csv(
        f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_{start_date}_{end_date}_sentences_data.csv",
    )

    # Extract sentences which are question
    sentences_data["is_question"] = sentences_data["sentences"].str.contains("\?")
    questions_data = sentences_data[sentences_data["is_question"]]

    docs = list(questions_data.drop_duplicates("sentences")["sentences"])
    dates = list(questions_data.drop_duplicates("sentences")["date"])

    # Identifying topics without using OpenAI's representation model
    topics_info, doc_info = run_faq_identification(
        min_topic_size=min_topic_size,
        docs=docs,
        questions_data=questions_data,
        reduce_outliers_to_zero=reduce_outliers_to_zero,
    )

    path_to_save_prefix = f"s3://{S3_BUCKET}/data/{source}/outputs/faqs/{source}_{filter_by_expression}_{start_date}_{end_date}"
    topics_info.to_csv(
        f"{path_to_save_prefix}_FAQ_topics_info_no_representation.csv",
        index=False,
    )
    doc_info.to_csv(
        f"{path_to_save_prefix}_FAQ_docs_info_no_representation.csv",
        index=False,
    )

    # Identifying topics using LLMs representation model
    # the topics should be the same as above, but a different representation model is used
    client = openai.OpenAI(api_key=openai_key)
    representation_model = OpenAI(
        client,
        model="gpt-4o-mini",
        chat=True,
        prompt=representative_question_prompt,
        nr_docs=max(
            10, math.ceil(min_topic_size * 0.1)
        ),  # 10% of the minimum topic size, or 10 if 10% is less than 10
        delay_in_seconds=3,
        diversity=0.1,
    )

    topics_info_OpenAI, doc_info_OpenAI = run_faq_identification(
        min_topic_size=min_topic_size,
        docs=docs,
        questions_data=questions_data,
        reduce_outliers_to_zero=reduce_outliers_to_zero,
        representation_model=representation_model,
    )

    topics_info_OpenAI.to_csv(
        f"{path_to_save_prefix}_FAQ_topics_info_OpenAI_representation.csv",
        index=False,
    )
    doc_info_OpenAI.to_csv(
        f"{path_to_save_prefix}_FAQ_docs_info_OpenAI_representation.csv",
        index=False,
    )
