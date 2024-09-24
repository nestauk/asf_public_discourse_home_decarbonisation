"""
Script for identifying topics of conversation in forum sentences.

The pipeline:
- Gets forum data (including original posts and replies)
- Cleans and enhances the forum data
- Breaks the forum text data into sentences
- Applies topic analysis to unique sentences to identify topics of conversation
- Outputs are saved to S3 including information about topics the sentences data

To run this script:
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source SOURCE --start_date START_DATE --end_date END_DATE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --filter_by_expression FILTER_BY_EXPRESSION --min_topic_size MIN_TOPIC_SIZE

where:
- SOURCE is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] FILTER_BY_EXPRESSION is the expression to filter by. Defaults to 'heat pump'
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 100.

Examples for MSE:
2018-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_date "2018-01-01" --end_date "2024-05-22"
2016-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "mse" --start_date "2016-01-01" --end_date "2024-05-23"

Examples for Buildhub:
2018-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "buildhub" --start_date "2018-01-01" --end_date "2024-05-22"
2016-2024 analysis: python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/sentence_topic_analysis.py --source "buildhub" --start_date "2016-01-01" --end_date "2024-05-23"
"""

# Package imports
import argparse
import logging
from asf_public_discourse_home_decarbonisation import config

logger = logging.getLogger(__name__)

# Local imports
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
)
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_text_prep_utils import (
    prepping_data_for_topic_analysis,
)
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    update_topics_with_duplicates,
    update_docs_with_duplicates,
    topic_model_definition,
)


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
        help="Minimum topic size. Defaults to 100",
        default=100,
        type=int,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Dealing with user defined arguments
    args = parse_arguments()
    source = args.source
    reduce_outliers_to_zero = args.reduce_outliers_to_zero
    filter_by_expression = args.filter_by_expression
    start_date = args.start_date
    end_date = args.end_date
    min_topic_size = args.min_topic_size

    # Reading data
    if source == "mse":
        forum_data = get_mse_data(
            category="all",
            collection_date=config["latest_data_collection_date"]["mse"],
            processing_level="raw",
        )
    elif source == "buildhub":
        forum_data = get_bh_data(
            category="all",
            collection_date=config["latest_data_collection_date"]["buildhub"],
        )
        forum_data.rename(columns={"url": "id", "date": "datetime"}, inplace=True)
    else:
        raise ValueError("Invalid source")

    # Creating dataset of sentences and preparing inputs for topic analysis
    sentences_data = prepping_data_for_topic_analysis(
        forum_data,
        filter_by_expression,
        start_date,
        end_date,
        phrases_to_remove=["thank", "happy to help", "kind wishes", "kind regards"],
    )

    docs = list(sentences_data.drop_duplicates("sentences")["sentences"])
    dates = list(sentences_data.drop_duplicates("sentences")["date"])

    # Topic analysis
    topic_model = topic_model_definition(min_topic_size)
    topics, probs = topic_model.fit_transform(docs)
    topics_, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)

    # Logging relevant information
    logger.info(f"Number of topics: {len(topics_info) - 1}")
    logger.info(
        f"% of outliers: {topics_info[topics_info['Topic'] == -1]['%'].values[0]}"
    )

    # Reducing outliers to zero where relevant
    if reduce_outliers_to_zero:
        new_topics = topic_model.reduce_outliers(docs, topics, strategy="embeddings")
        topic_model.update_topics(docs, topics=new_topics)
        topics, topics_info, doc_info = get_outputs_from_topic_model(topic_model, docs)
        topics_info.sort_values("Count", ascending=False)

    # Updating topics and docs information with duplicates (as only unique sentences are used for topic analysis)
    topics_info = update_topics_with_duplicates(topics_info, doc_info, sentences_data)
    doc_info = update_docs_with_duplicates(doc_info, sentences_data)

    path_to_save_prefix = f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/{source}_{filter_by_expression}_{start_date}_{end_date}"
    # Saving outputs to S3
    topics_info.to_csv(
        f"{path_to_save_prefix}_sentence_topics_info.csv",
        index=False,
    )
    doc_info.to_csv(
        f"{path_to_save_prefix}_sentence_docs_info.csv",
        index=False,
    )
    sentences_data.to_csv(
        f"{path_to_save_prefix}_sentences_data.csv",
        index=False,
    )
