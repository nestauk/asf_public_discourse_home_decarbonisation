"""
Script for identifying topics of conversation using forum thread titles as input.

The pipeline:
- Gets forum data;
- Applies topic analysis to unique titles to identify topics of conversation
- Outputs are saved to S3 including information about topics the titles data

To run this script:
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/title_topic_analysis.py --source SOURCE --start_date START_DATE --end_date END_DATE --reduce_outliers_to_zero REDUCE_OUTLIERS_TO_ZERO --min_topic_size MIN_TOPIC_SIZE

where:
- SOURCE is the source of the data: `mse` or `buildhub`
- [optional] START_DATE is the start date of the analysis in the format YYYY-MM-DD
- [optional] END_DATE is the end date of the analysis in the format YYYY-MM-DD
- [optional] REDUCE_OUTLIERS_TO_ZERO is True to reduce outliers to zero. Defaults to False
- [optional] MIN_TOPIC_SIZE is the minimum size of a topic. Defaults to 100.

Example usage:
python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/title_topic_analysis.py --source "mse" --end_date "2024-05-22" --min_topic_size 300

"""

# Package imports
import pandas as pd
import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Local imports
from asf_public_discourse_home_decarbonisation.utils.text_processing_utils import (
    process_abbreviations,
)
from asf_public_discourse_home_decarbonisation import S3_BUCKET
from asf_public_discourse_home_decarbonisation.getters.public_discourse_getters import (
    read_public_discourse_data,
)
from asf_public_discourse_home_decarbonisation.utils.topic_analysis_utils import (
    get_outputs_from_topic_model,
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
    start_date = args.start_date
    end_date = args.end_date
    min_topic_size = args.min_topic_size

    forum_data = read_public_discourse_data(source=source)

    # prep date/time data
    if source == "buildhub":
        forum_data.rename(columns={"url": "id", "date": "datetime"}, inplace=True)
    forum_data["datetime"] = pd.to_datetime(forum_data["datetime"])
    forum_data["date"] = forum_data["datetime"].dt.date
    forum_data["year"] = forum_data["datetime"].dt.year

    # filter by date
    if start_date is not None:
        forum_data = forum_data[
            forum_data["date"] >= datetime.strptime(start_date, "%Y-%m-%d").date()
        ]
    if end_date is not None:
        forum_data = forum_data[
            forum_data["date"] <= datetime.strptime(end_date, "%Y-%m-%d").date()
        ]

    forum_data["title"] = forum_data["title"].apply(process_abbreviations)

    # only replacing HP with heat pump because titles don't have URLs (example ending in .php)
    # and because we're only analysing home heating conversations. Otherwise this would lead to erronenous results
    forum_data["title"] = forum_data["title"].replace("hp", "heat pump")

    # only keep original posts
    forum_data = forum_data[forum_data["is_original_post"] == 1]

    docs = list(forum_data.drop_duplicates("title")["title"])

    dates = list(forum_data.drop_duplicates("title")["date"])

    # renaming titles to sentences - just for ease, due to pre-existing utils
    forum_data.rename(columns={"title": "sentences"}, inplace=True)

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

    # Updating topics and docs information with duplicates (as only unique titles are used for topic analysis)
    topics_info = update_topics_with_duplicates(topics_info, doc_info, forum_data)
    doc_info = update_docs_with_duplicates(doc_info, forum_data)

    path_to_save_prefix = f"s3://{S3_BUCKET}/data/{source}/outputs/topic_analysis/titles_analysis_{source}_{start_date}_{end_date}"
    # Saving outputs to S3
    topics_info.to_csv(
        f"{path_to_save_prefix}_titles_topics_info.csv",
        index=False,
    )
    doc_info.to_csv(
        f"{path_to_save_prefix}_titles_docs_info.csv",
        index=False,
    )
    forum_data.to_csv(
        f"{path_to_save_prefix}_forum_title_data.csv",
        index=False,
    )
