"""
A script to produce stats from public discourse data:
- total number of posts and replies
- total number of posts and replies by category or sub-forum

To run:

python asf_public_discourse_home_decarbonisation/pipeline/stats/data_source_stats.py --source DATA_SOURCE --end_date END_DATE

where:
[required] DATA_SOURCE: `mse` or `buildhub`
[optional] END_DATE: first date to not include in the analysis, in format YYYY-MM-DD (i.e. if the data collection happened on 2024-05-23,
the last complete date would be 2024-05-22, so we can set this to "2024-05-23" to get complete data up to 2024-05-22)
"""

# Package imports
import argparse

import logging

logger = logging.getLogger(__name__)

from asf_public_discourse_home_decarbonisation.getters.public_discourse_getters import (
    read_public_discourse_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        help="`mse` or `buildhub`",
        required=True,
    )
    parser.add_argument(
        "--end_date",
        help="First date to not include in the analysis, in format YYYY-MM-DD",
        default=None,
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = args.source
    end_date = args.end_date

    logger.info(f"Getting stats for source: {source}")

    data = read_public_discourse_data(source)

    if end_date is not None:
        if source == "buildhub":
            data.rename(columns={"date": "datetime"}, inplace=True)
        data = data[data["datetime"] < end_date]

    data["counts"] = data["is_original_post"].apply(
        lambda x: "Number of posts" if x == 1 else "Number of replies"
    )

    totals_by_post_type = data.value_counts("counts")
    logger.info(f"Total number of posts and replies:\n{totals_by_post_type}")

    totals_by_category = data.value_counts(["category", "counts"])
    logger.info(
        f"Total number of posts and replies by category or sub-forum:\n{totals_by_category}"
    )

    logger.info("Finished getting stats!")
