"""
Getter functions for BH data.
"""

import pandas as pd
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    load_s3_data,
)
import logging
from asf_public_discourse_home_decarbonisation import S3_BUCKET
import sys

logger = logging.getLogger(__name__)

bh_categories = [
    "119-air-source-heat-pumps-ashp",
    "120-ground-source-heat-pumps-gshp",
    "125-general-alternative-energy-issues",
    "136-underfloor-heating",
    "137-central-heating-radiators",
    "139-boilers-hot-water-tanks",
    "140-other-heating-systems",
]
bh_categories = [cat.replace("-", "_") for cat in bh_categories]


def get_bh_category_data(category: str, collection_date: str) -> pd.DataFrame:
    """
    Returns a dataframe with most up to date category/sub-forum data
    """
    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path=f"data/buildhub/outputs/buildhub_{category}_{collection_date}.csv",
    )


def get_all_bh_data(collection_date: str) -> pd.DataFrame:
    """
    Gets data from all BuildHub categories.

    Args:
        collection_date (str): A date in the format "YYYY_MM_DD"
    Returns:
        pd.DataFrame: a dataframe with data from all categories
    """
    all_bh_data = pd.DataFrame()
    for cat in bh_categories:
        aux = load_s3_data(
            bucket_name=S3_BUCKET,
            file_path=f"data/buildhub/outputs/buildhub_{cat}_{collection_date}.csv",
        )
        aux["category"] = cat
        all_bh_data = pd.concat([all_bh_data, aux])

    return all_bh_data.reset_index(drop=True)


def get_bh_data(category: str, collection_date: str = "24_02_01") -> pd.DataFrame:
    """
    Returns a dataframe with most up to date category/sub-forum data
    """
    accepted_categories = bh_categories + ["all"]
    try:
        # Check if the category is in the list if one of the accepted categories
        accepted_categories.index(category)
        if category in bh_categories:
            bh_data = get_bh_category_data(
                category,
                collection_date,
            )
        else:  # category == "all"
            bh_data = get_all_bh_data(collection_date)
        logger.info(f"Data for category '{category}' imported successfully from S3.")
    except ValueError:
        logger.error(f"{category} is not a valid category!")
        sys.exit(-1)

    return bh_data
