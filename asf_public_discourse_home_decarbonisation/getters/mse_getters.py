"""
Getter functions for Money Saving Expert (MSE) data.
"""

import pandas as pd
from asf_public_discourse_home_decarbonisation.getters.getter_utils import load_s3_data
import sys
import logging
from asf_public_discourse_home_decarbonisation import S3_BUCKET

logger = logging.getLogger(__name__)

mse_categories = [
    "green-ethical-moneysaving",
    "lpg-heating-oil-solid-other-fuels",
    "energy",
    "heat-pumps",
    "is-this-quote-fair",
    "heat-pumps",
]


def get_first_attempt_mse_data() -> pd.DataFrame:
    """
    Gets the first Money Saving Expert sample from S3.
    This was the first sample of data collected, used to generate the first insights.

    Returns:
        pd.DataFrame: a dataframe with 10 pages of MSE data from the Green Ethical Money Saving category.
    """
    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path="data/mse/first_attempt/green-ethical-moneysaving.parquet",
    )


def get_mse_category_data(
    category: str, collection_date: str, processing_level: str = "raw"
) -> pd.DataFrame:
    """
    Gets data from a specific Money Saving Expert category, collected on a certain date.

    Args:
        category (str): An MSE category
        collection_date (str): A date in the format "YYYY_MM_DD"
        processing_level (str): processing level (takes the values "raw" or "processed")
    Returns:
        pd.DataFrame: a dataframe with most up to date category/sub-forum data
    """
    try:
        # Check if processing_level is in the list if one of the accepted levels
        ["raw", "processed"].index(processing_level)
    except ValueError:
        logger.error(f"`{processing_level}` is not a valid processing level!")
        sys.exit(-1)

    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path=f"data/mse/outputs/{processing_level}/mse_data_category_{category}_{collection_date}.parquet",
    )


def get_all_mse_data(
    collection_date: str, processing_level: str = "raw"
) -> pd.DataFrame:
    """
    Gets data from all Money Saving Expert categories.

    Args:
        collection_date (str): A date in the format "YYYY_MM_DD"
        processing_level (str): processing level (takes the values "raw" or "processed")
    Returns:
        pd.DataFrame: a dataframe with data from all categories
    """
    try:
        # Check if the category is in the list if one of the accepted categories
        ["raw", "processed"].index(processing_level)
    except ValueError:
        logger.error(f"`{processing_level}` is not valid!")
        sys.exit(-1)

    all_mse_data = pd.DataFrame()
    for cat in mse_categories:
        aux = load_s3_data(
            bucket_name=S3_BUCKET,
            file_path=f"data/mse/outputs/{processing_level}/mse_data_category_{cat}_{collection_date}.parquet",
        )
        all_mse_data = pd.concat([all_mse_data, aux])

    return all_mse_data.reset_index(drop=True)


def get_mse_data(
    category: str, collection_date: str = "2023_11_15", processing_level: str = "raw"
) -> pd.DataFrame:
    """
    Gets a specific version of the MSE data:
        - either a specific category, a sample of the data or data from all categories;
        - on a specific collection date.
    Existing categories are: "green-ethical-moneysaving", "lpg-heating-oil-solid-other-fuels", "energy", "is-this-quote-fair"
    Additionally, you can also set category to "sample" (to get the initial sample collected) or "all" (to get all the MSE data collected).

    Args:
        category (str): An MSE category, "sample" or "all"
        collection_date (str): A date in the format "YYYY_MM_DD"
        processing_level (str): processing level (takes the values "raw" or "processed")
    Returns:
        pd.DataDrame: a dataframe with the MSE data
    """
    accepted_categories = mse_categories + ["sample", "all"]
    try:
        # Check if the category is in the list if one of the accepted categories
        accepted_categories.index(category)
        if category in mse_categories:
            mse_data = get_mse_category_data(
                category, collection_date, processing_level
            )
        elif category == "sample":
            category = "green-ethical-moneysaving"
            mse_data = get_first_attempt_mse_data()
        else:  # category == "all"
            mse_data = get_all_mse_data(collection_date, processing_level)
        logger.info(f"Data for category '{category}' imported successfully from S3.")
    except ValueError:
        logger.error(f"{category} is not a valid category!")
        sys.exit(-1)

    return mse_data
