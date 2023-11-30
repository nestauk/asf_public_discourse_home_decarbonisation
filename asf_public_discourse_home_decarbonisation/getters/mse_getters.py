"""
Getter functions for MSE data.
"""
import pandas as pd
from asf_public_discourse_home_decarbonisation.getters.getter_utils import load_s3_data
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
import sys
import logging

logger = logging.getLogger(__name__)

S3_BUCKET = "asf-public-discourse-home-decarbonisation"
PATH_TO_LOCAL_MSE_DATA = PROJECT_DIR / "inputs/data/mse"


def get_first_attempt_mse_data() -> pd.DataFrame:
    """
    Returns a dataframe with 10 pages of MSE data from the Green Ethical Money Saving category.
    This was the first sample of data collected, used to generate the first insights.
    """
    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path="data/mse/first_attempt/green-ethical-moneysaving.parquet",
    )


def get_mse_category_data(
    category: str, collection_date: str, read_from_local_path: bool = False
) -> pd.DataFrame:
    """
    Returns a dataframe with most up to date category/sub-forum data
    """
    if category == "energy":
        return load_s3_data(
            bucket_name=S3_BUCKET,
            file_path=f"data/mse/outputs/mse_data_category_energy.parquet",
        )

    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path=f"data/mse/outputs/mse_data_category_{category}_{collection_date}.parquet",
    )


def get_all_mse_data(collection_date: str) -> pd.DataFrame:
    """
    Returns a dataframe with data from all categories
    """
    categories = [
        "green-ethical-moneysaving",
        "lpg-heating-oil-solid-other-fuels",
        "energy",
        "is-this-quote-fair",
    ]
    all_mse_data = pd.DataFrame()
    for cat in categories:
        if cat != "energy":
            aux = load_s3_data(
                bucket_name=S3_BUCKET,
                file_path=f"data/mse/outputs/mse_data_category_{cat}_{collection_date}.parquet",
            )
        else:
            aux = load_s3_data(
                bucket_name=S3_BUCKET,
                file_path=f"data/mse/outputs/mse_data_category_energy.parquet",
            )
        all_mse_data = pd.concat([all_mse_data, aux])

    return all_mse_data.drop_duplicates().reset_index(drop=True)


def get_mse_data(category: str, collection_datetime: str) -> pd.DataFrame:
    """
    Get MSE data for a specific category and collection date.
    Existing categories are: "green-ethical-moneysaving", "lpg-heating-oil-solid-other-fuels", "energy", "is-this-quote-fair"
    Additionally, you can also use "sample" (to get the initial sample collected) or "all" (to get all the MSE data collected).

    Args:
        category (str): An MSE category, "sample" or "all"
        collection_datetime (str): A date in the format "YYYY_MM_DD"

    Returns:
        pd.DataDrame: a dataframe with the MSE data
    """
    mse_categories = [
        "green-ethical-moneysaving",
        "lpg-heating-oil-solid-other-fuels",
        "energy",
        "is-this-quote-fair",
    ]

    accepted_categories = mse_categories + ["sample", "all"]
    try:
        # Check if the category is in the list if one of the accepted categories
        category_index = accepted_categories.index(category)
        if category in [
            "green-ethical-moneysaving",
            "lpg-heating-oil-solid-other-fuels",
            "is-this-quote-fair",
            "energy",
        ]:
            mse_data = get_mse_category_data(category, collection_datetime)
        elif category == "sample":
            category = "green-ethical-moneysaving"
            mse_data = get_first_attempt_mse_data()
        else:  # category == "all"
            mse_data = get_all_mse_data(collection_datetime)
        logger.info(f"Data for category '{category}' imported successfully from S3.")
    except:
        logger.error(f"{category} is not a valid category!")
        sys.exit(-1)

    return mse_data
