"""
Getter functions for MSE data.
"""
import pandas as pd
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    load_s3_data,
    fetch_file_paths_from_s3_folder,
)

S3_BUCKET = "asf-public-discourse-home-decarbonisation"


def get_first_attempt_mse_data() -> pd.DataFrame:
    """
    Returns a dataframe with 10 pages of MSE data from the Green Ethical Money Saving category.
    """
    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path="data/mse/first_attempt/green-ethical-moneysaving.parquet",
    )


def get_mse_category_data(category: str) -> pd.DataFrame:
    """
    Returns a dataframe with most up to date category/sub-forum data
    """
    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path=f"data/mse/outputs/mse_data_category_{category}.parquet",
    )


def get_all_mse_data() -> pd.DataFrame:
    """
    Returns a dataframe with data from all categories
    """
    file_paths = fetch_file_paths_from_s3_folder("data/mse/outputs/")

    all_mse_data = pd.DataFrame()
    for fp in file_paths:
        aux = load_s3_data(bucket_name=S3_BUCKET, file_path=fp)
        all_mse_data = pd.concat([all_mse_data, aux])

    return all_mse_data.drop_duplicates().reset_index(drop=True)
