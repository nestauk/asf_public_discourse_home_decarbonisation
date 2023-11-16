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


def get_mse_category_data(category: str, collection_date: str) -> pd.DataFrame:
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
