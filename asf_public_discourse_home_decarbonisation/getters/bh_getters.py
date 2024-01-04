"""
Getter functions for BH data.
"""
import pandas as pd
from asf_public_discourse_home_decarbonisation.getters.getter_utils import (
    load_s3_data,
    fetch_file_paths_from_s3_folder,
)

S3_BUCKET = "asf-public-discourse-home-decarbonisation"


def get_bh_category_data(category: str, collection_date: str) -> pd.DataFrame:
    """
    Returns a dataframe with most up to date category/sub-forum data
    """
    return load_s3_data(
        bucket_name=S3_BUCKET,
        file_path=f"data/buildhub/outputs/buildhub_{category}_{collection_date}.csv",
    )
