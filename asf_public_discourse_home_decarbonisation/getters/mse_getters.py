"""
Getter functions for MSE data.
"""
import pandas as pd
from nesta_ds_utils.loading_saving import S3
from asf_public_discourse_home_decarbonisation import base_config

S3_BUCKET = base_config["S3_BUCKET"]


def get_first_attempt_mse_data() -> pd.DataFrame:
    """
    Returns a dataframe with 10 pages of MSE data
    from the Green Ethical Money Saving category.
    """
    return S3.download_obj(
        bucket=S3_BUCKET,
        path_from="data/mse/first_attempt/green-ethical-moneysaving.parquet",
        download_as="dataframe",
        kwargs_reading={"engine": "python"},
    )
