"""
Getters for the public discourse data.
"""

import pandas as pd
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.getters.bh_getters import get_bh_data


def read_public_discourse_data(
    source: str,
    category: str = "all",
    collection_date: str = None,
    processing_level: str = "raw",
) -> pd.DataFrame:
    """
    Reads forum data or from another source.
    If a path_to_data_file is provided, it reads the data from the file (local or S3 location).
    Otherwise, it reads either from MSE or Buildhub.

    Args:
        source (str): `mse` for Money Saving Expert, `buildhub` for Buildhub or (local or S3) path to a csv file.
        category (str, optional): category/subf-roum filter. Defaults to "all" (i.e. data from all sub-forums collected).
        path_to_data_file (str, optional):
        collection_date (str, optional): Data collection date. If not provided (None), the latest data collection date is used.
        processing_level (str, optional): processing level, either "raw" or "processed". Defaults to "raw". Only available for MSE data.

    Returns:
        pd.DataFrame: the data
    """
    if source == "mse":
        if collection_date is None:
            collection_date = "2023_11_15"
        data = get_mse_data(
            category=category,
            collection_date=collection_date,
            processing_level=processing_level,
        )
    elif source == "buildhub":
        if collection_date is None:
            collection_date = "24_02_01"
        data = get_bh_data(category=category, collection_date=collection_date)
    else:
        data = pd.read_csv(source)
    return data
