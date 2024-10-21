"""
Getters to return information about topics and sentences in each topic.
"""

import pandas as pd
from asf_public_discourse_home_decarbonisation import S3_BUCKET


def get_sentence_data(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    """
    Loads sentences data from S3.

    Args:
        source (str): data source
        filter_by (str): expression to filter by e.g. "heat pump"
        analysis_start_date (str): analysis start date in format YYYY_MM_DD
        analysis_end_date (str):  analysis end date in format YYYY_MM_DD

    Returns:
        pd.DataFrame: sentences data
    """
    path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"

    return pd.read_csv(
        f"{path_to_data}/topic_analysis/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentences_data.csv"
    )


def get_docs_info(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    """
    Loads data from S3 about documents and the topics they belong to.

    Args:
        source (str): data source
        filter_by (str): expression to filter by e.g. "heat pump"
        analysis_start_date (str): analysis start date in format YYYY_MM_DD
        analysis_end_date (str):  analysis end date in format YYYY_MM_DD

    Returns:
        pd.DataFrame: documents data
    """
    path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"

    return pd.read_csv(
        f"{path_to_data}/topic_analysis/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentence_docs_info.csv"
    )


def get_topics_info(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    """
    Loads topics data from S3.

    Args:
        source (str): data source
        filter_by (str): expression to filter by e.g. "heat pump"
        analysis_start_date (str): analysis start date in format YYYY_MM_DD
        analysis_end_date (str):  analysis end date in format YYYY_MM_DD

    Returns:
        pd.DataFrame: topics data
    """
    path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"

    return pd.read_csv(
        f"{path_to_data}/topic_analysis/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentence_topics_info.csv"
    )
