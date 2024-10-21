import pandas as pd
from asf_public_discourse_home_decarbonisation import S3_BUCKET


def get_sentence_sentiment(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    """
    Get sentence sentiment data from S3.

    Args:
        source (str): data source, "mse" or "buildhub"
        filter_by (str): expression to filter by, e.g. "heat pump"
        analysis_start_date (str): analysis start date in format "YYYY-MM-DD"
        analysis_end_date (str): analysis end date in format "YYYY-MM-DD"

    Returns:
        pd.DataFrame: sentence sentiment data
    """
    path_to_data = path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"
    return pd.read_csv(
        f"{path_to_data}/sentiment/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentence_topics_sentiment.csv"
    )
