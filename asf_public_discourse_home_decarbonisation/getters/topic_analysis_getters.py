import pandas as pd
from asf_public_discourse_home_decarbonisation import S3_BUCKET


def get_sentence_data(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"

    return pd.read_csv(
        f"{path_to_data}/topic_analysis/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentences_data.csv"
    )


def get_docs_info(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"

    return pd.read_csv(
        f"{path_to_data}/topic_analysis/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentence_docs_info.csv"
    )


def get_topics_info(
    source: str, filter_by: str, analysis_start_date: str, analysis_end_date: str
) -> pd.DataFrame:
    path_to_data = f"s3://{S3_BUCKET}/data/{source}/outputs"

    return pd.read_csv(
        f"{path_to_data}/topic_analysis/{source}_{filter_by}_{analysis_start_date}_{analysis_end_date}_sentence_topics_info.csv"
    )
