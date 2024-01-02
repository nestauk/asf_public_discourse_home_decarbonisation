from fnmatch import fnmatch
import json
import pickle
import pandas as pd
import boto3
from typing import List, Union, Dict

from asf_public_discourse_home_decarbonisation import logger, S3_BUCKET


def get_s3_resource():
    s3 = boto3.resource("s3")
    return s3


def load_s3_data(bucket_name: str, file_path: str) -> Union[pd.DataFrame, Dict]:
    """
    Load data from S3 location.

    Args:
        bucket_name (str) : The S3 bucket name
        file_path (str): S3 key to load
    Returns:
        Union[pd.DataFrame, Dict]: the loaded data
    """
    s3 = get_s3_resource()

    obj = s3.Object(bucket_name, file_path)
    if fnmatch(file_path, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_path, "*.csv"):
        return pd.read_csv("s3://" + bucket_name + "/" + file_path)
    elif fnmatch(file_path, "*.parquet"):
        return pd.read_parquet("s3://" + bucket_name + "/" + file_path)
    elif fnmatch(file_path, "*.pkl") or fnmatch(file_path, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        logger.error(
            'Function not supported for file type other than "*.csv", "*.json", "*.pkl" or "*.parquet"'
        )


def fetch_file_paths_from_s3_folder(path_folder: str) -> List[str]:
    """
    Fetches file paths from a specified S3 folder.
    Args:
        path_folder (str): The path to the folder in S3.

    Returns:
        List[str]: list of strings with the file paths.
    """
    # Create an S3 client
    s3_client = boto3.client("s3")

    objects_in_folder = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=path_folder,
    )

    # Extract the filepaths from the response
    all_object_keys = [obj["Key"] for obj in objects_in_folder.get("Contents", [])]

    next_marker = objects_in_folder.get("NextContinuationToken")

    while next_marker:
        # List objects in the specified folder
        objects_in_folder = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=path_folder,
            ContinuationToken=next_marker,
        )

        # Extract the filepaths from the response
        object_keys = [obj["Key"] for obj in objects_in_folder.get("Contents", [])]

        all_object_keys.extend(object_keys)

        # Check if there are more objects to retrieve
        next_marker = objects_in_folder.get("NextContinuationToken")

    # Removing the paths to folders, leaving only the paths to files ending with ".parquet"
    file_paths = [fp for fp in all_object_keys if fp.endswith(".parquet")]

    return file_paths
