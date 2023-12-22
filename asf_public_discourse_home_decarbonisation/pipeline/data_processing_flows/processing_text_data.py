"""
A Flow for processing title and text data from MSE sub-forums.
python asf_public_discourse_home_decarbonisation/pipeline/data_processing_flows/processing_text_data.py --datastore=s3 --package-suffixes=.txt run --max-num-splits 2000 --max-workers 100
"""

import os

# Upgrading pip and installing requirements
os.system("python -m pip install --upgrade pip")
os.system(
    f"pip install -r {os.path.dirname(os.path.realpath(__file__))}/flow_requirements.txt 1> /dev/null"
)
os.system("python -m spacy download en_core_web_sm")

from metaflow import FlowSpec, step, batch, Parameter
import pandas as pd
from flow_utils import remove_tokens_in_list

S3_BUCKET = "asf-public-discourse-home-decarbonisation"
MSE_S3_RAW_OUTPUTS_FOLDER_PATH = "data/mse/outputs"
MSE_S3_PROCESSED_OUTPUTS_FOLDER_PATH = "data/mse/outputs/processed"
CHUNKSIZE = 10000


def remove_tokens_from_specified_columns(
    data: pd.DataFrame, columns: list, list_of_tokens_to_remove: set
) -> pd.DataFrame:
    """
    Removes items in a list provided from certain columns in data.

    Args:
        data (pd.DataFrame): data
        columns (list): column names
        list_of_tokens_to_remove (list): list of items to be removed
    Returns:
        pd.DataFrame: MSE data with items removed from text and titles
    """
    for col in columns:
        data[col + "_no_stopwords"] = data.apply(
            lambda x: remove_tokens_in_list(x[col], list_of_tokens_to_remove), axis=1
        )

    return data


class TextProcessingFlow(FlowSpec):
    category = Parameter(
        name="category",
        help="A category or sub-forum",
        default="green-ethical-moneysaving",
    )

    batch_date = Parameter(
        name="batch",
        help='Batch date (e.g. date of collection "2023_11_15" or "newest" for the most up to date version of the data)',
        default="2023_11_15",
    )

    test = Parameter(
        name="test",
        help="Set to True to run flow on test mode.",
        default=False,
    )

    @step
    def start(self):
        """
        Starts the flow and reads raw data from S3.
        """
        from smart_open import open
        import pandas as pd

        s3_path = f"s3://{S3_BUCKET}/{MSE_S3_RAW_OUTPUTS_FOLDER_PATH}/mse_data_category_{self.category}_{self.batch_date}.parquet"
        with open(s3_path, "rb") as s3_file:
            self.mse_data = pd.read_parquet(s3_file)

        if self.test:  # working with less data if just testing
            self.mse_data = self.mse_data[:100]

        self.next(self.process_text_and_titles)

    @step
    def process_text_and_titles(self):
        """
        Pre-processing text and titles (before applying lemmatisation) by
        removing URLs, putting text to lowercase and removing username patterns.
        """
        from flow_utils import preprocess_text

        self.mse_data["processed_text"] = self.mse_data["text"].apply(
            lambda x: preprocess_text(x)
        )
        self.mse_data["processed_title"] = self.mse_data["title"].apply(
            lambda x: preprocess_text(x)
        )

        self.next(self.prepare_for_lemmatising)

    @batch
    @step
    def prepare_for_lemmatising(self):
        """
        Chunking data to allow for parallel lemmatisation.
        """
        self.chunks = [
            self.mse_data[i : i + CHUNKSIZE]
            for i in range(0, len(self.mse_data) + 1, CHUNKSIZE)
        ]
        self.next(self.lemmatising_and_tokenise_text_data, foreach="chunks")

    @batch(cpu=2, memory=8000)
    @step
    def lemmatising_and_tokenise_text_data(self):
        """
        Creates columns with lemmatising and tokenised titles and text from posts and replies.
        Because titles are the same for all replies to a post, we only lemmatise/tokenise unique titles.
        """
        from flow_utils import lemmatise, tokenise

        data = self.input

        # Start by lemmatising unique titles
        titles = data[["processed_title", "id"]]
        titles.drop_duplicates("id", inplace=True)

        # Mapping original titles to the lemmatised titles in the original data
        titles["lemmatised_tokens_title"] = titles["processed_title"].apply(lemmatise)
        titles["tokens_title"] = titles["processed_title"].apply(tokenise)

        # Setting the same index in both dataframes
        titles.set_index("id", inplace=True)
        data.set_index("id", inplace=True)

        # Assigning the lemmatised title according to ID
        data["lemmatised_tokens_title"] = titles["lemmatised_tokens_title"]
        data["tokens_title"] = titles["tokens_title"]

        # Reseting the dataset
        data.reset_index(inplace=True)

        # Lemmatising text from posts and replies
        data["lemmatised_tokens_text"] = data["processed_text"].apply(lemmatise)
        data["tokens_text"] = data["processed_text"].apply(tokenise)

        # Passing variable data to the next step using self
        self.data = data

        self.next(self.join_data_from_previous_step)

    @batch(cpu=2, memory=8000)
    @step
    def join_data_from_previous_step(self, inputs):
        """
        Joining data from all batches after lemmatisation.
        """
        import pandas as pd

        self.mse_data = pd.DataFrame()
        for input in inputs:
            self.mse_data = pd.concat([self.mse_data, input.data])

        self.next(self.remove_stopwords)

    @step
    def remove_stopwords(self):
        """
        Removing stopwords and punctuation.
        """
        from flow_utils import english_stopwords_definition

        stopwords = english_stopwords_definition()
        self.mse_data = remove_tokens_from_specified_columns(
            self.mse_data,
            [
                "tokens_text",
                "tokens_title",
                "lemmatised_tokens_title",
                "lemmatised_tokens_text",
            ],
            stopwords,
        )

        self.next(self.save_data)

    @step
    def save_data(self):
        """
        Saving data to S3.
        """
        if not self.test:  # only saves data if not in test mode
            self.mse_data.to_parquet(
                f"s3://{S3_BUCKET}/{MSE_S3_PROCESSED_OUTPUTS_FOLDER_PATH}/mse_data_category_{self.category}_{self.batch_date}.parquet",
                index=False,
            )

        self.next(self.end)

    @step
    def end(self):
        """
        Ends the flow.
        """
        pass


if __name__ == "__main__":
    TextProcessingFlow()
