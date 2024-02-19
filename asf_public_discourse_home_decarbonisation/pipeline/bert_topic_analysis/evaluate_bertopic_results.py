"""
Script to compare the results from different runs of BERTopic models.

The BERTopic model is run multiple times (the value of n_runs), and the results are evaluated using the following metrics:
- Distribution of outlier numers and percentages
- Distribution of number of topics
- Average probability of belonging to a non-outlier topic
- Distribution of outliers (# docs vs. # times a doc is an outlier)

To run this script (with the default parameters):
`python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/evaluate_bertopic_results.py`
This will run the script with the default parameter values:
- n_runs: 10
- path_to_config_file: "asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.bert_params_config"
- path_to_data_file: None

To run this script changing the default parameters:
`python asf_public_discourse_home_decarbonisation/pipeline/bert_topic_analysis/evaluate_bertopic_results.py --n_runs N_RUNS --path_to_config_file CONFIG_FILE_PATH --path_to_data_file PATH_DATA`
where
- N_RUNS is the number of times to run model
- CONFIG_FILE_PATH is location of your configuration file in the repository structure
e.g "asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.bert_params_config"
Note you don't need the file extension .py at the end.
- PATH_DATA if not reading standard forum data (e.g. if reading questions data)
"""

# Package imports
import argparse
from bertopic import BERTopic
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import os

# Local imports
from asf_public_discourse_home_decarbonisation.getters.mse_getters import get_mse_data
from asf_public_discourse_home_decarbonisation.getters.bh_getters import (
    get_bh_data,
)
from asf_public_discourse_home_decarbonisation import PROJECT_DIR
from asf_public_discourse_home_decarbonisation.config.keywords_dictionary import (
    keyword_dictionary,
)
from asf_public_discourse_home_decarbonisation.config.plotting_configs import (
    set_plotting_styles,
    NESTA_COLOURS,
)


# Path to save the figures
TOPIC_ANALYSIS_EVALUATION_PATH = os.path.join(
    PROJECT_DIR, "outputs/figures/topic_analysis/evaluation/"
)
os.makedirs(TOPIC_ANALYSIS_EVALUATION_PATH, exist_ok=True)

# Set plotting styles
set_plotting_styles()


def argparser() -> argparse.Namespace:
    """
    Argparser function to parse arguments from the command line: n_runs, path_to_config_file, path_to_data

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, help="number of runs", default=10)
    parser.add_argument(
        "--path_to_config_file",
        type=str,
        help="path to params config file",
        default="asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.bert_params_config",
    )
    parser.add_argument(
        "--path_to_data_file",
        type=str,
        help="Path to data file, if not standard forum data.",
        default=None,
    )
    args = parser.parse_args()
    return args


def get_configuration_params_file(path_to_config_file: str) -> tuple:
    """
    Loads file with configuration parameters including:
    - data source parameters
    - model and additional parameters
    And returns a tuple of two dictionaries with these configurations.

    Args:
        path_to_config_file (str): Path to configuration file in the repo structure
        e.g. asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.bert_params_config

    Returns:
        tuple: data source parameters and model/additional parameters
    """
    config_module = importlib.import_module(path_to_config_file)

    data_source_parms = config_module.data_source_parms
    model_and_additional_params = config_module.model_and_additional_params

    return data_source_parms, model_and_additional_params


def create_boxplots_with_results(output_configs: dict):
    """
    Creates and saves a figure with boxplots showing the distribution of results for:
    - Number of topics
    - Percentage of outliers
    - Average probability of beloging to a non-outlier topic

    Args:
        output_configs (dict): dictionary with results information
    """

    # Extracting information from output_configs dictionary
    source_name = output_configs["data_source"]
    category = output_configs["category"]
    n_docs = output_configs["n_docs"]
    n_runs = output_configs["n_runs"]
    keywords = output_configs["keywords"]
    number_of_outliers_df = output_configs["outliers_df"]
    number_of_topics_df = output_configs["topics_df"]
    avg_probablity_df = output_configs["probabilities_df"]
    outliers_vs_docs_dict = output_configs["outliers_vs_docs_dict"]

    # Plot horizontal boxplots for each dataframe: number_of_topics_df, number_of_outliers_df and avg_probablity_df
    fig, axes = plt.subplots(3, 1, figsize=(10, number_of_topics_df.shape[1] * 3))

    number_of_topics_df.boxplot(ax=axes[0], grid=False, vert=False)
    axes[0].set_xlabel("Number of topics")

    percentage_of_outliers_df = number_of_outliers_df / n_docs * 100
    percentage_of_outliers_df.boxplot(ax=axes[1], grid=False, vert=False)
    axes[1].set_xlabel("Percentage of outliers")
    # Making the x-axis always between 0 and 50 (just as visual aid)
    if percentage_of_outliers_df.max().max() <= 50:
        axes[1].set_xlim(0, 50)

    avg_probablity_df.boxplot(ax=axes[2], grid=False, vert=False)
    axes[2].set_xlabel("Average probablity of belonging to a non-outlier topic")

    # axes[3].hist([res["outlier_fraction"] for res in outliers_vs_docs_dict.values()],
    #     bins=30,
    #     color=[NESTA_COLOURS[0] for res in outliers_vs_docs_dict.values()],
    #     edgecolor="white",
    # )
    # axes[3].set_xlabel('fraction of runs where a document is an outlier')
    # axes[3].set_ylabel('# of docs')

    fig.suptitle(
        "Source: {}, Category: {}\n Keywords filter: {}, # Docs: {}, # Runs: {}".format(
            source_name, category, keywords, n_docs, n_runs
        )
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            TOPIC_ANALYSIS_EVALUATION_PATH,
            f"{source_name}_{category}_{keywords}_{n_runs}runs.png",
        )
    )

    plt.close()


def run_topic_model_evaluation(n_runs: int, docs: list, model_configs: dict) -> tuple:
    """
    Evaluates topic model on a specific number of runs.

    Args:
        n_runs (int): number of times to run the model
        docs (list): documents to cluster into topics
        model_configs (dict): model configurations

    Returns:
        tuple:
            - list of number of topics
            - list of number of outliers
            - list of average probability of belonging to a non-outlier cluster
            - distribution of outliers
    """
    runs_number_of_topics = []
    runs_number_of_outliers = []
    runs_avg_prob = []

    docs_outlier_count = pd.DataFrame(columns=["Document", "outlier_count"])
    docs_outlier_count["Document"] = docs
    docs_outlier_count["outlier_count"] = 0

    # Getting model configurations
    # default values found here: https://maartengr.github.io/BERTopic/api/bertopic.html#bertopic._bertopic.BERTopic.__init__
    nr_topics = model_configs.get("nr_topics", None)
    representation_model = model_configs.get("representation_model", None)
    min_topic_size = model_configs.get("min_topic_size", 10)
    vectorizer_model = model_configs.get("vectorizer_model", None)
    umap_model = model_configs.get("umap_model", None)
    hdbscan_model = model_configs.get("hdbscan_model", None)
    embedding_model = model_configs.get("embedding_model", None)

    for i in range(n_runs):
        topic_model = BERTopic(
            nr_topics=nr_topics,
            representation_model=representation_model,
            min_topic_size=min_topic_size,
            vectorizer_model=vectorizer_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            embedding_model=embedding_model,
        )

        try:
            topics, probs = topic_model.fit_transform(docs)
            topics = topic_model.get_topics()

            topics_info = topic_model.get_topic_info()

            doc_info = topic_model.get_document_info(docs)

            # Append results from each run
            runs_number_of_topics.append(len(topics))
            runs_number_of_outliers.append(
                topics_info[topics_info["Topic"] == -1]["Count"].iloc[0]
            )
            # Appending average probability of the documents (not in the outliers' cluster) belonging to topics
            runs_avg_prob.append(doc_info[doc_info["Topic"] > -1]["Probability"].mean())

            # Updating number of times a doc is an outlier
            docs_outlier_count = docs_outlier_count.merge(
                right=doc_info[["Document", "Topic"]], on="Document"
            )
            docs_outlier_count["outlier_count"] = docs_outlier_count.apply(
                lambda x: (
                    x["outlier_count"] + 1 if x["Topic"] == -1 else x["outlier_count"]
                ),
                axis=1,
            )
            docs_outlier_count.drop(columns="Topic", inplace=True)
        except:
            print(f"Run {i} failed due to no topics being found. Skipping...")

        outliers_vs_docs = (
            docs_outlier_count.groupby("outlier_count")
            .size()
            .reset_index(name="num_docs")
        )
        outliers_vs_docs = outliers_vs_docs[outliers_vs_docs["outlier_count"] > 0]

    return (
        runs_number_of_topics,
        runs_number_of_outliers,
        runs_avg_prob,
        outliers_vs_docs,
    )


def read_and_filter_data(
    source_name: str, category: str, path_to_data_file: str, keywords: list
) -> pd.DataFrame:
    """
    Loads and filters data before applying topic analysis.

    Args:
        source_name (str): source name e.g. mse or bh
        category (str): category e.g. "all"
        path_to_data_file (str): path to data file if not the standard forum data
        keywords (list): list of keywords to filter the data

    Returns:
        pd.DataFrame: filtered data
    """
    if path_to_data_file is not None:
        data = pd.read_csv(path_to_data_file)
    else:
        if source_name == "mse":
            data = get_mse_data(category)
        else:
            data = get_bh_data(category)

    # filter the data based on the keywords
    if keywords is not None:
        data = data[
            data["title"].str.contains(
                "|".join(keyword_dictionary[keywords]), case=False
            )
            | data["text"].str.contains(
                "|".join(keyword_dictionary[keywords]), case=False
            )
        ]
    return data


if __name__ == "__main__":
    args = argparser()
    n_runs = args.n_runs
    path_to_config_file = args.path_to_config_file
    path_to_data_file = args.path_to_data_file

    (
        data_source_parms,
        model_and_additional_params,
    ) = get_configuration_params_file(path_to_config_file)

    # for each data source comparing
    for data_source in data_source_parms:
        source_name = data_source["data_source"]

        # One figure will be created for each slide of data considered
        # i.e. data from a specific category or containing specific keywords
        # comparing different models
        for slice in data_source["slice"]:
            category = slice.get("category")
            keywords = slice.get("keywords")
            data = read_and_filter_data(
                source_name, category, path_to_data_file, keywords
            )

            number_of_topics_df = pd.DataFrame()
            number_of_outliers_df = pd.DataFrame()
            avg_probablity_df = pd.DataFrame()
            outliers_vs_docs_dict = dict()

            # for each model specification run and evaluate the model
            for model_param in model_and_additional_params:
                model_name = model_param["model_name"]
                text_column = model_param["text_column"]
                filter = model_param["filter"]

                # Preparing data for topic analysis
                if filter == "original_posts":
                    docs = data[data["is_original_post"] == 1]
                else:
                    docs = data.copy()
                docs = docs.drop_duplicates(text_column)[text_column]

                # Running topic model evaluation
                (
                    runs_number_of_topics,
                    runs_number_of_outliers,
                    runs_avg_prob,
                    outliers_vs_docs,
                ) = run_topic_model_evaluation(n_runs, docs, model_param)

                number_of_topics_df[model_name] = runs_number_of_topics
                number_of_outliers_df[model_name] = runs_number_of_outliers
                avg_probablity_df[model_name] = runs_avg_prob
                outliers_vs_docs_dict[model_name] = outliers_vs_docs

            # Create config dictionary with outputs
            output_configs = {
                "data_source": source_name,
                "category": category,
                "n_docs": len(docs),
                "n_runs": n_runs,
                "keywords": keywords,
                "outliers_df": number_of_outliers_df,
                "topics_df": number_of_topics_df,
                "probabilities_df": avg_probablity_df,
                "outliers_vs_docs_dict": outliers_vs_docs_dict,
            }

            # Plotting results
            create_boxplots_with_results(output_configs)
