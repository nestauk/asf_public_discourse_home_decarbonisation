"""
Script to evaluate the results of the specific BERTopic models.

The model is run multiple times, and the results are evaluated using the following metrics:
- Distribution of outlier numers and percentages
- Distribution of topic sizes
- Percentage/number of documents always in the outliers' cluster
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
from asf_public_discourse_home_decarbonisation.analysis.mse.initial_text_analysis_category_data import (
    keyword_dictionary,
)

# Path to save the figures
TOPIC_ANALYSIS_EVALUATION_PATH = os.path.join(
    PROJECT_DIR, "outputs/figures/topic_analysis/evaluation/"
)
os.makedirs(TOPIC_ANALYSIS_EVALUATION_PATH, exist_ok=True)


def get_configuration_params(path_to_config_file: str) -> tuple:
    """_summary_

    Args:
        path_to_config_file (str): _description_

    Returns:
        tuple: _description_
    """
    config_module = importlib.import_module(path_to_config_file)

    data_source_parms = config_module.data_source_parms
    model_and_additional_params = config_module.model_and_additional_params

    return data_source_parms, model_and_additional_params


def create_boxplots(output_configs: dict):
    """_summary_

    Args:
        output_configs (dict): _description_
    """

    source_name = output_configs["data_source"]
    category = output_configs["category"]
    n_docs = output_configs["n_docs"]
    n_runs = output_configs["n_runs"]
    keywords = output_configs["keywords"]
    number_of_outliers_df = output_configs["outliers_df"]
    number_of_topics_df = output_configs["topics_df"]
    avg_probablity_df = output_configs["probabilities_df"]

    fig, axes = plt.subplots(3, 1, figsize=(10, number_of_topics_df.shape[1] * 3))

    # Plot horizontal boxplots for each dataframe
    number_of_topics_df.boxplot(ax=axes[0], grid=False, vert=False)
    axes[0].set_xlabel("Number of topics")

    percentage_of_outliers_df = number_of_outliers_df / n_docs * 100
    percentage_of_outliers_df.boxplot(ax=axes[1], grid=False, vert=False)
    axes[1].set_xlabel("Percentage of outliers")
    if percentage_of_outliers_df.max().max() <= 50:
        axes[1].set_xlim(0, 50)

    avg_probablity_df.boxplot(ax=axes[2], grid=False, vert=False)
    axes[2].set_xlabel("Average probablity of belonging to a non-outlier topic")

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

    plt.tight_layout()

    plt.close()


def run_topic_model(n_runs, docs, model_configs):
    runs_number_of_topics = []
    runs_number_of_outliers = []
    runs_avg_prob = []

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

            # Append values from run
            runs_number_of_topics.append(len(topics))
            runs_number_of_outliers.append(
                topics_info[topics_info["Topic"] == -1]["Count"].iloc[0]
            )
            # Average probability of the documents (not in the outliers' cluster) belonging to topics
            runs_avg_prob.append(doc_info[doc_info["Topic"] > -1]["Probability"].mean())
        except:
            print(f"Run {i} failed due to no topics being found. Skipping...")

    return runs_number_of_topics, runs_number_of_outliers, runs_avg_prob


def argparser():
    """
    Argparser function to parse arguments from the command line: n_runs, path_to_config_file, path_to_data

    Returns:
        _type_: _description_
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_runs", type=int, help="number of runs", default=10)
    parser.add_argument(
        "--path_to_config_file",
        type=str,
        help="path to params config file",
        default="asf_public_discourse_home_decarbonisation.pipeline.bert_topic_analysis.bert_params_config",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    n_runs = args.n_runs
    path_to_config_file = args.path_to_config_file

    (
        data_source_parms,
        model_and_additional_params,
    ) = get_configuration_params(path_to_config_file)

    # for each data source
    for data_source in data_source_parms:
        source_name = data_source["data_source"]

        # for each slice of the data considered i.e. data from a specific category or containing specific keywords
        for part in data_source["part"]:
            category = part["category"]

            if source_name == "mse":
                data = get_mse_data(category)
            else:
                data = get_bh_data(category)

            # filter the data based on the keywords
            if "keywords" in part.keys():
                keywords = part["keywords"]
                data = data[
                    data["title"].str.contains(
                        "|".join(keyword_dictionary[keywords]), case=False
                    )
                    | data["text"].str.contains(
                        "|".join(keyword_dictionary[keywords]), case=False
                    )
                ]
            else:
                keywords = "None"

            number_of_topics_df = pd.DataFrame()
            number_of_outliers_df = pd.DataFrame()
            avg_probablity_df = pd.DataFrame()

            # for each model specification run and evaluate the model
            for model_param in model_and_additional_params:
                model_name = model_param["model_name"]
                text_column = model_param["text_column"]
                filter = model_param["filter"]

                if filter == "posts":
                    docs = data[data["is_original_post"] == 1]
                else:
                    docs = data.copy()

                if text_column == "title":
                    docs = docs.drop_duplicates("title")["title"]
                else:
                    docs = docs.drop_duplicates("text")["text"]

                runs_number_of_topics, runs_number_of_outliers, runs_avg_prob = (
                    run_topic_model(n_runs, docs, model_param)
                )

                number_of_topics_df[model_name] = runs_number_of_topics
                number_of_outliers_df[model_name] = runs_number_of_outliers
                avg_probablity_df[model_name] = runs_avg_prob

            output_configs = {
                "data_source": source_name,
                "category": category,
                "n_docs": len(docs),
                "n_runs": n_runs,
                "keywords": keywords,
                "outliers_df": number_of_outliers_df,
                "topics_df": number_of_topics_df,
                "probabilities_df": avg_probablity_df,
            }
            create_boxplots(output_configs)
