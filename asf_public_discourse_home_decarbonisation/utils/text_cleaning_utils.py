import re
import pandas as pd


def remove_urls(text):
    return re.sub(r"http\S+", "", text)


def remove_username_pattern(text):
    # Define the pattern to match "username wrote:" and remove it
    pattern = re.compile(r"\w+\s+wrote: »\n")
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def remove_introduction_patterns(text):
    # Define the patterns to match and remove
    patterns = [
        r"\b(hi|hello|heya|hey)\s+(everyone|all|there)?[!,.]?\s*\n?",
        r"\b(hi|hello|heya|hey)[!,.]?\s*\n?",
    ]

    # Combine patterns into a single regex pattern
    combined_pattern = "|".join(patterns)

    # Compile the combined pattern
    pattern = re.compile(combined_pattern, re.IGNORECASE)

    # Remove patterns from the text
    cleaned_text = re.sub(pattern, "", text)

    return cleaned_text


def process_abbreviations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces abbreviations in the title and text columns of the dataframe with their full forms.

    Args:
        data (pd.DataFrame): dataframe to process

    Returns:
        pd.DataFrame: dataframe with abbreviations replaced
    """
    for col in ["title", "text"]:
        data[col] = (
            data[col]
            .astype(str)
            .apply(
                lambda x: x.lower()
                .replace("ashps", "air source heat pumps")
                .replace("ashp", "air source heat pump")
                .replace("gshps", "ground source heat pumps")
                .replace("gshp", "ground source heat pump")
                .replace("hps", "heat pumps")
                .replace("hp", "heat pump")
                .replace("ufh", "under floor heating")
            )
        )
    return data
