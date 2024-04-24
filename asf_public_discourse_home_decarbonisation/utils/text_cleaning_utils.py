"""
Utility functions for cleaning text data.
"""

import re
import pandas as pd


def remove_urls(text: str) -> str:
    """
    Removes URLs from text.
    Args:
        text (str): a string, tipically one or multiple sentences long
    Returns:
        str: text without URLs
    """
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r"https?://\S+|www\.\S+")

    # Replace URLs with a space
    cleaned_text = url_pattern.sub(" ", text)
    return cleaned_text


def remove_username_pattern(text) -> str:
    """
    Removes the "username wrote:" pattern from text.

    Args:
        text (str): original text

    Returns:
        str: text without pattern
    """
    # Define the pattern to match "username wrote:" and remove it
    pattern = re.compile(r"\w+\s+wrote: »")
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text


def remove_introduction_patterns(text: str) -> str:
    """
    Removes introduction patterns from text such as "hello there".
    Args:
        text (str): the text data to process
    Returns:
        str: cleaned data
    """
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


def process_abbreviations(text: str) -> str:
    """
    Replaces abbreviations in text data with their full forms.

    Args:
        text (str): the text data to process

    Returns:
        text: processed text data
    """

    text = (
        text.replace("ashps", " air source heat pumps ")
        .replace("ashp", " air source heat pump ")
        .replace("gshps", " ground source heat pumps ")
        .replace("gshp", " ground source heat pump ")
        # .replace("hps ", " heat pumps ")
        # .replace("hp ", " heat pump ")
        .replace("ufh", "under floor heating ")
        .replace("rhi", " renewable heat incentive ")
        .replace("mcs", " microgeneration certification scheme ")
        .replace("dhw", " domestic hot water ")
        .replace("a2a", " air to air ")
        .replace(" ir ", " infrared ")
        .replace("uvcs", " unvented cylinders ")
        .replace("uvc", " unvented cylinder ")
    )

    return text
