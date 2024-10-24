"""
Forum analysis specific text processing utils such as:
- Removing URLs
- Removing username patterns
- Replacing username mentions
- Removing introduction patterns
- Processing abbreviations
- Checking if a sentence ends with punctuation
"""

import re


def remove_urls(text: str) -> str:
    """Remove URLs from text.

    Args:
        text (str): text to process

    Returns:
        str: text with URLs removed
    """
    # Define a regular expression pattern for matching URLs
    url_pattern = re.compile(r"https?://\S+|www\.\S+")

    # Replace URLs with a space
    cleaned_text = url_pattern.sub(" ", text)
    return cleaned_text


def remove_username_pattern(text: str) -> str:
    """
    Remove the pattern "username wrote:" from text.

    Args:
        text (str): text to process

    Returns:
        str: text with the pattern removed
    """
    # Define the pattern to match "username wrote:" and remove it
    pattern = re.compile(r"\w+\s+wrote: »\n")
    cleaned_text = re.sub(pattern, " ", text)

    return cleaned_text


def replace_username_mentions(text: str) -> str:
    """
    Replace username mentions with a space.

    Args:
        text (str): text to process

    Returns:
        str: text with username mentions replaced
    """
    # Define the pattern to match username mentions and replace them
    pattern = re.compile(r"@\w+\s")
    cleaned_text = re.sub(pattern, " ", text)

    return cleaned_text


def remove_introduction_patterns(text: str) -> str:
    """Remove introduction patterns from text.

    Args:
        text (str): text to process

    Returns:
        str: text with introduction patterns removed
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
    cleaned_text = re.sub(pattern, " ", text)

    return cleaned_text


def process_abbreviations(text: str) -> str:
    """
    Replaces abbreviations in text with their full forms.

    Args:
        text (str): text to process

    Returns:
        str: text with abbreviations replaced
    """
    text = (
        text.lower()
        .replace("ashps", " air source heat pumps ")
        .replace("ashp", " air source heat pump ")
        .replace("gshps", " ground source heat pumps ")
        .replace("gshp", " ground source heat pump ")
        .replace("ufh", " under floor heating ")
        .replace("rhi", " renewable heat incentive ")
        .replace("mcs", " microgeneration certification scheme ")
        .replace("dhw", " domestic hot water ")
        .replace("a2a", " air to air ")
        .replace(" ir ", " infrared ")
        .replace("uvcs", " unvented cylinders ")
        .replace("uvc", " unvented cylinder ")
    )

    # Getting rid of double spaces
    text = " ".join(text.split())
    return text


def ends_with_punctuation(sentence: str) -> bool:
    """
    Check if a sentence ends with punctuation.

    Args:
        sentence (str): sentence to check

    Returns:
        bool: True if the sentence ends with punctuation, False otherwise
    """
    # Regular expression to match a sentence ending with punctuation
    pattern = r"[.!?]$"
    return bool(re.search(pattern, sentence))
