"""
General utils.
"""


def list_chunks(orig_list: list, chunk_size: int = 100):
    """Chunks list into batches of a specified chunk_size."""
    for i in range(0, len(orig_list), chunk_size):
        yield orig_list[i : i + chunk_size]


def flatten_mapping(flat_mapping: dict, mapping: dict, parent_key: str = None) -> dict:
    """
    Flattens a mapping recursively returning the highest level of aggregation

    Args:
        flat_mapping (dict): a dictionary with a 1-1 mapping (or an empty dictionary)
        mapping (dict): dictionary mapping with multiple levels
        parent_key (str, optional): parent key. Defaults to None.

    Returns:
        dict: 1-1 mapping using highest level of aggregation
    """
    for key, value in mapping.items():
        if isinstance(value, dict):
            flatten_mapping(flat_mapping, value, parent_key=key)
        else:
            if parent_key is not None:
                flat_mapping[value] = parent_key
            else:
                flat_mapping[value] = key

    return flat_mapping


def map_values(flat_mapping: dict, value: str) -> str:
    """
    Maps valuesn using a dictionary

    Args:
        flat_mapping (dict): dictionary with 1-1 mapping
        value (str): a specific value

    Returns:
        str: the mapped value
    """
    return flat_mapping.get(value, value)


def flatten_mapping_child_key(flat_mapping_child: dict, mapping: dict) -> dict:
    """
    Flattens a mapping recursively

    Args:
        flat_mapping (dict): a dictionary with a 1-1 mapping (or an empty dictionary)
        mapping (dict): dictionary mapping with multiple levels

    Returns:
        dict: 1-1 mapping using lowwest level of aggregation
    """
    for key, value in mapping.items():
        if isinstance(value, dict):
            flatten_mapping_child_key(flat_mapping_child, value)
        else:
            flat_mapping_child[value] = key

    return flat_mapping_child
