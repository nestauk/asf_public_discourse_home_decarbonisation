"""
General utils.
"""


def list_chunks(orig_list: list, chunk_size: int = 100):
    """Chunks list into batches of a specified chunk_size."""
    for i in range(0, len(orig_list), chunk_size):
        yield orig_list[i : i + chunk_size]


# Function to flatten the mapping recursively
def flatten_mapping(flat_mapping, mapping, parent_key=None):
    for key, value in mapping.items():
        if isinstance(value, dict):
            flatten_mapping(flat_mapping, value, parent_key=key)
        else:
            if parent_key is not None:
                flat_mapping[value] = parent_key
            else:
                flat_mapping[value] = key

    return flat_mapping


# Function to map the values
def map_values(flat_mapping, value):
    return flat_mapping.get(value, value)


# Function to flatten the mapping recursively
def flatten_mapping_child_key(flat_mapping_child, mapping):
    for key, value in mapping.items():
        if isinstance(value, dict):
            flatten_mapping_child_key(flat_mapping_child, value)
        else:
            flat_mapping_child[value] = key

    return flat_mapping_child
