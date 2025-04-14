"""Yaml parsing"""

import pathlib
import typing as t

import astropy.units as u
import yaml
from astropy.io.typing import PathLike

T = t.TypeVar("T")


def recursively_process_dictionary(
    dictionary: dict[str, t.Any | str],
    callable: t.Callable[[str], T],
    ignore_keys: t.Optional[t.Sequence[str]] = (),
) -> dict[str, t.Any | T]:
    """Recursively process a dictionary, parsing strings into quantities.

    Args:
        dictionary: Dictionary to process.
        callable: Function to apply to each value.

    """

    search_type = str

    for key, value in dictionary.items():
        if key in ignore_keys:
            continue
        if isinstance(value, search_type):
            dictionary[key] = callable(value)
        elif isinstance(value, dict):
            dictionary[key] = recursively_process_dictionary(value, callable, ignore_keys=ignore_keys)
        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            value = list(value)
            for index, item in enumerate(value):
                if isinstance(item, search_type):
                    value[index] = callable(item)
                elif isinstance(item, dict):
                    value[index] = recursively_process_dictionary(item, callable, ignore_keys=ignore_keys)
            dictionary[key] = value
    return dictionary


def process_quantity(value: str) -> u.Quantity | str | u.Unit:
    """Process a quantity.

    This function attempts to convert a string to an astropy Quantity or unit.

    Args:
        value: The value to process.

    Returns:
        The processed value.
    """

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return u.Quantity(value)
    except TypeError:
        try:
            return u.Unit(value)
        except ValueError:
            return value
    except ValueError:
        return value


def load_yaml(yaml_data: str) -> dict:
    """Load yaml string.

    Args:
        yaml_data: The Yaml data to load.

    Returns:
        The loaded Yaml data as a dictionary.


    """
    loader = yaml.SafeLoader
    yaml_dict = yaml.load(yaml_data, Loader=loader)
    return recursively_process_dictionary(yaml_dict, process_quantity, ignore_keys=["elements"])


def load_yaml_from_file(file_path: PathLike) -> dict:
    """Load yaml file.

    Args:
        file_path: The path to the Yaml file.

    Returns:
        yaml dictionary

    """
    file_path = pathlib.Path(file_path)
    with open(file_path) as f:
        return load_yaml(f.read())
