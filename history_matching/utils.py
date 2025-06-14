"""Implementation of various helper functions."""

from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd

PARAMETER_SPACE_COLUMNS = ["parameter", "minimum", "maximum"]
OBSERVATIONS_COLUMNS = ["feature", "mean", "std"]


def mean_and_std_for_observations(observations: Dict[str, Union[List, np.ndarray]]) -> pd.DataFrame:
    """
    Return a Pandas DataFrame with expected columns for a set of raw observations.

    Args:
        observations: a dictionary mapping one or more features to one or more recorded values for that feature

    Returns:
        Pandas DataFrame with columns "feature" (feature name: string), "mean" (mean of recorded values), and "std" (standard deviation of recorded values)
    """

    data = [(key, np.mean(values), np.std(values, ddof=1)) for key, values in observations.items()]

    statistics = pd.DataFrame(data=data, columns=OBSERVATIONS_COLUMNS).set_index("feature", drop=False)

    return statistics


def features_from_observations(observations: pd.DataFrame) -> List[str]:
    """
    Return a list of features from a Pandas DataFrame of observations.

    Args:
        observations: Pandas DataFrame of observations

    Returns:
        List of features
    """

    features = list(observations.feature)

    return features
