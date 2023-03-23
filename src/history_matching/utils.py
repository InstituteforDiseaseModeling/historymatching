from io import BytesIO
from typing import Dict, List, Union

import numpy as np
import pandas as pd


def mean_and_variance_for_observations(observations: Dict[str, Union[List, np.ndarray]]) -> pd.DataFrame:

    """
    Return a Pandas DataFrame with expected columns for a set of raw observations.

    Args:
        observations: a dictionary mapping one or more features to one or more recorded values for that feature

    Returns:
        Pandas DataFrame with columns "features" (feature name: string), "means" (mean of recorded values), and "variances" (variance of recorded values)
    """

    data = [(key, np.mean(values), np.var(values, ddof=1)) for key, values in observations.items()]

    statistics = pd.DataFrame(data=data, columns=["features", "means", "variances"]).set_index("features", drop=False)

    return statistics


def features_from_observations(observations: pd.DataFrame) -> List[str]:

    """
    Return a list of features from a Pandas DataFrame of observations.

    Args:
        observations: Pandas DataFrame of observations

    Returns:
        List of features
    """

    features = list(observations.features)

    return features


def dataframe_to_ndarray(df: pd.DataFrame) -> np.ndarray:

    """
    Convert a Pandas DataFrame to a NumPy ndarray.

    Args:
        df: Pandas DataFrame

    Returns:
        NumPy ndarray
    """

    if df is not None:
        buffer = BytesIO()
        df.reset_index(drop=True).to_feather(buffer)
        result = np.array(buffer.getbuffer(), dtype=np.uint8)
    else:
        result = None

    return result


def ndarray_to_dataframe(nd: np.ndarray) -> pd.DataFrame:

    if nd is not None:
        result = pd.read_feather(BytesIO(nd.data))
    else:
        result = None

    return result
