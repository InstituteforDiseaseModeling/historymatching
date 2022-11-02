from io import BytesIO
from typing import List

import numpy as np
import pandas as pd


def mean_and_variance_for_observations(observations: pd.DataFrame) -> pd.DataFrame:

    statistics = pd.DataFrame(data=["mean", "variance"], columns=["statistic"])    # one row for mean, one row for variance

    for feature in observations.columns:
        statistics[feature] = [observations[feature].mean(), observations[feature].var()]

    return statistics


def features_from_observations(observations: pd.DataFrame) -> List[str]:

    features = list([column for column in observations.columns if column != "statistic"])

    return features


def dataframe_to_ndarray(df: pd.DataFrame) -> np.ndarray:

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