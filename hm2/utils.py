from typing import List

import pandas as pd


def mean_and_variance_for_observations(observations: pd.DataFrame) -> pd.DataFrame:

    statistics = pd.DataFrame(data=["mean", "variance"], columns=["statistic"])    # one row for mean, one row for variance

    for feature in observations.columns:
        statistics[feature] = [observations[feature].mean(), observations[feature].var()]

    return statistics


def features_from_observations(observations: pd.DataFrame) -> List[str]:

    features = list([column for column in observations.columns if column != "statistic"])

    return features
