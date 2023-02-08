"""Parameter space point samplers."""

from itertools import product
# import logging

import numpy as np
import pandas as pd

# logger = logging.getLogger()


def lhs(parameter_space: pd.DataFrame, n_samples: int = 8) -> pd.DataFrame:

    # TODO - consider delegating to pyDOE lhs() and its options
    samples = pd.DataFrame()

    for entry in parameter_space.itertuples():

        # n+1 edges = n buckets
        edges = np.linspace(entry.minimum, entry.maximum, n_samples + 1)
        # choose points from center of buckets
        points = (edges[:-1] + edges[1:]) / 2
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        samples[entry.parameter] = points[indices]

    return samples


def grid(
    parameter_space: pd.DataFrame, samples_per_dimension: int = 16
) -> pd.DataFrame:

    values = []
    for entry in parameter_space.itertuples():
        edges = np.linspace(entry.minimum, entry.maximum, samples_per_dimension + 1)
        points = (edges[:-1] + edges[1:]) / 2
        values.append(points)
    cartesian = product(*values)
    samples = pd.DataFrame(cartesian, columns=parameter_space.parameter)

    return samples


def random(parameter_space: pd.DataFrame, n_samples: int = 16) -> pd.DataFrame:

    samples = pd.DataFrame()

    for entry in parameter_space.itertuples():

        points = np.random.default_rng().uniform(entry.minimum, entry.maximum, n_samples)
        samples[entry.parameter] = points

    return samples
