"""Bring emulators into `emulators` namespace so we don't have to
`import history_matching.emulators.linear.LinearModel`, for example."""

from .base import BaseEmulator  # noqa: F401 isort: skip
from .linear import LinearModel  # noqa: F401 isort: skip
from .glm import GLM  # noqa: F401 isort: skip
from .gpr import GPR  # noqa: F401 isort: skip
