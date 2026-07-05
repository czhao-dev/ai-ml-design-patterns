"""Common interface every uplift/CATE estimator in this project implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class UpliftModel(ABC):
    """A conditional average treatment effect (CATE) estimator.

    `fit` takes the observed covariates, the binary treatment assignment, and
    the observed outcome. `predict_uplift` returns, for each row, the
    estimated individual treatment effect tau(x) = E[Y(1) - Y(0) | X=x] --
    the incremental outcome the treatment is predicted to cause for that
    customer, not the customer's raw predicted outcome.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray) -> "UpliftModel":
        raise NotImplementedError

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
