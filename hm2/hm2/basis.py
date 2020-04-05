import abc

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class BasisBase(abc.ABC):
  @abc.abstractmethod
  def __call__(self, X):
    """Function that runs the initialized model.

    Args:
      X - Data to be transformed
    """
    pass



class PolynomialBasis(BasisBase):
  def __init__(self, degree, intercept):
    """Create a polynomial basis

    Args:
      degree - The degree of the polynomial features.
      intercept - Whether to include an intercept.
    """
    self.intercept = intercept
    self.polyfit = PolynomialFeatures(
      degree           = degree, 
      interaction_only = False, 
      include_bias     = intercept
    )

  def __call__(self, X):
    if not isinstance(X, pd.DataFrame):
      raise TypeError("Basis must be passed a DataFrame!")
    fit     = self.polyfit.fit_transform(X)
    columns = self.polyfit.get_feature_names()
    columns = ['Intercept' if x=='1' else x for x in columns]
    return pd.DataFrame(fit, columns = columns)



class IdentityBasis(BasisBase):
  def __init__(self, intercept):
    """Create a polynomial basis

    Args:
      intercept (bool) - Whether to include an intercept.
    """
    self.intercept = intercept
    self.polyfit = PolynomialFeatures(
      degree           = 1, 
      interaction_only = False, 
      include_bias     = intercept
    )

  def __call__(self, X):
    if not isinstance(X, pd.DataFrame):
      raise TypeError("Basis must be passed a DataFrame!")
    transformed = self.polyfit.fit_transform(X)
    return pd.DataFrame(
      self.polyfit.fit_transform(X),
      columns = (['Intercept'] if self.intercept else []) + X.columns.tolist()
    )