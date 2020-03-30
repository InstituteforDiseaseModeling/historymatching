import abc

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

class BasisBase(abc.ABC):
  @abc.abstractmethod
  def fit_transform(self, X):
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
    self.polyfit = PolynomialFeatures(
      degree           = degree, 
      interaction_only = False, 
      include_bias     = intercept
    )

  def fit_transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise TypeError("Basis must be passed a DataFrame!")
    return pd.DataFrame(
      self.polyfit.fit_transform(X),
      columns = X.columns
    )



class IdentityBasis(BasisBase):
  def __init__(self, intercept):
    """Create a polynomial basis

    Args:
      degree - The degree of the polynomial features.
      intercept - Whether to include an intercept.
    """
    self.polyfit = PolynomialFeatures(
      degree           = 1, 
      interaction_only = False, 
      include_bias     = intercept
    )

  def fit_transform(self, X):
    if not isinstance(X, pd.DataFrame):
      raise TypeError("Basis must be passed a DataFrame!")
    return pd.DataFrame(
      self.polyfit.fit_transform(X),
      columns = X.columns
    )