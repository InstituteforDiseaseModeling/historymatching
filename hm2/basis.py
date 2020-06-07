import abc

import sklearn.preprocessing as skp
import pandas as pd

class BasisBase(abc.ABC): # pragma: no cover
  @abc.abstractmethod
  def __call__(self, X):
    """Function that runs the initialized model.

    Args:
      X - Data to be transformed
    """
    pass



class PolynomialBasis(BasisBase):
  def __init__(self, degree, intercept, scale=True):
    """Create a polynomial basis

    Args:
      degree (int): The degree of the polynomial features.
      intercept (bool): Whether to include an intercept.
      scale (bool): Whether to center and scale the data by centering to the
                    mean and component-wise scaling to unit variance.
    """
    assert isinstance(degree,int)
    assert degree>=0
    assert isinstance(intercept, bool)
    assert isinstance(scale,bool)

    self.scale = scale
    self.intercept = intercept
    self.polyfit = skp.PolynomialFeatures(
      degree           = degree,
      interaction_only = False,
      include_bias     = intercept
    )

  def __call__(self, X):
    """Apply the basis to X, performing scaling if requested"""
    if not isinstance(X, pd.DataFrame):
      raise TypeError("Basis must be passed a DataFrame!")
    X = X.copy()
    if self.scale:
      X[:] = skp.scale(X)
    fit     = self.polyfit.fit_transform(X)
    columns = self.polyfit.get_feature_names(X.columns)
    columns = ['Intercept' if x=='1' else x for x in columns]
    return pd.DataFrame(fit, columns = columns)



class IdentityBasis(BasisBase):
  def __init__(self, intercept, scale=True):
    """
    Create a polynomial basis

    Args:
      intercept (bool) - Whether to include an intercept.
      scale (bool): Whether to center and scale the data by centering to the
                    mean and component-wise scaling to unit variance.
    """
    assert isinstance(intercept,bool)
    assert isinstance(scale,bool)

    self.scale = scale
    self.intercept = intercept
    self.polyfit = skp.PolynomialFeatures(
      degree           = 1,
      interaction_only = False,
      include_bias     = intercept
    )

  def __call__(self, X):
    """Apply the basis to X, performing scaling if requested"""
    if not isinstance(X, pd.DataFrame):
      raise TypeError("Basis must be passed a DataFrame!")
    X = X.copy()
    if self.scale:
      X[:] = skp.scale(X)
    transformed = self.polyfit.fit_transform(X)

    columns = self.polyfit.get_feature_names(X.columns)
    columns = ['Intercept' if x=='1' else x for x in columns]

    return pd.DataFrame(self.polyfit.fit_transform(X), columns = columns)
