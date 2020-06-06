import unittest

import numpy as np
import pandas as pd

from hm2.basis import IdentityBasis
from hm2.error import HistoryMatchingError
from hm2.glm import GLM



class GLMTest(unittest.TestCase):
    def test_bad_family(self):
      self.assertRaises(HistoryMatchingError, GLM, family="bad")
      self.assertRaises(HistoryMatchingError, GLM, family=234234)
      self.assertRaises(HistoryMatchingError, GLM, family=None)

    def test_families(self):
      GLM(family="binomial")
      GLM(family="gamma")
      GLM(family="gaussian")
      GLM(family="poisson")

    def test_untrained(self):
      glm = GLM(family="poisson")
      self.assertRaises(HistoryMatchingError, glm.predict, [1,2,3])

    def test_fit_predict_line_through_origin(self):
      #See if we can predict y=3*x
      np.random.seed(123456)
      glm = GLM(family="gaussian")
      x = pd.DataFrame({"x":np.linspace(start=0, stop=100, num=1000)})
      y = 3*np.linspace(start=0, stop=100, num=1000)+0.01*np.random.randn(1000)
      basis=IdentityBasis(intercept=False, scale=False)
      glm.fit(train_x=basis(x), train_y=y)
      ypred = glm.predict(basis(x))
      self.assertTrue(((y-ypred)<0.2).all()) #NOTE: This should be safe because of the random seed

    def test_fit_predict_line(self):
      #See if we can predict y=3*x+4
      np.random.seed(123456)
      glm = GLM(family="gaussian")
      x = pd.DataFrame({"x":np.linspace(start=0, stop=100, num=1000)})
      y = 3*np.linspace(start=0, stop=100, num=1000)+0.01*np.random.randn(1000)+4
      basis=IdentityBasis(intercept=True, scale=False)
      glm.fit(train_x=basis(x), train_y=y)
      ypred = glm.predict(basis(x))
      self.assertTrue(((y-ypred)<0.2).all()) #NOTE: This should be safe because of the random seed

    # def fit(self, train_x, train_y, maxiter=1000):