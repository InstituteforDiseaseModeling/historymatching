import unittest

import numpy as np
import pandas as pd

from hm2.error import HistoryMatchingError
from hm2.glm import GLM
from hm2.basis import IdentityBasis



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

    def test_fit_predict(self):
      glm = GLM(family="poisson")
      x = pd.DataFrame({"x":np.linspace(start=0, stop=100, num=1000)})
      y = 3*np.linspace(start=0, stop=100, num=1000)
      basis=IdentityBasis(intercept=True, scale=False)
      glm.fit(train_x=basis(x), train_y=y)
      ypred = glm.predict(basis(x))
      # breakpoint() #TODO: This test doesn't work yet. GLM isn't rpedicting right values.

    # def fit(self, train_x, train_y, maxiter=1000):