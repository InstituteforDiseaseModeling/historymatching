import unittest

import numpy as np
import pandas as pd

import hm2.basis
from .data import test_params_df, test_y



class TestIdentityBasis(unittest.TestCase):
    def test_bad_intercept(self):
      self.assertRaises(AssertionError, hm2.basis.IdentityBasis, intercept="bad")

    def test_bad_scale(self):
      self.assertRaises(AssertionError, hm2.basis.IdentityBasis, intercept=True, scale="bad")

    def test_needs_df(self):
      ibasis = hm2.basis.IdentityBasis(intercept=False, scale=False)
      self.assertRaises(TypeError, ibasis, "bad'")

    def test_pure_identity_basis(self):
      ibasis = hm2.basis.IdentityBasis(intercept=False, scale=False)
      basified_df = ibasis(test_params_df)
      self.assertTrue(basified_df.equals(test_params_df))

    def test_intercept_identity_basis(self):
      ibasis = hm2.basis.IdentityBasis(intercept=True, scale=False)
      basified_df = ibasis(test_params_df)
      test_params_df_copy = test_params_df.copy()
      test_params_df_copy.insert(0, 'Intercept', 1.0)
      self.assertTrue(basified_df.equals(test_params_df_copy))

#TODO: Test scaling



class TestPolynomialBasis(unittest.TestCase):
    def test_bad_intercept(self):
      self.assertRaises(AssertionError, hm2.basis.PolynomialBasis, degree=2, intercept="bad")

    def test_bad_scale(self):
      self.assertRaises(AssertionError, hm2.basis.PolynomialBasis, degree=2, intercept=True, scale="bad")

    def test_degree_bad(self):
      self.assertRaises(AssertionError, hm2.basis.PolynomialBasis, degree="bad", intercept=True)

    def test_degree_good_number(self):
      self.assertRaises(AssertionError, hm2.basis.PolynomialBasis, degree=-1, intercept=True)

    def test_needs_df(self):
      ibasis = hm2.basis.PolynomialBasis(degree=1, intercept=False, scale=False)
      self.assertRaises(TypeError, ibasis, "bad'")

    def test_pure_identity_basis(self):
      ibasis = hm2.basis.PolynomialBasis(degree=1, intercept=False, scale=False)
      basified_df = ibasis(test_params_df)
      self.assertTrue(basified_df.equals(test_params_df))

    def test_intercept_identity_basis(self):
      ibasis = hm2.basis.PolynomialBasis(degree=1, intercept=True, scale=False)
      basified_df = ibasis(test_params_df)
      test_params_df_copy = test_params_df.copy()
      test_params_df_copy.insert(0, 'Intercept', 1.0)
      self.assertTrue(basified_df.equals(test_params_df_copy))

    def test_higher_order(self):
      ibasis = hm2.basis.PolynomialBasis(degree=3, intercept=True, scale=False)
      basified_df = ibasis(test_params_df)
      self.assertListEqual(basified_df.columns.tolist(), ['Intercept', 'beta', 'gamma', 'beta^2', 'beta gamma', 'gamma^2', 'beta^3', 'beta^2 gamma', 'beta gamma^2', 'gamma^3'])
