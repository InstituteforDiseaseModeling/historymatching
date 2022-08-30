import numpy as np
import pandas as pd
import unittest
from patsy import PatsyError

from history_matching.basis import Basis

class BasisTest(unittest.TestCase):

    def test_naming(self):
        # test creation of param dict by make_param_dict()
        # make_param_dict() sanitizes parameter names for use with the [patsy package](https://github.com/pydata/patsy)
        self.assertEqual(Basis.make_param_dict(['x']), {'x':'x'})
        self.assertEqual(Basis.make_param_dict(['x','y']), {'x':'x','y':'y'})
        self.assertEqual(Basis.make_param_dict(['t:e&s t-s']), {'t:e&s t-s':'te_s_t_s'})

    def test_identity(self):
        # test creation of identity basis
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.identity_basis(params=data.keys())
        self.assertEqual(b.param_dict, {'x':'x','y':'y','z':'z'})
        self.assertTrue((b.generate_dmatrix(data)==data).all().all())
 
    def test_none_value(self):
        # `None` is not a valid value for basis data and should raise an exception
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,None,6], 'z':[7,8,9]})
        b = Basis.identity_basis(params=data.keys())
        self.assertRaises(SystemExit, b.generate_dmatrix, data)
 
    def test_nan_value(self):
        # `NaN` is not a valid value for basis data and should raise an exception.
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,np.nan,6], 'z':[7,8,9]})
        b = Basis.identity_basis(params=data.keys())
        self.assertRaises(SystemExit, b.generate_dmatrix, data)
 
    def test_polynomial_intercept(self):
        # test polynomial basis creation with intercept, `generate_dmatrix()` returns a DataFrame representing the _formula_ of the basis,
        # `1` indicates an intercept/offset term in the given dimension/parameter
        # e.g. y ~ mx + b => x + 1 (linear in x, with an intercept/constant offset)
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, first_order=False)
        self.assertEqual(b.param_dict, {'x':'x', 'y':'y', 'z':'z'})
        intercept_frame = pd.DataFrame({'Intercept': [1,1,1]})
        self.assertTrue((b.generate_dmatrix(data)==intercept_frame).all().all())

    def test_polynomial_first(self):
        # test polynomial basis with first-order terms (the source parameters, e.g. `x`, `y`, and `z`, no squared, cubed, or higher powers)
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, first_order=True)
        self.assertEqual(b.param_dict, {'x':'x', 'y':'y', 'z':'z'})
        intercept_frame = pd.DataFrame({'Intercept': [1,1,1]})
        self.assertTrue((b.generate_dmatrix(data)==pd.concat([intercept_frame, data], axis=1)).all().all())

    def test_polynomial_second(self):
        # test polynomial basis with 2nd-order terms, the relevant coeff are formed by enumerating the right order of
        # terms from the resulting combination (say for 2nd order with (x,y) it's x*x, x*y, y*y), then multiple their coeff accordingly
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, second_order=True)
        self.assertEqual(b.param_dict, {'x':'x', 'y':'y', 'z':'z'})
        ans = pd.DataFrame({'Intercept': {0: 1.0, 1: 1.0, 2: 1.0}, 'x': {0: 1.0, 1: 2.0, 2: 3.0}, 'y': {0: 4.0, 1: 5.0, 2: 6.0}, 'z': {0: 7.0, 1: 8.0, 2: 9.0}, 'x ** 2': {0: 1.0, 1: 4.0, 2: 9.0}, 'y ** 2': {0: 16.0, 1: 25.0, 2: 36.0}, 'z ** 2': {0: 49.0, 1: 64.0, 2: 81.0}, 'x * y': {0: 4.0, 1: 10.0, 2: 18.0}, 'x * z': {0: 7.0, 1: 16.0, 2: 27.0}, 'y * z': {0: 28.0, 1: 40.0, 2: 54.0}})
        ans = ans.reindex(sorted(ans.columns), axis=1)
        g = b.generate_dmatrix(data)
        g = g.reindex(sorted(g.columns), axis=1)
        self.assertTrue((g==ans).all().all())

    def test_polynomial_third(self):
        # test polynomial basis with 3rd-order terms (see `test_polynomial_second()` above)
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, second_order=True, third_order=True)
        ans = pd.DataFrame({'Intercept': {0: 1.0, 1: 1.0, 2: 1.0}, 'x': {0: 1.0, 1: 2.0, 2: 3.0}, 'y': {0: 4.0, 1: 5.0, 2: 6.0}, 'z': {0: 7.0, 1: 8.0, 2: 9.0}, 'x ** 2': {0: 1.0, 1: 4.0, 2: 9.0}, 'y ** 2': {0: 16.0, 1: 25.0, 2: 36.0}, 'z ** 2': {0: 49.0, 1: 64.0, 2: 81.0}, 'x * y': {0: 4.0, 1: 10.0, 2: 18.0}, 'x * z': {0: 7.0, 1: 16.0, 2: 27.0}, 'y * z': {0: 28.0, 1: 40.0, 2: 54.0}, 'x ** 3': {0: 1.0, 1: 8.0, 2: 27.0}, 'y ** 3': {0: 64.0, 1: 125.0, 2: 216.0}, 'z ** 3': {0: 343.0, 1: 512.0, 2: 729.0}, 'x * y ** 2': {0: 16.0, 1: 50.0, 2: 108.0}, 'x * z ** 2': {0: 49.0, 1: 128.0, 2: 243.0}, 'y * z ** 2': {0: 196.0, 1: 320.0, 2: 486.0}, 'x ** 2 * y': {0: 4.0, 1: 20.0, 2: 54.0}, 'x ** 2 * z': {0: 7.0, 1: 32.0, 2: 81.0}, 'y ** 2 * z': {0: 112.0, 1: 200.0, 2: 324.0}, 'x * y * z': {0: 28.0, 1: 80.0, 2: 162.0}})
        ans = ans.reindex(sorted(ans.columns), axis=1)
        g = b.generate_dmatrix(data)
        g = g.reindex(sorted(g.columns), axis=1)
        self.assertTrue((g==ans).all().all())

    def test_polynomial_fourth(self):
        # test polynomial basis with 4th-order terms (see `test_polynomial_second()` above)
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, second_order=True, third_order=True, fourth_order=True)
        ans = pd.DataFrame({'Intercept': {0: 1.0, 1: 1.0, 2: 1.0}, 'x': {0: 1.0, 1: 2.0, 2: 3.0}, 'y': {0: 4.0, 1: 5.0, 2: 6.0}, 'z': {0: 7.0, 1: 8.0, 2: 9.0}, 'x ** 2': {0: 1.0, 1: 4.0, 2: 9.0}, 'y ** 2': {0: 16.0, 1: 25.0, 2: 36.0}, 'z ** 2': {0: 49.0, 1: 64.0, 2: 81.0}, 'x * y': {0: 4.0, 1: 10.0, 2: 18.0}, 'x * z': {0: 7.0, 1: 16.0, 2: 27.0}, 'y * z': {0: 28.0, 1: 40.0, 2: 54.0}, 'x ** 3': {0: 1.0, 1: 8.0, 2: 27.0}, 'y ** 3': {0: 64.0, 1: 125.0, 2: 216.0}, 'z ** 3': {0: 343.0, 1: 512.0, 2: 729.0}, 'x * y ** 2': {0: 16.0, 1: 50.0, 2: 108.0}, 'x * z ** 2': {0: 49.0, 1: 128.0, 2: 243.0}, 'y * z ** 2': {0: 196.0, 1: 320.0, 2: 486.0}, 'x ** 2 * y': {0: 4.0, 1: 20.0, 2: 54.0}, 'x ** 2 * z': {0: 7.0, 1: 32.0, 2: 81.0}, 'y ** 2 * z': {0: 112.0, 1: 200.0, 2: 324.0}, 'x * y * z': {0: 28.0, 1: 80.0, 2: 162.0}, 'x ** 4': {0: 1.0, 1: 16.0, 2: 81.0}, 'y ** 4': {0: 256.0, 1: 625.0, 2: 1296.0}, 'z ** 4': {0: 2401.0, 1: 4096.0, 2: 6561.0}, 'x ** 3 * y': {0: 4.0, 1: 40.0, 2: 162.0}, 'x ** 3 * z': {0: 7.0, 1: 64.0, 2: 243.0}, 'y ** 3 * z': {0: 448.0, 1: 1000.0, 2: 1944.0}, 'x * y ** 3': {0: 64.0, 1: 250.0, 2: 648.0}, 'x * z ** 3': {0: 343.0, 1: 1024.0, 2: 2187.0}, 'y * z ** 3': {0: 1372.0, 1: 2560.0, 2: 4374.0}, 'x ** 2 * y ** 2': {0: 16.0, 1: 100.0, 2: 324.0}, 'x ** 2 * z ** 2': {0: 49.0, 1: 256.0, 2: 729.0}, 'y ** 2 * z ** 2': {0: 784.0, 1: 1600.0, 2: 2916.0}, 'x ** 2 * y * z': {0: 28.0, 1: 160.0, 2: 486.0}, 'x * y ** 2 * z': {0: 112.0, 1: 400.0, 2: 972.0}, 'x * y * z ** 2': {0: 196.0, 1: 640.0, 2: 1458.0}})
        ans = ans.reindex(sorted(ans.columns), axis=1)
        g = b.generate_dmatrix(data)
        g = g.reindex(sorted(g.columns), axis=1)
        self.assertTrue((g==ans).all().all())

    def test_polynomial_fourth_only(self):
        # similar to test_polynomial_fourth() above, but this one illustrate when we specify four_order only then only 1st order(by default)
        # and fourth_order terms are expected to be there
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, fourth_order=True)
        ans = pd.DataFrame({'Intercept': {0: 1.0, 1: 1.0, 2: 1.0}, 'x': {0: 1.0, 1: 2.0, 2: 3.0}, 'y': {0: 4.0, 1: 5.0, 2: 6.0}, 'z': {0: 7.0, 1: 8.0, 2: 9.0}, 'x ** 4': {0: 1.0, 1: 16.0, 2: 81.0}, 'y ** 4': {0: 256.0, 1: 625.0, 2: 1296.0}, 'z ** 4': {0: 2401.0, 1: 4096.0, 2: 6561.0}, 'x ** 3 * y': {0: 4.0, 1: 40.0, 2: 162.0}, 'x ** 3 * z': {0: 7.0, 1: 64.0, 2: 243.0}, 'y ** 3 * z': {0: 448.0, 1: 1000.0, 2: 1944.0}, 'x * y ** 3': {0: 64.0, 1: 250.0, 2: 648.0}, 'x * z ** 3': {0: 343.0, 1: 1024.0, 2: 2187.0}, 'y * z ** 3': {0: 1372.0, 1: 2560.0, 2: 4374.0}, 'x ** 2 * y ** 2': {0: 16.0, 1: 100.0, 2: 324.0}, 'x ** 2 * z ** 2': {0: 49.0, 1: 256.0, 2: 729.0}, 'y ** 2 * z ** 2': {0: 784.0, 1: 1600.0, 2: 2916.0}, 'x ** 2 * y * z': {0: 28.0, 1: 160.0, 2: 486.0}, 'x * y ** 2 * z': {0: 112.0, 1: 400.0, 2: 972.0}, 'x * y * z ** 2': {0: 196.0, 1: 640.0, 2: 1458.0}})
        ans = ans.reindex(sorted(ans.columns), axis=1)
        g = b.generate_dmatrix(data)
        g = g.reindex(sorted(g.columns), axis=1)
        self.assertTrue((g==ans).all().all())

    def test_polynomial_fifth(self):
        # test polynomial basis with 5th-order terms (see `test_polynomial_second()` above)
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, second_order=True, third_order=True, fourth_order=True, fifth_order=True)
        ans = pd.DataFrame({'Intercept': {0: 1.0, 1: 1.0, 2: 1.0}, 'x': {0: 1.0, 1: 2.0, 2: 3.0}, 'y': {0: 4.0, 1: 5.0, 2: 6.0}, 'z': {0: 7.0, 1: 8.0, 2: 9.0}, 'x ** 2': {0: 1.0, 1: 4.0, 2: 9.0}, 'y ** 2': {0: 16.0, 1: 25.0, 2: 36.0}, 'z ** 2': {0: 49.0, 1: 64.0, 2: 81.0}, 'x * y': {0: 4.0, 1: 10.0, 2: 18.0}, 'x * z': {0: 7.0, 1: 16.0, 2: 27.0}, 'y * z': {0: 28.0, 1: 40.0, 2: 54.0}, 'x ** 3': {0: 1.0, 1: 8.0, 2: 27.0}, 'y ** 3': {0: 64.0, 1: 125.0, 2: 216.0}, 'z ** 3': {0: 343.0, 1: 512.0, 2: 729.0}, 'x * y ** 2': {0: 16.0, 1: 50.0, 2: 108.0}, 'x * z ** 2': {0: 49.0, 1: 128.0, 2: 243.0}, 'y * z ** 2': {0: 196.0, 1: 320.0, 2: 486.0}, 'x ** 2 * y': {0: 4.0, 1: 20.0, 2: 54.0}, 'x ** 2 * z': {0: 7.0, 1: 32.0, 2: 81.0}, 'y ** 2 * z': {0: 112.0, 1: 200.0, 2: 324.0}, 'x * y * z': {0: 28.0, 1: 80.0, 2: 162.0}, 'x ** 4': {0: 1.0, 1: 16.0, 2: 81.0}, 'y ** 4': {0: 256.0, 1: 625.0, 2: 1296.0}, 'z ** 4': {0: 2401.0, 1: 4096.0, 2: 6561.0}, 'x ** 3 * y': {0: 4.0, 1: 40.0, 2: 162.0}, 'x ** 3 * z': {0: 7.0, 1: 64.0, 2: 243.0}, 'y ** 3 * z': {0: 448.0, 1: 1000.0, 2: 1944.0}, 'x * y ** 3': {0: 64.0, 1: 250.0, 2: 648.0}, 'x * z ** 3': {0: 343.0, 1: 1024.0, 2: 2187.0}, 'y * z ** 3': {0: 1372.0, 1: 2560.0, 2: 4374.0}, 'x ** 2 * y ** 2': {0: 16.0, 1: 100.0, 2: 324.0}, 'x ** 2 * z ** 2': {0: 49.0, 1: 256.0, 2: 729.0}, 'y ** 2 * z ** 2': {0: 784.0, 1: 1600.0, 2: 2916.0}, 'x ** 2 * y * z': {0: 28.0, 1: 160.0, 2: 486.0}, 'x * y ** 2 * z': {0: 112.0, 1: 400.0, 2: 972.0}, 'x * y * z ** 2': {0: 196.0, 1: 640.0, 2: 1458.0}, 'x ** 5': {0: 1.0, 1: 32.0, 2: 243.0}, 'y ** 5': {0: 1024.0, 1: 3125.0, 2: 7776.0}, 'z ** 5': {0: 16807.0, 1: 32768.0, 2: 59049.0}, 'x ** 4 * y': {0: 4.0, 1: 80.0, 2: 486.0}, 'x ** 4 * z': {0: 7.0, 1: 128.0, 2: 729.0}, 'y ** 4 * z': {0: 1792.0, 1: 5000.0, 2: 11664.0}, 'x * y ** 4': {0: 256.0, 1: 1250.0, 2: 3888.0}, 'x * z ** 4': {0: 2401.0, 1: 8192.0, 2: 19683.0}, 'y * z ** 4': {0: 9604.0, 1: 20480.0, 2: 39366.0}, 'x ** 3 * y ** 2': {0: 16.0, 1: 200.0, 2: 972.0}, 'x ** 3 * z ** 2': {0: 49.0, 1: 512.0, 2: 2187.0}, 'y ** 3 * z ** 2': {0: 3136.0, 1: 8000.0, 2: 17496.0}, 'x ** 2 * y ** 3': {0: 64.0, 1: 500.0, 2: 1944.0}, 'x ** 2 * z ** 3': {0: 343.0, 1: 2048.0, 2: 6561.0}, 'y ** 2 * z ** 3': {0: 5488.0, 1: 12800.0, 2: 26244.0}, 'x ** 3 * y * z': {0: 28.0, 1: 320.0, 2: 1458.0}, 'x * y ** 3 * z': {0: 448.0, 1: 2000.0, 2: 5832.0}, 'x * y * z ** 3': {0: 1372.0, 1: 5120.0, 2: 13122.0}, 'x ** 2 * y ** 2 * z': {0: 112.0, 1: 800.0, 2: 2916.0}, 'x ** 2 * y * z ** 2': {0: 196.0, 1: 1280.0, 2: 4374.0}, 'x * y ** 2 * z ** 2': {0: 784.0, 1: 3200.0, 2: 8748.0}})
        ans = ans.reindex(sorted(ans.columns), axis=1)
        g = b.generate_dmatrix(data)
        g = g.reindex(sorted(g.columns), axis=1)
        self.assertTrue((g==ans).all().all())

    def test_polynomial_higher(self):
        # test polynomial basis with higher order terms (a selected subset of all possible 6th and 7th order terms).
        data = pd.DataFrame({'x':[1,2,3], 'y':[4,5,6], 'z':[7,8,9]})
        b = Basis.polynomial_basis(params=data.keys(), intercept=True, second_order=True, third_order=True, fourth_order=True, fifth_order=True, higher_order=True)
        ans = pd.DataFrame({'Intercept': {0: 1.0, 1: 1.0, 2: 1.0}, 'x': {0: 1.0, 1: 2.0, 2: 3.0}, 'y': {0: 4.0, 1: 5.0, 2: 6.0}, 'z': {0: 7.0, 1: 8.0, 2: 9.0}, 'x ** 2': {0: 1.0, 1: 4.0, 2: 9.0}, 'y ** 2': {0: 16.0, 1: 25.0, 2: 36.0}, 'z ** 2': {0: 49.0, 1: 64.0, 2: 81.0}, 'x * y': {0: 4.0, 1: 10.0, 2: 18.0}, 'x * z': {0: 7.0, 1: 16.0, 2: 27.0}, 'y * z': {0: 28.0, 1: 40.0, 2: 54.0}, 'x ** 3': {0: 1.0, 1: 8.0, 2: 27.0}, 'y ** 3': {0: 64.0, 1: 125.0, 2: 216.0}, 'z ** 3': {0: 343.0, 1: 512.0, 2: 729.0}, 'x * y ** 2': {0: 16.0, 1: 50.0, 2: 108.0}, 'x * z ** 2': {0: 49.0, 1: 128.0, 2: 243.0}, 'y * z ** 2': {0: 196.0, 1: 320.0, 2: 486.0}, 'x ** 2 * y': {0: 4.0, 1: 20.0, 2: 54.0}, 'x ** 2 * z': {0: 7.0, 1: 32.0, 2: 81.0}, 'y ** 2 * z': {0: 112.0, 1: 200.0, 2: 324.0}, 'x * y * z': {0: 28.0, 1: 80.0, 2: 162.0}, 'x ** 4': {0: 1.0, 1: 16.0, 2: 81.0}, 'y ** 4': {0: 256.0, 1: 625.0, 2: 1296.0}, 'z ** 4': {0: 2401.0, 1: 4096.0, 2: 6561.0}, 'x ** 3 * y': {0: 4.0, 1: 40.0, 2: 162.0}, 'x ** 3 * z': {0: 7.0, 1: 64.0, 2: 243.0}, 'y ** 3 * z': {0: 448.0, 1: 1000.0, 2: 1944.0}, 'x * y ** 3': {0: 64.0, 1: 250.0, 2: 648.0}, 'x * z ** 3': {0: 343.0, 1: 1024.0, 2: 2187.0}, 'y * z ** 3': {0: 1372.0, 1: 2560.0, 2: 4374.0}, 'x ** 2 * y ** 2': {0: 16.0, 1: 100.0, 2: 324.0}, 'x ** 2 * z ** 2': {0: 49.0, 1: 256.0, 2: 729.0}, 'y ** 2 * z ** 2': {0: 784.0, 1: 1600.0, 2: 2916.0}, 'x ** 2 * y * z': {0: 28.0, 1: 160.0, 2: 486.0}, 'x * y ** 2 * z': {0: 112.0, 1: 400.0, 2: 972.0}, 'x * y * z ** 2': {0: 196.0, 1: 640.0, 2: 1458.0}, 'x ** 5': {0: 1.0, 1: 32.0, 2: 243.0}, 'y ** 5': {0: 1024.0, 1: 3125.0, 2: 7776.0}, 'z ** 5': {0: 16807.0, 1: 32768.0, 2: 59049.0}, 'x ** 4 * y': {0: 4.0, 1: 80.0, 2: 486.0}, 'x ** 4 * z': {0: 7.0, 1: 128.0, 2: 729.0}, 'y ** 4 * z': {0: 1792.0, 1: 5000.0, 2: 11664.0}, 'x * y ** 4': {0: 256.0, 1: 1250.0, 2: 3888.0}, 'x * z ** 4': {0: 2401.0, 1: 8192.0, 2: 19683.0}, 'y * z ** 4': {0: 9604.0, 1: 20480.0, 2: 39366.0}, 'x ** 3 * y ** 2': {0: 16.0, 1: 200.0, 2: 972.0}, 'x ** 3 * z ** 2': {0: 49.0, 1: 512.0, 2: 2187.0}, 'y ** 3 * z ** 2': {0: 3136.0, 1: 8000.0, 2: 17496.0}, 'x ** 2 * y ** 3': {0: 64.0, 1: 500.0, 2: 1944.0}, 'x ** 2 * z ** 3': {0: 343.0, 1: 2048.0, 2: 6561.0}, 'y ** 2 * z ** 3': {0: 5488.0, 1: 12800.0, 2: 26244.0}, 'x ** 3 * y * z': {0: 28.0, 1: 320.0, 2: 1458.0}, 'x * y ** 3 * z': {0: 448.0, 1: 2000.0, 2: 5832.0}, 'x * y * z ** 3': {0: 1372.0, 1: 5120.0, 2: 13122.0}, 'x ** 2 * y ** 2 * z': {0: 112.0, 1: 800.0, 2: 2916.0}, 'x ** 2 * y * z ** 2': {0: 196.0, 1: 1280.0, 2: 4374.0}, 'x * y ** 2 * z ** 2': {0: 784.0, 1: 3200.0, 2: 8748.0}, 'x ** 6': {0: 1.0, 1: 64.0, 2: 729.0}, 'y ** 6': {0: 4096.0, 1: 15625.0, 2: 46656.0}, 'z ** 6': {0: 117649.0, 1: 262144.0, 2: 531441.0}, 'x ** 5 * y': {0: 4.0, 1: 160.0, 2: 1458.0}, 'x ** 5 * z': {0: 7.0, 1: 256.0, 2: 2187.0}, 'y ** 5 * z': {0: 7168.0, 1: 25000.0, 2: 69984.0}, 'x * y ** 5': {0: 1024.0, 1: 6250.0, 2: 23328.0}, 'x * z ** 5': {0: 16807.0, 1: 65536.0, 2: 177147.0}, 'y * z ** 5': {0: 67228.0, 1: 163840.0, 2: 354294.0}, 'x ** 7': {0: 1.0, 1: 128.0, 2: 2187.0}, 'y ** 7': {0: 16384.0, 1: 78125.0, 2: 279936.0}, 'z ** 7': {0: 823543.0, 1: 2097152.0, 2: 4782969.0}, 'x ** 6 * y': {0: 4.0, 1: 320.0, 2: 4374.0}, 'x ** 6 * z': {0: 7.0, 1: 512.0, 2: 6561.0}, 'y ** 6 * z': {0: 28672.0, 1: 125000.0, 2: 419904.0}, 'x * y ** 6': {0: 4096.0, 1: 31250.0, 2: 139968.0}, 'x * z ** 6': {0: 117649.0, 1: 524288.0, 2: 1594323.0}, 'y * z ** 6': {0: 470596.0, 1: 1310720.0, 2: 3188646.0}})
        ans = ans.reindex(sorted(ans.columns), axis=1)
        g = b.generate_dmatrix(data)
        g = g.reindex(sorted(g.columns), axis=1)
        self.assertTrue((g==ans).all().all())


