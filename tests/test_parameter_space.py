"""
Unit tests for ParameterSpace domain object.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from historymatching.constants import PARAMETER_SPACE_COLUMNS
from fixtures import TestDataFactory, TestAssertions, TestConstants
import historymatching as hm


class ParameterSpaceTests(unittest.TestCase):
    """Tests for ParameterSpace class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.simple_params_df = TestDataFactory.create_simple_parameter_space(3)
        self.simple_space = hm.ParameterSpace(self.simple_params_df)
        
    def test_constructor_with_dataframe(self):
        """Test constructor with valid DataFrame."""
        space = hm.ParameterSpace(self.simple_params_df)
        
        self.assertEqual(len(space), 3)
        self.assertEqual(space.get_parameter_names(), ['param_a', 'param_b', 'param_c'])
        
    def test_constructor_with_dict(self):
        """Test constructor with parameter dict."""
        param_dict = {
            'param_a': (0.0, 10.0),
            'param_b': (10.0, 20.0),
            'param_c': (20.0, 30.0)
        }
        
        space = hm.ParameterSpace(param_dict)
        
        self.assertEqual(len(space), 3)
        self.assertEqual(space.get_parameter_names(), ['param_a', 'param_b', 'param_c'])
        self.assertEqual(space.get_bounds('param_a'), (0.0, 10.0))
        
    def test_constructor_validation_missing_columns(self):
        """Test constructor fails with missing columns."""
        invalid_df = pd.DataFrame([['param_a', 0.0]], columns=['parameter', 'minimum'])
        
        with self.assertRaises(ValueError) as cm:
            hm.ParameterSpace(invalid_df)
        self.assertIn("Parameter space must have columns", str(cm.exception))
        
    def test_constructor_validation_invalid_bounds(self):
        """Test constructor fails with invalid bounds."""
        # Test minimum >= maximum
        invalid_df = pd.DataFrame([
            ['param_a', 10.0, 5.0]  # min > max
        ], columns=PARAMETER_SPACE_COLUMNS)
        
        with self.assertRaises(ValueError) as cm:
            hm.ParameterSpace(invalid_df)
        self.assertIn("minimum", str(cm.exception))
        self.assertIn("maximum", str(cm.exception))
        
    def test_constructor_validation_non_finite_bounds(self):
        """Test constructor fails with non-finite bounds."""
        invalid_df = pd.DataFrame([
            ['param_a', np.inf, 10.0]  # infinite minimum
        ], columns=PARAMETER_SPACE_COLUMNS)
        
        with self.assertRaises(ValueError) as cm:
            hm.ParameterSpace(invalid_df)
        self.assertIn("non-finite bounds", str(cm.exception))
        
    def test_constructor_validation_duplicate_parameters(self):
        """Test constructor fails with duplicate parameter names."""
        invalid_df = pd.DataFrame([
            ['param_a', 0.0, 10.0],
            ['param_a', 5.0, 15.0]  # duplicate name
        ], columns=PARAMETER_SPACE_COLUMNS)
        
        with self.assertRaises(ValueError) as cm:
            hm.ParameterSpace(invalid_df)
        self.assertIn("Duplicate parameter names", str(cm.exception))
        
    def test_get_bounds(self):
        """Test getting parameter bounds."""
        bounds_a = self.simple_space.get_bounds('param_a')
        self.assertEqual(bounds_a, (0.0, 10.0))
        
        bounds_b = self.simple_space.get_bounds('param_b')
        self.assertEqual(bounds_b, (10.0, 20.0))
        
    def test_get_bounds_invalid_parameter(self):
        """Test getting bounds for non-existent parameter."""
        with self.assertRaises(ValueError) as cm:
            self.simple_space.get_bounds('nonexistent_param')
        self.assertIn("not found in parameter space", str(cm.exception))
        
    def test_get_parameter_names(self):
        """Test getting parameter names."""
        names = self.simple_space.get_parameter_names()
        expected = ['param_a', 'param_b', 'param_c']
        self.assertEqual(names, expected)
        
    def test_constrain_parameter(self):
        """Test constraining a single parameter."""
        # Constrain param_a from [0, 10] to [2, 8]
        new_space = self.simple_space.constrain_parameter('param_a', 2.0, 8.0)
        
        # Check new bounds
        self.assertEqual(new_space.get_bounds('param_a'), (2.0, 8.0))
        # Other parameters unchanged
        self.assertEqual(new_space.get_bounds('param_b'), (10.0, 20.0))
        self.assertEqual(new_space.get_bounds('param_c'), (20.0, 30.0))
        
        # Original space unchanged
        self.assertEqual(self.simple_space.get_bounds('param_a'), (0.0, 10.0))
        
    def test_constrain_parameter_invalid_bounds(self):
        """Test constraining parameter with invalid bounds."""
        # Test min >= max
        with self.assertRaises(ValueError) as cm:
            self.simple_space.constrain_parameter('param_a', 8.0, 2.0)
        self.assertIn("minimum", str(cm.exception))
        
    def test_constrain_parameter_expansion(self):
        """Test constraining parameter cannot expand bounds."""
        # Try to expand bounds beyond current limits
        with self.assertRaises(ValueError) as cm:
            self.simple_space.constrain_parameter('param_a', -5.0, 15.0)
        self.assertIn("Cannot expand parameter space", str(cm.exception))
        
    def test_constrain_parameter_nonexistent(self):
        """Test constraining non-existent parameter."""
        with self.assertRaises(ValueError) as cm:
            self.simple_space.constrain_parameter('nonexistent', 1.0, 2.0)
        self.assertIn("not found", str(cm.exception))
        
    def test_constrain_to_samples(self):
        """Test constraining space to sample bounds."""
        # Create samples within current bounds
        samples = TestDataFactory.create_sample_data(self.simple_params_df, n_samples=100, seed=42)
        
        # Constrain to 90th percentile
        new_space = self.simple_space.constrain_to_samples(samples, percentile=90)
        
        # New bounds should be within original bounds
        for param in self.simple_space.get_parameter_names():
            orig_min, orig_max = self.simple_space.get_bounds(param)
            new_min, new_max = new_space.get_bounds(param)
            
            self.assertGreaterEqual(new_min, orig_min)
            self.assertLessEqual(new_max, orig_max)
            self.assertLess(new_min, new_max)
            
    def test_constrain_to_samples_invalid(self):
        """Test constraining to samples outside bounds."""
        # Create samples outside bounds
        invalid_samples = pd.DataFrame({
            'param_a': [-5.0, 15.0],  # Outside [0, 10]
            'param_b': [5.0, 25.0],   # Outside [10, 20]
            'param_c': [15.0, 35.0]   # Outside [20, 30]
        })
        
        # Out-of-bounds samples are now allowed (with a warning) — the result
        # is clipped to the current parameter space bounds.
        constrained = self.simple_space.constrain_to_samples(invalid_samples)
        for param in ['param_a', 'param_b', 'param_c']:
            orig_min, orig_max = self.simple_space.get_bounds(param)
            new_min, new_max = constrained.get_bounds(param)
            self.assertGreaterEqual(new_min, orig_min)
            self.assertLessEqual(new_max, orig_max)
        
    def test_volume_fraction_remaining(self):
        """Test calculating volume fraction remaining."""
        # Create constrained space (half the volume in each dimension)
        constrained_dict = {
            'param_a': (2.5, 7.5),   # 5/10 = 0.5
            'param_b': (12.5, 17.5), # 5/10 = 0.5  
            'param_c': (22.5, 27.5)  # 5/10 = 0.5
        }
        constrained_space = hm.ParameterSpace(constrained_dict)
        
        # Volume fraction should be 0.5^3 = 0.125
        fraction = constrained_space.volume_fraction_remaining(self.simple_space)
        self.assertAlmostEqual(fraction, 0.125, places=6)
        
    def test_volume_fraction_remaining_different_parameters(self):
        """Test volume fraction with different parameter sets."""
        different_dict = {
            'param_x': (0.0, 10.0),
            'param_y': (10.0, 20.0)
        }
        different_space = hm.ParameterSpace(different_dict)
        
        with self.assertRaises(ValueError) as cm:
            self.simple_space.volume_fraction_remaining(different_space)
        self.assertIn("same parameters", str(cm.exception))
        
    def test_sample_uniformly(self):
        """Test uniform sampling."""
        samples = self.simple_space.sample_uniformly(100, seed=42)
        
        # Check structure
        self.assertEqual(len(samples), 100)
        expected_columns = ['param_a', 'param_b', 'param_c']
        self.assertEqual(list(samples.columns), expected_columns)
        
        # Check bounds are respected
        TestAssertions.assert_parameter_bounds_respected(samples, self.simple_params_df)
        
    def test_sample_uniformly_reproducible(self):
        """Test uniform sampling is reproducible with seed."""
        samples1 = self.simple_space.sample_uniformly(50, seed=123)
        samples2 = self.simple_space.sample_uniformly(50, seed=123)
        
        pd.testing.assert_frame_equal(samples1, samples2)
        
    def test_validate_samples_valid(self):
        """Test validating valid samples."""
        valid_samples = TestDataFactory.create_sample_data(self.simple_params_df, n_samples=50)
        
        self.assertTrue(self.simple_space.validate_samples(valid_samples))
        
    def test_validate_samples_invalid(self):
        """Test validating invalid samples."""
        invalid_samples = pd.DataFrame({
            'param_a': [-1.0, 11.0],  # Outside [0, 10]
            'param_b': [9.0, 21.0],   # Outside [10, 20]
            'param_c': [19.0, 31.0]   # Outside [20, 30]
        })
        
        self.assertFalse(self.simple_space.validate_samples(invalid_samples))
        
    def test_validate_samples_missing_parameters(self):
        """Test validating samples with missing parameters."""
        partial_samples = pd.DataFrame({
            'param_a': [5.0, 7.0],
            # param_b and param_c missing
        })
        
        # Should still return True (missing parameters are skipped)
        self.assertTrue(self.simple_space.validate_samples(partial_samples))
        
    def test_to_dataframe(self):
        """Test converting to DataFrame."""
        df = self.simple_space.to_dataframe()
        
        # Check structure
        TestAssertions.assert_dataframe_structure(df, PARAMETER_SPACE_COLUMNS)
        self.assertEqual(len(df), 3)
        
        # Check data integrity
        for i, param_name in enumerate(['param_a', 'param_b', 'param_c']):
            row = df[df['parameter'] == param_name].iloc[0]
            expected_min = i * 10.0
            expected_max = expected_min + 10.0
            self.assertEqual(row['minimum'], expected_min)
            self.assertEqual(row['maximum'], expected_max)
            
    def test_len(self):
        """Test length operator."""
        self.assertEqual(len(self.simple_space), 3)
        
        single_param_dict = {'param_x': (0.0, 1.0)}
        single_space = hm.ParameterSpace(single_param_dict)
        self.assertEqual(len(single_space), 1)
        
    def test_equality(self):
        """Test equality comparison."""
        # Same data should be equal
        space1 = hm.ParameterSpace(self.simple_params_df)
        space2 = hm.ParameterSpace(self.simple_params_df.copy())
        self.assertEqual(space1, space2)
        
        # Different data should not be equal
        different_df = TestDataFactory.create_simple_parameter_space(2)
        space3 = hm.ParameterSpace(different_df)
        self.assertNotEqual(space1, space3)
        
        # Different type should not be equal
        self.assertNotEqual(space1, "not a parameter space")
        
    def test_repr(self):
        """Test string representation."""
        repr_str = repr(self.simple_space)
        self.assertIn("ParameterSpace", repr_str)
        self.assertIn("3 parameters", repr_str)
        self.assertIn("param_a", repr_str)


if __name__ == '__main__':
    unittest.main()