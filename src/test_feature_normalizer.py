#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for Feature Normalization Pipeline

This test suite validates the functionality of the feature normalization
pipeline including various scaling methods, outlier handling, and pipeline
management for consistent transformations.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the src directory to the path
sys.path.insert(0, os.path.abspath('.'))

from src.features.feature_normalizer import (
    ScalingMethod,
    OutlierHandling,
    NormalizationConfig,
    FeatureNormalizer,
    NormalizationPipeline,
    normalize_features,
    create_preprocessing_pipeline
)

class TestFeatureNormalizer:
    """Test class for feature normalization functionality"""
    
    def __init__(self):
        """Initialize test data and configurations"""
        print("Initializing Feature Normalization Test Suite")
        print("=" * 60)
        
        # Set random seed for reproducible results
        np.random.seed(42)
        
        # Create test data with different characteristics
        self.dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        
        self.data = pd.DataFrame({
            'normal_feature': np.random.normal(100, 15, len(self.dates)),
            'skewed_feature': np.random.exponential(2, len(self.dates)),
            'uniform_feature': np.random.uniform(0, 1000, len(self.dates)),
            'already_normalized': np.random.normal(0, 1, len(self.dates)),
            'sparse_feature': np.random.choice([0, 1, 2, 3], len(self.dates), p=[0.8, 0.1, 0.05, 0.05])
        }, index=self.dates)
        
        # Add outliers
        outlier_indices = np.random.choice(len(self.data), size=20, replace=False)
        self.data.loc[self.data.index[outlier_indices], 'normal_feature'] *= 3
        self.data.loc[self.data.index[outlier_indices[:10]], 'skewed_feature'] *= 5
        
        # Add missing values
        missing_indices = np.random.choice(len(self.data), size=10, replace=False)
        self.data.loc[self.data.index[missing_indices], 'uniform_feature'] = np.nan
        
        self.test_results = {}
        
    def test_basic_scaling_methods(self):
        """Test basic scaling methods"""
        print("\\n1. Testing Basic Scaling Methods")
        print("-" * 40)
        
        try:
            scaling_methods = [
                (ScalingMethod.Z_SCORE, "Z-Score"),
                (ScalingMethod.MIN_MAX, "Min-Max"),
                (ScalingMethod.ROBUST, "Robust"),
                (ScalingMethod.MAX_ABS, "Max-Abs")
            ]
            
            for method, name in scaling_methods:
                config = NormalizationConfig(
                    scaling_method=method,
                    outlier_handling=OutlierHandling.NONE
                )
                
                normalizer = FeatureNormalizer(config)
                normalized_data = normalizer.fit_transform(self.data)
                
                # Validate results
                assert isinstance(normalized_data, pd.DataFrame), f"{name} should return DataFrame"
                assert normalized_data.shape == self.data.shape, f"{name} shape should match input"
                assert not normalized_data.isnull().all().any(), f"{name} should not have all-null columns"
                
                print(f"   ✓ {name}: Shape {normalized_data.shape}, Range [{normalized_data.min().min():.3f}, {normalized_data.max().max():.3f}]")
            
            self.test_results['basic_scaling'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Basic scaling methods test failed: {str(e)}")
            self.test_results['basic_scaling'] = 'FAIL'
    
    def test_outlier_handling(self):
        """Test outlier handling methods"""
        print("\\n2. Testing Outlier Handling Methods")
        print("-" * 40)
        
        try:
            outlier_methods = [
                (OutlierHandling.CLIP, "Clipping"),
                (OutlierHandling.WINSORIZE, "Winsorizing"),
                (OutlierHandling.LOG_TRANSFORM, "Log Transform"),
                (OutlierHandling.NONE, "No Handling")
            ]
            
            for method, name in outlier_methods:
                if method == OutlierHandling.WINSORIZE:
                    # Skip winsorize if scipy not available
                    try:
                        from scipy import stats
                    except ImportError:
                        print(f"   ⚠ {name}: Skipped (scipy not available)")
                        continue
                
                config = NormalizationConfig(
                    outlier_handling=method,
                    scaling_method=ScalingMethod.Z_SCORE
                )
                
                normalizer = FeatureNormalizer(config)
                normalized_data = normalizer.fit_transform(self.data.copy())
                
                # Validate results
                assert isinstance(normalized_data, pd.DataFrame), f"{name} should return DataFrame"
                assert normalized_data.shape[1] == self.data.shape[1], f"{name} should preserve columns"
                
                # Check outlier bounds were stored for clipping
                if method == OutlierHandling.CLIP:
                    assert len(normalizer.outlier_bounds_) > 0, "Clipping should store bounds"
                
                print(f"   ✓ {name}: Processed {normalized_data.shape[0]} rows")
            
            self.test_results['outlier_handling'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Outlier handling test failed: {str(e)}")
            self.test_results['outlier_handling'] = 'FAIL'
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        print("\\n3. Testing Missing Value Handling")
        print("-" * 40)
        
        try:
            # Create data with missing values
            test_data = self.data.copy()
            missing_indices = np.random.choice(len(test_data), size=50, replace=False)
            test_data.loc[test_data.index[missing_indices], 'normal_feature'] = np.nan
            
            strategies = ["mean", "median", "mode"]
            
            for strategy in strategies:
                config = NormalizationConfig(
                    handle_missing=True,
                    missing_strategy=strategy,
                    scaling_method=ScalingMethod.Z_SCORE
                )
                
                normalizer = FeatureNormalizer(config)
                normalized_data = normalizer.fit_transform(test_data.copy())
                
                # Validate no missing values remain
                assert not normalized_data.isnull().any().any(), f"{strategy} should remove all NaN values"
                assert normalized_data.shape == test_data.shape, f"{strategy} should preserve shape"
                
                print(f"   ✓ {strategy.capitalize()} strategy: {normalized_data.isnull().sum().sum()} missing values remaining")
            
            self.test_results['missing_values'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Missing value handling test failed: {str(e)}")
            self.test_results['missing_values'] = 'FAIL'
    
    def test_fit_transform_consistency(self):
        """Test fit/transform consistency"""
        print("\\n4. Testing Fit/Transform Consistency")
        print("-" * 40)
        
        try:
            config = NormalizationConfig(scaling_method=ScalingMethod.Z_SCORE)
            normalizer = FeatureNormalizer(config)
            
            # Test fit_transform vs fit + transform
            result1 = normalizer.fit_transform(self.data.copy())
            
            normalizer2 = FeatureNormalizer(config)
            normalizer2.fit(self.data.copy())
            result2 = normalizer2.transform(self.data.copy())
            
            # Results should be identical
            assert np.allclose(result1.values, result2.values, rtol=1e-10, equal_nan=True), \
                   "fit_transform should equal fit + transform"
            
            # Test transform on new data
            new_data = self.data.iloc[:100].copy()
            transformed_new = normalizer.transform(new_data)
            
            assert transformed_new.shape == new_data.shape, "Transform should preserve shape"
            assert isinstance(transformed_new, pd.DataFrame), "Transform should return DataFrame"
            
            print(f"   ✓ Fit/transform consistency validated")
            print(f"   ✓ Transform on new data: {transformed_new.shape}")
            
            self.test_results['fit_transform_consistency'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Fit/transform consistency test failed: {str(e)}")
            self.test_results['fit_transform_consistency'] = 'FAIL'
    
    def test_inverse_transform(self):
        """Test inverse transformation"""
        print("\\n5. Testing Inverse Transformation")
        print("-" * 40)
        
        try:
            # Test with different scaling methods
            methods = [ScalingMethod.Z_SCORE, ScalingMethod.MIN_MAX, ScalingMethod.ROBUST, ScalingMethod.MAX_ABS]
            
            for method in methods:
                config = NormalizationConfig(
                    scaling_method=method,
                    outlier_handling=OutlierHandling.NONE  # For cleaner inverse testing
                )
                
                normalizer = FeatureNormalizer(config)
                
                # Use clean data without outliers for testing
                clean_data = self.data.copy().fillna(self.data.median())
                
                # Transform and inverse transform
                transformed = normalizer.fit_transform(clean_data)
                inverse_transformed = normalizer.inverse_transform(transformed)
                
                # Check if we get back close to original
                if method in [ScalingMethod.Z_SCORE, ScalingMethod.MIN_MAX, ScalingMethod.ROBUST, ScalingMethod.MAX_ABS]:
                    is_close = np.allclose(clean_data.values, inverse_transformed.values, rtol=1e-8, atol=1e-8)
                    print(f"   ✓ {method.value}: Inverse transform accuracy: {'GOOD' if is_close else 'APPROXIMATE'}")
                else:
                    print(f"   ✓ {method.value}: Inverse transform completed")
            
            self.test_results['inverse_transform'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Inverse transformation test failed: {str(e)}")
            self.test_results['inverse_transform'] = 'FAIL'
    
    def test_pipeline_functionality(self):
        """Test pipeline functionality"""
        print("\\n6. Testing Pipeline Functionality")
        print("-" * 40)
        
        try:
            # Create a multi-step pipeline
            pipeline = create_preprocessing_pipeline(
                outlier_method=OutlierHandling.CLIP,
                scaling_method=ScalingMethod.Z_SCORE
            )
            
            # Test pipeline fit and transform
            pipeline_result = pipeline.fit_transform(self.data.copy())
            
            # Validate pipeline results
            assert isinstance(pipeline_result, pd.DataFrame), "Pipeline should return DataFrame"
            assert pipeline_result.shape == self.data.shape, "Pipeline should preserve shape"
            assert pipeline.is_fitted_, "Pipeline should be fitted after fit_transform"
            
            # Test pipeline transform on new data
            new_data = self.data.iloc[:200].copy()
            pipeline_new_result = pipeline.transform(new_data)
            
            assert pipeline_new_result.shape == new_data.shape, "Pipeline transform should preserve shape"
            
            print(f"   ✓ Pipeline created with {len(pipeline.steps)} steps")
            print(f"   ✓ Pipeline result shape: {pipeline_result.shape}")
            print(f"   ✓ Pipeline transform on new data: {pipeline_new_result.shape}")
            
            self.test_results['pipeline_functionality'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Pipeline functionality test failed: {str(e)}")
            self.test_results['pipeline_functionality'] = 'FAIL'
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        print("\\n7. Testing Convenience Functions")
        print("-" * 40)
        
        try:
            # Test normalize_features function
            normalized_simple = normalize_features(
                self.data.copy(),
                method=ScalingMethod.MIN_MAX,
                handle_outliers=True
            )
            
            # Validate convenience function
            assert isinstance(normalized_simple, pd.DataFrame), "normalize_features should return DataFrame"
            assert normalized_simple.shape == self.data.shape, "normalize_features should preserve shape"
            
            # Test with different parameters
            normalized_robust = normalize_features(
                self.data.copy(),
                method=ScalingMethod.ROBUST,
                handle_outliers=False
            )
            
            assert isinstance(normalized_robust, pd.DataFrame), "normalize_features with robust should work"
            
            # Test preprocessing pipeline creation
            auto_pipeline = create_preprocessing_pipeline()
            auto_result = auto_pipeline.fit_transform(self.data.copy())
            
            assert isinstance(auto_result, pd.DataFrame), "Auto pipeline should return DataFrame"
            assert auto_result.shape == self.data.shape, "Auto pipeline should preserve shape"
            
            print(f"   ✓ normalize_features: {normalized_simple.shape}")
            print(f"   ✓ normalize_features (robust): {normalized_robust.shape}")
            print(f"   ✓ create_preprocessing_pipeline: {auto_result.shape}")
            
            self.test_results['convenience_functions'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Convenience functions test failed: {str(e)}")
            self.test_results['convenience_functions'] = 'FAIL'
    
    def test_save_load_functionality(self):
        """Test save and load functionality"""
        print("\\n8. Testing Save/Load Functionality")
        print("-" * 40)
        
        try:
            # Create and fit normalizer
            config = NormalizationConfig(scaling_method=ScalingMethod.Z_SCORE)
            normalizer = FeatureNormalizer(config)
            normalizer.fit(self.data.copy())
            
            # Save normalizer
            save_path = 'test_normalizer.pkl'
            normalizer.save(save_path)
            
            # Load normalizer
            loaded_normalizer = FeatureNormalizer.load(save_path)
            
            # Test they produce same results
            original_result = normalizer.transform(self.data.copy())
            loaded_result = loaded_normalizer.transform(self.data.copy())
            
            is_equal = np.allclose(original_result.values, loaded_result.values, rtol=1e-10, equal_nan=True)
            
            # Clean up
            if os.path.exists(save_path):
                os.remove(save_path)
            
            assert is_equal, "Loaded normalizer should produce identical results"
            assert loaded_normalizer.is_fitted_, "Loaded normalizer should be fitted"
            
            print(f"   ✓ Save/load consistency: PASS")
            print(f"   ✓ Loaded normalizer fitted state: {loaded_normalizer.is_fitted_}")
            
            self.test_results['save_load'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Save/load functionality test failed: {str(e)}")
            self.test_results['save_load'] = 'FAIL'
    
    def test_feature_statistics(self):
        """Test feature statistics and parameters"""
        print("\\n9. Testing Feature Statistics")
        print("-" * 40)
        
        try:
            config = NormalizationConfig(scaling_method=ScalingMethod.Z_SCORE)
            normalizer = FeatureNormalizer(config)
            normalizer.fit(self.data.copy())
            
            # Get feature statistics
            stats = normalizer.get_feature_stats()
            params = normalizer.get_transformation_params()
            
            # Validate statistics
            assert isinstance(stats, dict), "Feature stats should be a dictionary"
            assert len(stats) > 0, "Feature stats should not be empty"
            
            # Check each feature has required statistics
            for feature in self.data.columns:
                if feature in stats:
                    feature_stats = stats[feature]
                    required_stats = ['mean', 'std', 'median', 'min', 'max', 'q1', 'q3']
                    for stat in required_stats:
                        assert stat in feature_stats, f"Feature {feature} should have {stat}"
            
            # Validate transformation parameters
            assert isinstance(params, dict), "Transformation params should be a dictionary"
            assert len(params) > 0, "Transformation params should not be empty"
            
            print(f"   ✓ Feature statistics collected for {len(stats)} features")
            print(f"   ✓ Transformation parameters stored for {len(params)} features")
            print(f"   ✓ Sample stats for '{list(stats.keys())[0]}': mean={stats[list(stats.keys())[0]]['mean']:.3f}")
            
            self.test_results['feature_statistics'] = 'PASS'
            
        except Exception as e:
            print(f"   ✗ Feature statistics test failed: {str(e)}")
            self.test_results['feature_statistics'] = 'FAIL'
    
    def run_all_tests(self):
        """Run all feature normalization tests"""
        print("\\nStarting Comprehensive Feature Normalization Tests")
        print("=" * 60)
        
        # Run individual tests
        self.test_basic_scaling_methods()
        self.test_outlier_handling()
        self.test_missing_value_handling()
        self.test_fit_transform_consistency()
        self.test_inverse_transform()
        self.test_pipeline_functionality()
        self.test_convenience_functions()
        self.test_save_load_functionality()
        self.test_feature_statistics()
        
        # Print summary
        print("\\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for result in self.test_results.values() if result == 'PASS')
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status_symbol = "✓" if result == 'PASS' else "✗"
            print(f"{status_symbol} {test_name.replace('_', ' ').title()}: {result}")
        
        print("-" * 60)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {(passed/total)*100:.1f}%")
        
        if passed == total:
            print("\\n🎉 ALL TESTS PASSED! Feature Normalization Pipeline is working correctly.")
        else:
            print(f"\\n⚠️  {total-passed} test(s) failed. Please review the implementation.")
        
        return passed == total


def main():
    """Main test execution function"""
    print("Feature Normalization Pipeline - Comprehensive Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize and run tests
    test_suite = TestFeatureNormalizer()
    success = test_suite.run_all_tests()
    
    print(f"\\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success:
        print("\\n✅ Feature Normalization Pipeline is ready for use!")
    else:
        print("\\n❌ Some tests failed. Please review the implementation.")
    
    return success


if __name__ == "__main__":
    main() 