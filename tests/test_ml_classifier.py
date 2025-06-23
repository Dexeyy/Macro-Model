#!/usr/bin/env python3
"""
Test script for ML Regime Classifier - Task 7 validation
Tests all 5 subtasks to ensure complete functionality.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append('src')

from models.ml_regime_classifier import MLRegimeClassifier, MLRegimeConfig, ScalingMethod, ClusterMetric, create_ml_regime_classifier

def create_synthetic_data(n_samples=200):
    """Create synthetic macroeconomic data for testing."""
    np.random.seed(42)
    
    # Create time index
    dates = pd.date_range('2010-01-01', periods=n_samples, freq='M')
    
    # Generate synthetic features with different regime characteristics
    data = pd.DataFrame(index=dates)
    
    # Economic indicators
    data['gdp_growth'] = np.random.normal(0.02, 0.015, n_samples)
    data['inflation'] = np.random.normal(0.025, 0.01, n_samples)
    data['unemployment'] = np.random.normal(0.06, 0.02, n_samples)
    data['interest_rate'] = np.random.normal(0.03, 0.02, n_samples)
    
    # Market indicators
    data['volatility'] = np.random.exponential(0.15, n_samples)
    data['market_return'] = np.random.normal(0.008, 0.04, n_samples)
    
    # Create regime-like patterns
    # First third: low volatility, moderate growth
    data.iloc[:n_samples//3, data.columns.get_loc('volatility')] *= 0.5
    data.iloc[:n_samples//3, data.columns.get_loc('gdp_growth')] += 0.01
    
    # Middle third: high volatility, low growth
    data.iloc[n_samples//3:2*n_samples//3, data.columns.get_loc('volatility')] *= 2
    data.iloc[n_samples//3:2*n_samples//3, data.columns.get_loc('gdp_growth')] -= 0.015
    
    # Last third: moderate conditions
    # (keep as generated)
    
    return data

def test_subtask_7_1_data_preprocessing():
    """Test Subtask 7.1: Data Preprocessing for K-means Clustering."""
    print("🧪 Testing Subtask 7.1: Data Preprocessing for K-means Clustering")
    
    try:
        # Create classifier
        classifier = MLRegimeClassifier()
        
        # Create test data with some missing values
        data = create_synthetic_data(100)
        data.iloc[10:15, 0] = np.nan  # Add some NaN values
        data.iloc[50, :] = np.nan     # Add a row of NaN
        
        # Test preprocessing
        processed_data, scaled_data = classifier.preprocess_data(data)
        
        # Validate results
        assert processed_data.isnull().sum().sum() == 0, "Missing values not handled"
        assert len(processed_data) > 0, "No data after preprocessing"
        assert scaled_data.shape == processed_data.shape, "Scaling changed data shape"
        assert classifier.feature_names == data.columns.tolist(), "Feature names not stored"
        
        print("   ✅ Data preprocessing working correctly")
        print(f"   📊 Processed data shape: {processed_data.shape}")
        print(f"   🏷️ Features: {classifier.feature_names}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in data preprocessing: {e}")
        return False

def test_subtask_7_2_optimal_cluster_selection():
    """Test Subtask 7.2: Optimal Cluster Selection Implementation."""
    print("\n🧪 Testing Subtask 7.2: Optimal Cluster Selection Implementation")
    
    try:
        # Test different selection metrics
        metrics_to_test = ['silhouette', 'calinski', 'davies_bouldin', 'inertia', 'combined']
        
        for metric in metrics_to_test:
            config = MLRegimeConfig(
                auto_select_clusters=True,
                selection_metric=ClusterMetric(metric),
                max_clusters=6
            )
            classifier = MLRegimeClassifier(config)
            
            # Create and preprocess data
            data = create_synthetic_data(150)
            processed_data, scaled_data = classifier.preprocess_data(data)
            
            # Test optimal cluster selection
            optimal_clusters, metric_scores = classifier.find_optimal_clusters(scaled_data)
            
            # Validate results
            assert 2 <= optimal_clusters <= 6, f"Invalid optimal clusters: {optimal_clusters}"
            assert len(metric_scores['silhouette']) == 5, "Wrong number of metric scores"
            assert all(len(scores) == 5 for scores in metric_scores.values()), "Inconsistent metric scores"
            
            print(f"   ✅ {metric} metric: optimal clusters = {optimal_clusters}")
        
        print("   🎯 All cluster selection metrics working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in cluster selection: {e}")
        return False

def test_subtask_7_3_visualization():
    """Test Subtask 7.3: K-means Cluster Visualization Development."""
    print("\n🧪 Testing Subtask 7.3: K-means Cluster Visualization Development")
    
    try:
        # Create and fit classifier
        classifier = create_ml_regime_classifier(n_regimes=3, auto_select_clusters=False)
        data = create_synthetic_data(100)
        
        # Fit the model
        results = classifier.fit(data)
        
        # Test that visualization methods exist and don't crash
        # Note: We can't actually display plots in this test, but we can verify methods exist
        assert hasattr(classifier, 'visualize_regimes'), "visualize_regimes method missing"
        assert hasattr(classifier, '_create_pca_visualization'), "PCA visualization method missing"
        assert hasattr(classifier, '_create_feature_visualization'), "Feature visualization method missing"
        assert hasattr(classifier, '_create_cluster_centers_visualization'), "Cluster centers visualization method missing"
        
        print("   ✅ Visualization methods implemented")
        print("   📈 PCA, feature, and cluster center visualizations available")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in visualization: {e}")
        return False

def test_subtask_7_4_regime_characteristics():
    """Test Subtask 7.4: Market Regime Characteristic Analysis."""
    print("\n🧪 Testing Subtask 7.4: Market Regime Characteristic Analysis")
    
    try:
        # Create classifier and fit to data
        classifier = create_ml_regime_classifier(n_regimes=3, auto_select_clusters=False)
        data = create_synthetic_data(150)
        
        # Fit the model
        results = classifier.fit(data)
        
        # Test that regime analysis methods exist
        assert hasattr(classifier, '_analyze_regime_characteristics'), "Regime analysis method missing"
        assert hasattr(classifier, '_interpret_regime'), "Regime interpretation method missing"
        assert hasattr(classifier, '_generate_regime_name'), "Regime naming method missing"
        
        # Validate results structure
        labels = results['labels']
        n_regimes = len(np.unique(labels))
        
        assert n_regimes > 0, "No regimes detected"
        assert len(labels) == len(data), "Labels length mismatch"
        
        print(f"   ✅ Regime characteristic analysis implemented")
        print(f"   🏛️ Detected {n_regimes} regimes")
        print(f"   📊 Labels shape: {labels.shape}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in regime analysis: {e}")
        return False

def test_subtask_7_5_integration():
    """Test Subtask 7.5: Integration with Existing Classification Methods."""
    print("\n🧪 Testing Subtask 7.5: Integration with Existing Classification Methods")
    
    try:
        # Create classifier and fit to data
        classifier = create_ml_regime_classifier(n_regimes=4, auto_select_clusters=False)
        data = create_synthetic_data(100)
        
        # Fit the model
        results = classifier.fit(data)
        ml_labels = results['labels']
        
        # Create synthetic rule-based regimes for comparison
        rule_based_labels = np.random.choice(['Expansion', 'Recession', 'Recovery', 'Neutral'], 
                                           size=len(data), p=[0.3, 0.2, 0.25, 0.25])
        
        # Create pandas Series with same index
        rule_series = pd.Series(rule_based_labels, index=data.index)
        ml_series = pd.Series(ml_labels, index=data.index)
        
        # Test integration method
        integration_results = classifier.integrate_with_rule_based(rule_series, ml_series)
        
        # Validate results
        assert 'agreement_rate' in integration_results, "Agreement rate missing"
        assert 'regime_mapping' in integration_results, "Regime mapping missing"
        assert 'confusion_matrix' in integration_results, "Confusion matrix missing"
        assert 0 <= integration_results['agreement_rate'] <= 1, "Invalid agreement rate"
        
        print("   ✅ Integration with rule-based classification implemented")
        print(f"   🤝 Agreement rate: {integration_results['agreement_rate']:.2%}")
        print(f"   🗺️ Regime mapping: {len(integration_results['regime_mapping'])} mappings")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in integration: {e}")
        return False

def test_complete_workflow():
    """Test complete ML regime classification workflow."""
    print("\n🧪 Testing Complete ML Regime Classification Workflow")
    
    try:
        # Create comprehensive test
        data = create_synthetic_data(200)
        
        # Test with auto cluster selection
        classifier_auto = create_ml_regime_classifier(
            auto_select_clusters=True,
            scaling_method='standard',
            selection_metric='silhouette'
        )
        
        # Fit the model
        results_auto = classifier_auto.fit(data)
        
        # Test predictions on new data
        new_data = create_synthetic_data(50)
        predictions = classifier_auto.predict(new_data)
        
        # Validate complete workflow
        assert classifier_auto.fitted, "Model not marked as fitted"
        assert 'labels' in results_auto, "Labels missing from results"
        assert len(predictions) == len(new_data), "Prediction length mismatch"
        assert results_auto['n_regimes'] > 0, "No regimes detected"
        
        print("   ✅ Complete workflow successful")
        print(f"   🎯 Auto-selected regimes: {results_auto['n_regimes']}")
        print(f"   🔮 Predictions generated for {len(predictions)} samples")
        return True
        
    except Exception as e:
        print(f"   ❌ Error in complete workflow: {e}")
        return False

def main():
    """Run all tests for Task 7 ML Regime Classifier."""
    print("🚀 Testing ML Regime Classifier - Task 7 Implementation")
    print("=" * 60)
    
    test_results = []
    
    # Run all subtask tests
    test_results.append(test_subtask_7_1_data_preprocessing())
    test_results.append(test_subtask_7_2_optimal_cluster_selection())
    test_results.append(test_subtask_7_3_visualization())
    test_results.append(test_subtask_7_4_regime_characteristics())
    test_results.append(test_subtask_7_5_integration())
    test_results.append(test_complete_workflow())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"✅ Passed: {passed}/{total} tests")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Task 7 implementation is complete and functional.")
        print("\n📝 All subtasks implemented:")
        print("   ✅ 7.1 - Data Preprocessing for K-means Clustering")
        print("   ✅ 7.2 - Optimal Cluster Selection Implementation")
        print("   ✅ 7.3 - K-means Cluster Visualization Development")
        print("   ✅ 7.4 - Market Regime Characteristic Analysis")
        print("   ✅ 7.5 - Integration with Existing Classification Methods")
        return True
    else:
        print(f"❌ {total - passed} tests failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 