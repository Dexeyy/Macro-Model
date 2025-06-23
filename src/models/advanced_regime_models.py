"""
Advanced Regime Classification Models

This module implements sophisticated regime classification methods including:
- Hidden Markov Models (HMM)
- Factor Analysis Models
- Regime Probability Forecasting
- Model Comparison and Ensemble Methods
- Integration with existing classification systems

Author: Macro Regime Model Project
Date: 2025-01-22
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ML and statistical libraries
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import minimize

# Warnings and logging
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRegimeModels:
    """
    Comprehensive advanced regime classification models.
    
    This class implements state-of-the-art regime classification methods
    including Hidden Markov Models, Factor Analysis, and ensemble approaches.
    """
    
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        """
        Initialize the advanced regime models.
        
        Parameters:
        -----------
        n_regimes : int
            Number of regimes to identify
        random_state : int
            Random state for reproducibility
        """
        self.n_regimes = n_regimes
        self.random_state = random_state
        
        # Scalers and transformers
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.pca = PCA(n_components=2)
        
        # Models
        self.hmm_model = None
        self.factor_model = None
        self.ensemble_model = None
        
        # Results storage
        self.model_results = {}
        self.comparison_results = {}
        
        logger.info(f"AdvancedRegimeModels initialized with {n_regimes} regimes")
    
    def fit_hmm(self, data: pd.DataFrame, n_regimes: int = None, 
                covariance_type: str = "full", n_iter: int = 1000) -> dict:
        """
        Fit Hidden Markov Model to the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for regime classification
        n_regimes : int, optional
            Number of regimes (overrides default)
        covariance_type : str
            Type of covariance matrix ("full", "diag", "tied", "spherical")
        n_iter : int
            Maximum number of iterations for EM algorithm
            
        Returns:
        --------
        dict : Dictionary containing HMM results
        """
        logger.info("Fitting Hidden Markov Model...")
        
        if n_regimes is not None:
            self.n_regimes = n_regimes
        
        # Prepare data
        data_clean = data.dropna()
        scaled_data = self.scaler.fit_transform(data_clean)
        
        # Initialize and fit HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=self.random_state,
            tol=1e-6
        )
        
        try:
            # Fit the model
            self.hmm_model.fit(scaled_data)
            
            # Get regime sequence
            hidden_states = self.hmm_model.predict(scaled_data)
            
            # Get regime probabilities
            regime_probs = self.hmm_model.predict_proba(scaled_data)
            
            # Create results
            regime_prob_df = pd.DataFrame(
                regime_probs,
                index=data_clean.index,
                columns=[f'Regime_{i}_Prob' for i in range(self.n_regimes)]
            )
            
            # Add regime labels with meaningful names
            regime_names = self._assign_regime_names(hidden_states, data_clean)
            regime_labels = pd.Series(
                [regime_names[state] for state in hidden_states], 
                index=data_clean.index
            )
            
            # Calculate model statistics
            log_likelihood = self.hmm_model.score(scaled_data)
            aic = -2 * log_likelihood + 2 * self._count_parameters()
            bic = -2 * log_likelihood + np.log(len(scaled_data)) * self._count_parameters()
            
            results = {
                'model': self.hmm_model,
                'hidden_states': hidden_states,
                'regime_labels': regime_labels,
                'regime_probabilities': regime_prob_df,
                'transition_matrix': self.hmm_model.transmat_,
                'means': self.scaler.inverse_transform(self.hmm_model.means_),
                'covars': self.hmm_model.covars_,
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'regime_names': regime_names,
                'n_regimes': self.n_regimes
            }
            
            self.model_results['hmm'] = results
            logger.info(f"HMM fitting completed. Log-likelihood: {log_likelihood:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            raise
    
    def _assign_regime_names(self, hidden_states: np.ndarray, data: pd.DataFrame) -> dict:
        """Assign meaningful names to regimes based on characteristics."""
        regime_stats = {}
        
        for regime in range(self.n_regimes):
            mask = hidden_states == regime
            if np.sum(mask) > 0:
                regime_data = data[mask]
                mean_return = regime_data.mean().mean()
                volatility = regime_data.std().mean()
                
                regime_stats[regime] = {
                    'mean_return': mean_return,
                    'volatility': volatility,
                    'count': np.sum(mask)
                }
        
        # Sort regimes by return characteristics
        sorted_regimes = sorted(regime_stats.items(), 
                               key=lambda x: x[1]['mean_return'], reverse=True)
        
        # Assign names based on return and volatility characteristics
        regime_names = {}
        for i, (regime_id, stats) in enumerate(sorted_regimes):
            if stats['mean_return'] > 0.001:  # High positive returns
                if stats['volatility'] > 0.02:
                    regime_names[regime_id] = 'Bull_High_Vol'
                else:
                    regime_names[regime_id] = 'Bull_Low_Vol'
            elif stats['mean_return'] < -0.001:  # Negative returns
                if stats['volatility'] > 0.02:
                    regime_names[regime_id] = 'Bear_High_Vol'
                else:
                    regime_names[regime_id] = 'Bear_Low_Vol'
            else:  # Neutral returns
                if stats['volatility'] > 0.015:
                    regime_names[regime_id] = 'Neutral_High_Vol'
                else:
                    regime_names[regime_id] = 'Neutral_Low_Vol'
        
        return regime_names
    
    def _count_parameters(self) -> int:
        """Count the number of parameters in the HMM model."""
        if self.hmm_model is None:
            return 0
        
        n_components = self.hmm_model.n_components
        n_features = self.hmm_model.n_features
        
        # Transition matrix parameters
        trans_params = n_components * (n_components - 1)
        
        # Start probability parameters
        start_params = n_components - 1
        
        # Emission parameters (means and covariances)
        means_params = n_components * n_features
        
        if self.hmm_model.covariance_type == "full":
            cov_params = n_components * n_features * (n_features + 1) // 2
        elif self.hmm_model.covariance_type == "diag":
            cov_params = n_components * n_features
        elif self.hmm_model.covariance_type == "tied":
            cov_params = n_features * (n_features + 1) // 2
        else:  # spherical
            cov_params = n_components
        
        return trans_params + start_params + means_params + cov_params
    
    def predict_hmm(self, data: pd.DataFrame) -> dict:
        """
        Predict regimes for new data using fitted HMM.
        
        Parameters:
        -----------
        data : pd.DataFrame
            New data for regime prediction
            
        Returns:
        --------
        dict : Dictionary containing predictions
        """
        if self.hmm_model is None:
            raise ValueError("HMM model must be fitted before prediction")
        
        # Prepare data
        data_clean = data.dropna()
        scaled_data = self.scaler.transform(data_clean)
        
        # Predict hidden states
        hidden_states = self.hmm_model.predict(scaled_data)
        
        # Get regime probabilities
        regime_probs = self.hmm_model.predict_proba(scaled_data)
        
        # Create results
        regime_prob_df = pd.DataFrame(
            regime_probs,
            index=data_clean.index,
            columns=[f'Regime_{i}_Prob' for i in range(self.n_regimes)]
        )
        
        # Get regime names from fitted model
        regime_names = self.model_results['hmm']['regime_names']
        regime_labels = pd.Series(
            [regime_names[state] for state in hidden_states], 
            index=data_clean.index
        )
        
        return {
            'hidden_states': hidden_states,
            'regime_labels': regime_labels,
            'regime_probabilities': regime_prob_df
        }
    
    def fit_factor_model(self, data: pd.DataFrame, n_factors: int = 3) -> dict:
        """
        Fit factor analysis model to the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for factor analysis
        n_factors : int
            Number of factors to extract
            
        Returns:
        --------
        dict : Dictionary containing factor analysis results
        """
        logger.info(f"Fitting Factor Analysis model with {n_factors} factors...")
        
        # Prepare data
        data_clean = data.dropna()
        scaled_data = self.scaler.fit_transform(data_clean)
        
        # Initialize and fit Factor Analysis
        self.factor_model = FactorAnalysis(
            n_components=n_factors, 
            random_state=self.random_state,
            max_iter=1000
        )
        
        try:
            # Fit the model and transform data
            factor_loadings = self.factor_model.fit_transform(scaled_data)
            
            # Create DataFrame with factor loadings
            factor_df = pd.DataFrame(
                factor_loadings,
                index=data_clean.index,
                columns=[f'Factor_{i+1}' for i in range(n_factors)]
            )
            
            # Get component matrix (factor loadings for original variables)
            components = pd.DataFrame(
                self.factor_model.components_.T,
                columns=[f'Factor_{i+1}' for i in range(n_factors)],
                index=data_clean.columns
            )
            
            # Calculate explained variance (approximate)
            explained_variance_ratio = self._calculate_factor_variance_explained(
                scaled_data, factor_loadings
            )
            
            # Classify regimes based on factor loadings
            factor_regimes = self._classify_factor_regimes(factor_df)
            
            results = {
                'model': self.factor_model,
                'factor_loadings': factor_df,
                'components': components,
                'explained_variance_ratio': explained_variance_ratio,
                'factor_regimes': factor_regimes,
                'n_factors': n_factors
            }
            
            self.model_results['factor'] = results
            logger.info(f"Factor Analysis completed. Explained variance: {explained_variance_ratio.sum():.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Factor Analysis fitting failed: {e}")
            raise
    
    def _calculate_factor_variance_explained(self, original_data: np.ndarray, 
                                           factor_loadings: np.ndarray) -> np.ndarray:
        """Calculate variance explained by each factor."""
        # For Factor Analysis, we calculate variance explained differently
        # since it doesn't have inverse_transform method
        
        # Calculate total variance of original data
        total_variance = np.var(original_data, axis=0).sum()
        
        # Get the components matrix
        components = self.factor_model.components_
        
        explained_variance = []
        for i in range(factor_loadings.shape[1]):
            # Calculate variance explained by factor i
            # This is approximated by the variance of the factor loadings
            # weighted by the corresponding component loadings
            factor_var = np.var(factor_loadings[:, i])
            component_weights = np.sum(np.abs(components[i, :]))
            
            # Approximate explained variance
            explained_var = (factor_var * component_weights) / total_variance
            explained_variance.append(min(explained_var, 1.0))  # Cap at 100%
        
        # Normalize so total doesn't exceed 100%
        explained_variance = np.array(explained_variance)
        if explained_variance.sum() > 1.0:
            explained_variance = explained_variance / explained_variance.sum()
        
        return explained_variance
    
    def _classify_factor_regimes(self, factor_df: pd.DataFrame) -> pd.Series:
        """Classify regimes based on factor loadings using K-means."""
        # Apply K-means clustering to factor loadings
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.random_state)
        regime_labels = kmeans.fit_predict(factor_df.values)
        
        # Create meaningful regime names based on factor characteristics
        regime_names = {}
        for regime in range(self.n_regimes):
            mask = regime_labels == regime
            if np.sum(mask) > 0:
                factor_means = factor_df[mask].mean()
                dominant_factor = factor_means.abs().idxmax()
                factor_value = factor_means[dominant_factor]
                
                if factor_value > 0.5:
                    regime_names[regime] = f'High_{dominant_factor}'
                elif factor_value < -0.5:
                    regime_names[regime] = f'Low_{dominant_factor}'
                else:
                    regime_names[regime] = f'Moderate_{dominant_factor}'
        
        # Map numeric labels to names
        regime_series = pd.Series(
            [regime_names.get(label, f'Regime_{label}') for label in regime_labels],
            index=factor_df.index
        )
        
        return regime_series
    
    def forecast_regime_probabilities(self, current_probs: np.ndarray, 
                                    steps: int = 10) -> pd.DataFrame:
        """
        Forecast future regime probabilities based on transition matrix.
        
        Parameters:
        -----------
        current_probs : np.ndarray
            Current regime probabilities
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        pd.DataFrame : Forecasted regime probabilities
        """
        if self.hmm_model is None:
            raise ValueError("HMM model must be fitted before forecasting")
        
        logger.info(f"Forecasting regime probabilities for {steps} steps...")
        
        # Get transition matrix
        transition_matrix = self.hmm_model.transmat_
        
        # Initialize forecast array
        forecast = np.zeros((steps + 1, self.n_regimes))
        forecast[0] = current_probs
        
        # Forecast future probabilities
        for i in range(1, steps + 1):
            forecast[i] = np.dot(forecast[i-1], transition_matrix)
        
        # Create DataFrame with regime names
        regime_names = self.model_results['hmm']['regime_names']
        column_names = [regime_names[i] for i in range(self.n_regimes)]
        
        forecast_df = pd.DataFrame(
            forecast,
            columns=column_names,
            index=range(steps + 1)
        )
        
        return forecast_df
    
    def compare_models(self, data: pd.DataFrame, 
                      models: list = None) -> dict:
        """
        Compare different regime classification models.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data for model comparison
        models : list, optional
            List of models to compare
            
        Returns:
        --------
        dict : Comparison results
        """
        if models is None:
            models = ['hmm', 'factor']
        
        logger.info(f"Comparing models: {models}")
        
        comparison_results = {}
        
        # Prepare data
        data_clean = data.dropna()
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for model_name in models:
            if model_name == 'hmm' and 'hmm' in self.model_results:
                # HMM model evaluation
                model_scores = []
                for train_idx, test_idx in tscv.split(data_clean):
                    train_data = data_clean.iloc[train_idx]
                    test_data = data_clean.iloc[test_idx]
                    
                    # Fit model on training data
                    temp_model = hmm.GaussianHMM(
                        n_components=self.n_regimes,
                        random_state=self.random_state
                    )
                    
                    scaled_train = self.scaler.fit_transform(train_data)
                    scaled_test = self.scaler.transform(test_data)
                    
                    temp_model.fit(scaled_train)
                    score = temp_model.score(scaled_test)
                    model_scores.append(score)
                
                comparison_results[model_name] = {
                    'cv_scores': model_scores,
                    'mean_score': np.mean(model_scores),
                    'std_score': np.std(model_scores),
                    'aic': self.model_results['hmm']['aic'],
                    'bic': self.model_results['hmm']['bic']
                }
            
            elif model_name == 'factor' and 'factor' in self.model_results:
                # Factor model evaluation (using reconstruction error)
                model_scores = []
                for train_idx, test_idx in tscv.split(data_clean):
                    train_data = data_clean.iloc[train_idx]
                    test_data = data_clean.iloc[test_idx]
                    
                    # Fit model on training data
                    temp_scaler = StandardScaler()
                    temp_model = FactorAnalysis(
                        n_components=self.model_results['factor']['n_factors'],
                        random_state=self.random_state
                    )
                    
                    scaled_train = temp_scaler.fit_transform(train_data)
                    scaled_test = temp_scaler.transform(test_data)
                    
                    temp_model.fit(scaled_train)
                    
                    # Calculate reconstruction error using factor loadings
                    test_transformed = temp_model.transform(scaled_test)
                    # Approximate reconstruction using components
                    test_reconstructed = test_transformed @ temp_model.components_
                    reconstruction_error = np.mean((scaled_test - test_reconstructed) ** 2)
                    
                    model_scores.append(-reconstruction_error)  # Negative for consistency
                
                comparison_results[model_name] = {
                    'cv_scores': model_scores,
                    'mean_score': np.mean(model_scores),
                    'std_score': np.std(model_scores)
                }
        
        self.comparison_results = comparison_results
        logger.info("Model comparison completed")
        
        return comparison_results
    
    def create_ensemble_model(self, data: pd.DataFrame, 
                            base_models: list = None) -> dict:
        """
        Create an ensemble model combining multiple regime classifiers.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data
        base_models : list, optional
            List of base models to ensemble
            
        Returns:
        --------
        dict : Ensemble model results
        """
        if base_models is None:
            base_models = ['hmm', 'factor']
        
        logger.info(f"Creating ensemble model with: {base_models}")
        
        # Prepare data
        data_clean = data.dropna()
        
        # Get predictions from base models
        ensemble_predictions = {}
        ensemble_probabilities = {}
        
        for model_name in base_models:
            if model_name == 'hmm' and 'hmm' in self.model_results:
                hmm_pred = self.predict_hmm(data_clean)
                ensemble_predictions[model_name] = hmm_pred['regime_labels']
                ensemble_probabilities[model_name] = hmm_pred['regime_probabilities']
            
            elif model_name == 'factor' and 'factor' in self.model_results:
                factor_pred = self.model_results['factor']['factor_regimes']
                ensemble_predictions[model_name] = factor_pred
        
        # Create ensemble predictions using voting
        if len(ensemble_predictions) > 1:
            # Simple majority voting for regime labels
            prediction_df = pd.DataFrame(ensemble_predictions)
            
            # For each time point, find the most common prediction
            ensemble_labels = prediction_df.mode(axis=1)[0]
            
            # Average probabilities if available
            if ensemble_probabilities:
                avg_probabilities = None
                for model_probs in ensemble_probabilities.values():
                    if avg_probabilities is None:
                        avg_probabilities = model_probs.copy()
                    else:
                        avg_probabilities += model_probs
                
                avg_probabilities = avg_probabilities / len(ensemble_probabilities)
            else:
                avg_probabilities = None
            
            ensemble_results = {
                'ensemble_labels': ensemble_labels,
                'ensemble_probabilities': avg_probabilities,
                'base_predictions': ensemble_predictions,
                'base_models': base_models
            }
            
            self.model_results['ensemble'] = ensemble_results
            logger.info("Ensemble model created successfully")
            
            return ensemble_results
        
        else:
            logger.warning("Need at least 2 base models for ensemble")
            return {}
    
    def integrate_with_existing_methods(self, existing_classifier) -> dict:
        """
        Integrate advanced models with existing classification methods.
        
        Parameters:
        -----------
        existing_classifier : object
            Existing regime classifier with classify_regime method
            
        Returns:
        --------
        dict : Integration results
        """
        logger.info("Integrating with existing classification methods...")
        
        # This method provides a bridge between advanced models and existing ones
        integration_results = {
            'existing_classifier': existing_classifier,
            'advanced_models': list(self.model_results.keys()),
            'integration_strategy': 'ensemble_voting'
        }
        
        # Create a wrapper class that combines predictions
        class IntegratedClassifier:
            def __init__(self, advanced_models, existing_classifier):
                self.advanced_models = advanced_models
                self.existing_classifier = existing_classifier
            
            def classify_regime(self, data):
                """Classify regime using integrated approach."""
                predictions = []
                
                # Get prediction from existing classifier
                if hasattr(self.existing_classifier, 'classify_regime'):
                    existing_pred = self.existing_classifier.classify_regime(data)
                    predictions.append(existing_pred)
                
                # Get predictions from advanced models
                if 'hmm' in self.advanced_models.model_results:
                    hmm_pred = self.advanced_models.predict_hmm(data)
                    # Convert to simple regime name
                    most_likely = hmm_pred['regime_probabilities'].idxmax(axis=1).iloc[-1]
                    predictions.append(most_likely.split('_')[0])  # Simplified name
                
                # Use majority vote or return most confident prediction
                if predictions:
                    return max(set(predictions), key=predictions.count)
                else:
                    return 'Unknown'
        
        integrated_classifier = IntegratedClassifier(self, existing_classifier)
        integration_results['integrated_classifier'] = integrated_classifier
        
        logger.info("Integration completed")
        return integration_results 