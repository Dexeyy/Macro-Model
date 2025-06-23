"""
Portfolio Construction Module

This module provides comprehensive portfolio construction capabilities for regime-based
investment strategies. It includes functionality for calculating regime-specific statistics,
optimizing portfolios using various algorithms, handling constraints, and evaluating
performance metrics.

Key Features:
- Regime-specific portfolio statistics calculation
- Multiple optimization algorithms (Sharpe ratio, minimum variance, maximum return, risk parity)
- Flexible constraint handling (position limits, sector exposures)
- Comprehensive performance metrics
- Integration with regime classification system
"""

import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress optimization warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Supported portfolio optimization methods"""
    SHARPE = "sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"

@dataclass
class PortfolioConstraints:
    """Configuration for portfolio constraints"""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    max_positions: Optional[int] = None  # Maximum number of positions
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None  # Sector min/max exposures
    turnover_limit: Optional[float] = None  # Maximum portfolio turnover
    leverage_limit: float = 1.0  # Maximum leverage (1.0 = no leverage)

@dataclass
class PortfolioResult:
    """Results from portfolio optimization"""
    weights: pd.Series
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_success: bool
    method_used: str
    regime: Optional[str] = None
    constraints_satisfied: bool = True

class PortfolioConstructor:
    """
    Advanced portfolio construction system for regime-based investment strategies.
    
    This class provides comprehensive portfolio optimization capabilities including:
    - Regime-specific statistics calculation
    - Multiple optimization algorithms
    - Flexible constraint handling
    - Performance evaluation
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 constraints: Optional[PortfolioConstraints] = None):
        """
        Initialize the portfolio constructor.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculations
            constraints: Portfolio constraints configuration
        """
        self.risk_free_rate = risk_free_rate
        self.constraints = constraints or PortfolioConstraints()
        logger.info("Initialized PortfolioConstructor")
    
    def calculate_regime_statistics(self, 
                                  returns: pd.DataFrame, 
                                  regimes: pd.Series) -> Dict[str, Dict]:
        """
        Calculate comprehensive statistics for each market regime.
        
        This method implements Subtask 5.1: Calculate Regime Statistics
        
        Args:
            returns: DataFrame with asset returns (dates x assets)
            regimes: Series with regime classifications for each date
            
        Returns:
            Dictionary with regime statistics including:
            - Mean returns and covariance matrices
            - Risk-adjusted metrics (Sharpe ratios)
            - Distribution statistics (skewness, kurtosis)
            - Regime frequency and count
        """
        try:
            # Ensure index alignment
            common_dates = returns.index.intersection(regimes.index)
            if len(common_dates) == 0:
                raise ValueError("No common dates between returns and regimes")
            
            returns_aligned = returns.loc[common_dates]
            regimes_aligned = regimes.loc[common_dates]
            
            regime_stats = {}
            unique_regimes = regimes_aligned.unique()
            
            logger.info(f"Calculating statistics for {len(unique_regimes)} regimes")
            
            for regime in unique_regimes:
                if pd.isna(regime):
                    continue
                
                # Filter data for this regime
                regime_mask = regimes_aligned == regime
                regime_returns = returns_aligned[regime_mask]
                
                if len(regime_returns) < 2:
                    logger.warning(f"Insufficient data for regime {regime} ({len(regime_returns)} observations)")
                    continue
                
                # Calculate basic statistics
                mean_returns = regime_returns.mean()
                covariance = regime_returns.cov()
                
                # Handle singular covariance matrices for numerical stability
                try:
                    # Add small amount to diagonal for numerical stability
                    covariance_stable = covariance + np.eye(len(covariance)) * 1e-8
                    np.linalg.cholesky(covariance_stable)  # Test for positive definiteness
                    covariance = covariance_stable
                except np.linalg.LinAlgError:
                    logger.warning(f"Singular covariance matrix for regime {regime}, using diagonal approximation")
                    covariance = np.diag(np.diag(covariance)) + np.eye(len(covariance)) * 1e-6
                
                # Calculate additional metrics
                volatility = regime_returns.std()
                sharpe_ratios = (mean_returns * 252 - self.risk_free_rate) / (volatility * np.sqrt(252))
                
                # Correlation matrix
                correlation = regime_returns.corr()
                
                # Regime-specific comprehensive statistics
                regime_stats[regime] = {
                    'count': len(regime_returns),
                    'mean_returns': mean_returns,
                    'annualized_returns': mean_returns * 252,
                    'volatility': volatility,
                    'covariance': covariance,
                    'correlation': correlation,
                    'sharpe_ratios': sharpe_ratios,
                    'max_return': regime_returns.max(),
                    'min_return': regime_returns.min(),
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'frequency': len(regime_returns) / len(returns_aligned)
                }
                
                logger.info(f"Regime {regime}: {len(regime_returns)} observations, "
                          f"avg return: {mean_returns.mean():.4f}")
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"Error calculating regime statistics: {e}")
            raise
    
    def _portfolio_variance(self, weights: np.ndarray, covariance_matrix: pd.DataFrame) -> float:
        """Calculate portfolio variance"""
        return np.dot(weights.T, np.dot(covariance_matrix.values, weights))
    
    def _portfolio_return(self, weights: np.ndarray, returns: pd.Series) -> float:
        """Calculate portfolio expected return"""
        return np.sum(returns.values * weights)
    
    def _negative_sharpe_ratio(self, 
                             weights: np.ndarray, 
                             returns: pd.Series, 
                             covariance: pd.DataFrame,
                             risk_free_rate: float = None) -> float:
        """Calculate negative Sharpe ratio (for minimization)"""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate
        
        portfolio_return = self._portfolio_return(weights, returns) * 252  # Annualize
        portfolio_volatility = np.sqrt(self._portfolio_variance(weights, covariance)) * np.sqrt(252)
        
        if portfolio_volatility == 0:
            return -np.inf if portfolio_return > risk_free_rate else 0
        
        return -(portfolio_return - risk_free_rate) / portfolio_volatility
    
    def _risk_parity_objective(self, weights: np.ndarray, covariance: pd.DataFrame) -> float:
        """Risk parity objective function (minimize sum of squared risk contributions)"""
        portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance))
        if portfolio_vol == 0:
            return 1e6
        marginal_contrib = np.dot(covariance.values, weights) / portfolio_vol
        contrib = weights * marginal_contrib
        target_contrib = portfolio_vol / len(weights)  # Equal risk contribution
        return np.sum((contrib - target_contrib) ** 2)
    
    def _build_constraints(self, n_assets: int, method: OptimizationMethod) -> List[Dict]:
        """
        Build optimization constraints.
        
        This method implements part of Subtask 5.3: Handle Portfolio Constraints
        """
        constraints = []
        
        # Weights must sum to 1 (or leverage limit)
        constraints.append({
            'type': 'eq',
            'fun': lambda x: np.sum(x) - self.constraints.leverage_limit
        })
        
        return constraints
    
    def _build_bounds(self, n_assets: int) -> List[Tuple[float, float]]:
        """Build weight bounds for optimization"""
        return [(self.constraints.min_weight, self.constraints.max_weight) 
                for _ in range(n_assets)]
    
    def optimize_portfolio(self, 
                         regime_stats: Dict[str, Dict], 
                         regime: str,
                         method: OptimizationMethod = OptimizationMethod.SHARPE,
                         custom_constraints: Optional[List[Dict]] = None) -> PortfolioResult:
        """
        Optimize portfolio for a specific regime using specified method.
        
        This method implements Subtask 5.2: Implement Optimization Algorithms
        
        Args:
            regime_stats: Dictionary containing regime statistics
            regime: Regime name to optimize for
            method: Optimization method to use
            custom_constraints: Additional constraints for optimization
            
        Returns:
            PortfolioResult with optimization results
        """
        try:
            if regime not in regime_stats:
                raise ValueError(f"Regime '{regime}' not found in statistics")
            
            stats = regime_stats[regime]
            returns = stats['mean_returns']
            covariance = stats['covariance']
            n_assets = len(returns)
            
            # Handle equal weight case
            if method == OptimizationMethod.EQUAL_WEIGHT:
                weights = np.ones(n_assets) / n_assets
                return self._create_portfolio_result(weights, returns, covariance, method.value, regime)
            
            # Initial guess - equal weights
            initial_weights = np.ones(n_assets) / n_assets
            
            # Build constraints and bounds
            constraints = self._build_constraints(n_assets, method)
            if custom_constraints:
                constraints.extend(custom_constraints)
            
            bounds = self._build_bounds(n_assets)
            
            # Define objective function based on method
            if method == OptimizationMethod.SHARPE:
                objective_func = lambda w: self._negative_sharpe_ratio(w, returns, covariance)
            elif method == OptimizationMethod.MIN_VARIANCE:
                objective_func = lambda w: self._portfolio_variance(w, covariance)
            elif method == OptimizationMethod.MAX_RETURN:
                objective_func = lambda w: -self._portfolio_return(w, returns)
            elif method == OptimizationMethod.RISK_PARITY:
                objective_func = lambda w: self._risk_parity_objective(w, covariance)
            else:
                raise ValueError(f"Unsupported optimization method: {method}")
            
            # Perform optimization
            result = minimize(
                objective_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'disp': False}
            )
            
            # Handle optimization failure
            if not result.success:
                logger.warning(f"Optimization failed for regime {regime} with method {method.value}: {result.message}")
                logger.warning("Falling back to equal weights")
                weights = initial_weights
                success = False
            else:
                weights = result.x
                success = True
            
            return self._create_portfolio_result(weights, returns, covariance, method.value, regime, success)
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio for regime {regime}: {e}")
            # Fallback to equal weights
            n_assets = len(regime_stats[regime]['mean_returns'])
            weights = np.ones(n_assets) / n_assets
            return self._create_portfolio_result(
                weights, 
                regime_stats[regime]['mean_returns'], 
                regime_stats[regime]['covariance'], 
                method.value, 
                regime, 
                False
            )
    
    def _create_portfolio_result(self, 
                               weights: np.ndarray, 
                               returns: pd.Series, 
                               covariance: pd.DataFrame,
                               method: str,
                               regime: str,
                               success: bool = True) -> PortfolioResult:
        """Create a PortfolioResult object from optimization results"""
        weights_series = pd.Series(weights, index=returns.index)
        
        expected_return = self._portfolio_return(weights, returns) * 252  # Annualized
        expected_volatility = np.sqrt(self._portfolio_variance(weights, covariance)) * np.sqrt(252)
        
        sharpe_ratio = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        return PortfolioResult(
            weights=weights_series,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_success=success,
            method_used=method,
            regime=regime,
            constraints_satisfied=self._check_constraints(weights_series)
        )
    
    def _check_constraints(self, weights: pd.Series) -> bool:
        """
        Check if portfolio satisfies constraints.
        
        This method implements part of Subtask 5.3: Handle Portfolio Constraints
        """
        try:
            # Check weight bounds
            if (weights < self.constraints.min_weight - 1e-6).any() or (weights > self.constraints.max_weight + 1e-6).any():
                return False
            
            # Check leverage constraint
            if abs(weights.sum() - self.constraints.leverage_limit) > 1e-6:
                return False
            
            # Check maximum positions
            if self.constraints.max_positions:
                non_zero_positions = (weights > 1e-6).sum()
                if non_zero_positions > self.constraints.max_positions:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking constraints: {e}")
            return False
    
    def create_regime_portfolios(self, 
                               returns: pd.DataFrame, 
                               regimes: pd.Series,
                               methods: Optional[List[OptimizationMethod]] = None) -> Dict[str, Dict[str, PortfolioResult]]:
        """
        Create optimized portfolios for all regimes using multiple methods.
        
        This method implements Subtask 5.4: Create Regime-Specific Portfolios
        
        Args:
            returns: DataFrame with asset returns
            regimes: Series with regime classifications
            methods: List of optimization methods to use
            
        Returns:
            Dictionary with portfolios for each regime and method
        """
        try:
            if methods is None:
                methods = [OptimizationMethod.SHARPE, OptimizationMethod.MIN_VARIANCE, 
                          OptimizationMethod.RISK_PARITY]
            
            # Calculate regime statistics
            logger.info("Calculating regime statistics...")
            regime_stats = self.calculate_regime_statistics(returns, regimes)
            
            # Create portfolios for each regime and method
            portfolios = {}
            total_combinations = len(regime_stats) * len(methods)
            current_combination = 0
            
            for regime in regime_stats.keys():
                portfolios[regime] = {}
                
                for method in methods:
                    current_combination += 1
                    logger.info(f"Optimizing {method.value} portfolio for regime {regime} "
                              f"({current_combination}/{total_combinations})")
                    
                    portfolio = self.optimize_portfolio(regime_stats, regime, method)
                    portfolios[regime][method.value] = portfolio
            
            logger.info(f"Successfully created {total_combinations} portfolios")
            return portfolios
            
        except Exception as e:
            logger.error(f"Error creating regime portfolios: {e}")
            raise
    
    def calculate_portfolio_performance(self, 
                                      weights: pd.Series, 
                                      returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics for a portfolio.
        
        This method implements Subtask 5.5: Develop Performance Metrics Calculation
        
        Args:
            weights: Portfolio weights
            returns: Historical returns DataFrame
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = returns.dot(weights)
            
            # Basic return metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            annualized_volatility = portfolio_returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / running_max) - 1
            max_drawdown = drawdowns.min()
            
            # Win/Loss analysis
            win_rate = (portfolio_returns > 0).mean()
            loss_rate = (portfolio_returns < 0).mean()
            avg_win = portfolio_returns[portfolio_returns > 0].mean() if (portfolio_returns > 0).any() else 0
            avg_loss = portfolio_returns[portfolio_returns < 0].mean() if (portfolio_returns < 0).any() else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
            
            # Downside risk metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Value at Risk (VaR) and Conditional VaR (CVaR) at 95% confidence level
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if (portfolio_returns <= var_95).any() else var_95
            
            # Distribution statistics
            skewness = portfolio_returns.skew()
            kurtosis = portfolio_returns.kurtosis()
            
            # Additional performance metrics
            best_month = portfolio_returns.max()
            worst_month = portfolio_returns.min()
            positive_months = (portfolio_returns > 0).sum()
            negative_months = (portfolio_returns < 0).sum()
            flat_months = (portfolio_returns == 0).sum()
            
            # Calmar ratio (annualized return / max drawdown)
            calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown != 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'profit_factor': profit_factor,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'best_month': best_month,
                'worst_month': worst_month,
                'positive_months': positive_months,
                'negative_months': negative_months,
                'flat_months': flat_months,
                'total_observations': len(portfolio_returns)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {}

# Legacy functions for backward compatibility
def create_equal_weight_portfolio(returns_df):
    """Create an equal-weight portfolio (legacy function)"""
    try:
        n_assets = returns_df.shape[1]
        weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = returns_df.dot(weights)
        
        logger.info(f"Created equal-weight portfolio with {n_assets} assets")
        return portfolio_returns
    
    except Exception as e:
        logger.error(f"Error creating equal-weight portfolio: {e}")
        return None

def calculate_portfolio_metrics(returns_series, risk_free_rate=0.02):
    """Calculate portfolio metrics (legacy function)"""
    try:
        constructor = PortfolioConstructor(risk_free_rate=risk_free_rate)
        weights = pd.Series([1.0], index=['portfolio'])
        returns_df = pd.DataFrame({'portfolio': returns_series})
        return constructor.calculate_portfolio_performance(weights, returns_df)
    except Exception as e:
        logger.error(f"Error in legacy calculate_portfolio_metrics: {e}")
        return {}

def optimize_portfolio_weights(returns_df, objective='sharpe', constraints=None, bounds=None):
    """Optimize portfolio weights (legacy function)"""
    try:
        constructor = PortfolioConstructor()
        
        # Convert objective to enum
        method_map = {
            'sharpe': OptimizationMethod.SHARPE,
            'min_var': OptimizationMethod.MIN_VARIANCE,
            'max_return': OptimizationMethod.MAX_RETURN
        }
        method = method_map.get(objective, OptimizationMethod.SHARPE)
        
        # Create dummy regime data
        regimes = pd.Series(['single_regime'] * len(returns_df), index=returns_df.index)
        regime_stats = constructor.calculate_regime_statistics(returns_df, regimes)
        
        result = constructor.optimize_portfolio(regime_stats, 'single_regime', method)
        return result.weights.values
    except Exception as e:
        logger.error(f"Error in legacy optimize_portfolio_weights: {e}")
        # Return equal weights as fallback
        n_assets = returns_df.shape[1]
        return np.ones(n_assets) / n_assets

def create_regime_based_portfolio(returns_df, regime_data, regime_performance):
    """Create regime-based portfolio (legacy function)"""
    try:
        constructor = PortfolioConstructor()
        
        # Extract regime classifications
        regimes = regime_data.iloc[:, 0] if isinstance(regime_data, pd.DataFrame) else regime_data
        
        # Create regime portfolios using the new system
        portfolios = constructor.create_regime_portfolios(returns_df, regimes, 
                                                        [OptimizationMethod.SHARPE])
        
        # Build result DataFrame similar to original function
        common_dates = returns_df.index.intersection(regimes.index)
        portfolio_df = pd.DataFrame(index=common_dates)
        portfolio_df['Regime'] = regimes.loc[common_dates]
        
        # Apply regime-specific weights
        portfolio_df['portfolio_return'] = 0.0
        
        for date in portfolio_df.index:
            regime = portfolio_df.loc[date, 'Regime']
            if regime in portfolios and 'sharpe' in portfolios[regime]:
                weights = portfolios[regime]['sharpe'].weights
                portfolio_df.loc[date, 'portfolio_return'] = returns_df.loc[date].dot(weights)
            else:
                # Equal weights fallback
                portfolio_df.loc[date, 'portfolio_return'] = returns_df.loc[date].mean()
        
        return portfolio_df
        
    except Exception as e:
        logger.error(f"Error in legacy create_regime_based_portfolio: {e}")
        return None
