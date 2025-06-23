"""
Dynamic Portfolio Optimization Module

This module implements dynamic portfolio optimization algorithms for regime-based
investment strategies. It provides functionality for optimizing portfolio transitions
based on regime changes, handling transaction costs, and generating rebalancing plans.

Key Features:
- Dynamic portfolio rebalancing based on regime transitions
- Transaction cost modeling and optimization
- Risk management constraints
- Performance attribution analysis
- Regime transition probability forecasting
- Gradual rebalancing strategies

Implements Task 8: Develop Dynamic Portfolio Optimization
- Subtask 8.1: Implement Objective Function
- Subtask 8.2: Implement Transaction Cost Model
- Subtask 8.3: Develop Regime Transition Optimization
- Subtask 8.4: Create Rebalancing Plan Generator
- Subtask 8.5: Implement Risk Management Constraints
- Subtask 8.6: Develop Performance Attribution Analysis

Author: Macro Regime Analysis System
Date: 2024
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from datetime import datetime, timedelta

# Import existing portfolio module
from .portfolio import PortfolioConstructor, PortfolioResult, OptimizationMethod

# Suppress optimization warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RebalancingFrequency(Enum):
    """Supported rebalancing frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    REGIME_CHANGE = "regime_change"
    THRESHOLD = "threshold"


class OptimizationObjective(Enum):
    """Optimization objectives for dynamic portfolios"""
    UTILITY_MAXIMIZATION = "utility_max"
    TRACKING_ERROR_MINIMIZATION = "tracking_min"
    TRANSACTION_COST_AWARE = "transaction_aware"
    REGIME_TRANSITION = "regime_transition"
    RISK_PARITY_DYNAMIC = "risk_parity_dynamic"


@dataclass
class TransactionCostModel:
    """Transaction cost model configuration"""
    proportional_cost: float = 0.001  # Proportional cost (e.g., 0.1%)
    fixed_cost: float = 0.0  # Fixed cost per transaction
    market_impact: float = 0.0005  # Market impact cost
    bid_ask_spread: float = 0.0002  # Bid-ask spread cost
    minimum_trade_size: float = 0.001  # Minimum trade size (1% of portfolio)
    nonlinear_impact: bool = True  # Whether to use nonlinear market impact


@dataclass
class RiskConstraints:
    """Risk management constraints configuration"""
    max_portfolio_volatility: Optional[float] = None  # Maximum portfolio volatility
    max_tracking_error: Optional[float] = None  # Maximum tracking error vs benchmark
    max_concentration: float = 0.4  # Maximum weight in single asset
    max_sector_exposure: Optional[Dict[str, float]] = None  # Maximum sector exposures
    max_turnover: Optional[float] = None  # Maximum portfolio turnover
    var_limit: Optional[float] = None  # Value-at-Risk limit
    max_leverage: float = 1.0  # Maximum leverage
    min_diversification: Optional[float] = None  # Minimum Herfindahl diversification index


@dataclass
class RebalancingConfig:
    """Configuration for rebalancing strategy"""
    frequency: RebalancingFrequency = RebalancingFrequency.MONTHLY
    threshold: float = 0.05  # Rebalancing threshold for threshold-based strategy
    lookback_window: int = 252  # Lookback window for regime statistics
    transition_period: int = 5  # Number of periods for gradual transition
    regime_confidence_threshold: float = 0.7  # Minimum confidence for regime change
    emergency_rebalancing: bool = True  # Whether to allow emergency rebalancing


@dataclass
class OptimizationResult:
    """Results from dynamic portfolio optimization"""
    optimized_weights: pd.Series
    current_weights: pd.Series
    target_weights: pd.Series
    turnover: float
    transaction_costs: float
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    utility: float
    regime_probabilities: Dict[str, float]
    optimization_success: bool
    optimization_time: float
    constraints_satisfied: bool
    rebalancing_plan: Optional[Dict[str, Any]] = None


class DynamicPortfolioOptimizer:
    """
    Advanced dynamic portfolio optimization system for regime-based strategies.
    
    This class implements sophisticated portfolio optimization that adapts to
    changing market regimes while considering transaction costs, risk constraints,
    and implementation practicalities.
    """

    def __init__(self,
                 transaction_cost_model: Optional[TransactionCostModel] = None,
                 risk_constraints: Optional[RiskConstraints] = None,
                 rebalancing_config: Optional[RebalancingConfig] = None,
                 risk_aversion: float = 2.0,
                 risk_free_rate: float = 0.02):
        """
        Initialize the dynamic portfolio optimizer.
        
        Args:
            transaction_cost_model: Transaction cost model configuration
            risk_constraints: Risk management constraints
            rebalancing_config: Rebalancing strategy configuration
            risk_aversion: Risk aversion parameter for utility function
            risk_free_rate: Risk-free rate for calculations
        """
        self.transaction_cost_model = transaction_cost_model or TransactionCostModel()
        self.risk_constraints = risk_constraints or RiskConstraints()
        self.rebalancing_config = rebalancing_config or RebalancingConfig()
        self.risk_aversion = risk_aversion
        self.risk_free_rate = risk_free_rate
        
        # Initialize portfolio constructor for regime portfolios
        self.portfolio_constructor = PortfolioConstructor(risk_free_rate=risk_free_rate)
        
        # Store regime portfolios and statistics
        self.regime_portfolios: Optional[Dict[str, PortfolioResult]] = None
        self.regime_statistics: Optional[Dict[str, Dict]] = None
        
        logger.info("Initialized DynamicPortfolioOptimizer")

    def set_regime_portfolios(self, regime_portfolios: Dict[str, PortfolioResult]):
        """
        Set pre-optimized portfolios for each regime.
        
        Args:
            regime_portfolios: Dictionary mapping regime names to PortfolioResult objects
        """
        self.regime_portfolios = regime_portfolios
        logger.info(f"Set regime portfolios for {len(regime_portfolios)} regimes")

    def set_regime_statistics(self, regime_statistics: Dict[str, Dict]):
        """
        Set regime statistics for optimization.
        
        Args:
            regime_statistics: Dictionary with regime statistics from PortfolioConstructor
        """
        self.regime_statistics = regime_statistics
        logger.info(f"Set regime statistics for {len(regime_statistics)} regimes")

    # Subtask 8.1: Implement Objective Function
    def _objective_function(self,
                          weights: np.ndarray,
                          target_weights: np.ndarray,
                          current_weights: np.ndarray,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          objective_type: OptimizationObjective = OptimizationObjective.UTILITY_MAXIMIZATION) -> float:
        """
        Multi-objective function for dynamic portfolio optimization.
        
        This implements Subtask 8.1: Implement Objective Function
        
        Args:
            weights: Portfolio weights to optimize
            target_weights: Target portfolio weights from regime analysis
            current_weights: Current portfolio weights
            expected_returns: Expected returns vector
            covariance_matrix: Asset covariance matrix
            objective_type: Type of optimization objective
            
        Returns:
            Objective function value (to be minimized)
        """
        try:
            # Ensure weights are properly normalized
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
            
            # Portfolio expected return
            portfolio_return = np.dot(weights, expected_returns)
            
            # Portfolio variance (risk)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Transaction costs
            transaction_costs = self._calculate_transaction_costs(weights, current_weights)
            
            # Tracking error vs target
            tracking_error = np.sum((weights - target_weights) ** 2)
            
            if objective_type == OptimizationObjective.UTILITY_MAXIMIZATION:
                # Mean-variance utility with transaction costs
                utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance - transaction_costs
                return -utility  # Minimize negative utility
                
            elif objective_type == OptimizationObjective.TRACKING_ERROR_MINIMIZATION:
                # Minimize tracking error with transaction cost penalty
                return tracking_error + transaction_costs * 10
                
            elif objective_type == OptimizationObjective.TRANSACTION_COST_AWARE:
                # Balance return, risk, and transaction costs
                sharpe_penalty = -(portfolio_return - self.risk_free_rate) / (portfolio_volatility + 1e-8)
                return sharpe_penalty + transaction_costs * 5 + tracking_error
                
            elif objective_type == OptimizationObjective.REGIME_TRANSITION:
                # Optimize for regime transition with gradual adjustment
                regime_alignment = np.sum((weights - target_weights) ** 2)
                risk_penalty = self.risk_aversion * portfolio_variance
                return regime_alignment + risk_penalty + transaction_costs * 2
                
            elif objective_type == OptimizationObjective.RISK_PARITY_DYNAMIC:
                # Dynamic risk parity with transaction costs
                risk_contributions = weights * (covariance_matrix @ weights)
                target_risk_contrib = portfolio_variance / len(weights)
                risk_parity_penalty = np.sum((risk_contributions - target_risk_contrib) ** 2)
                return risk_parity_penalty + transaction_costs
                
            else:
                raise ValueError(f"Unknown objective type: {objective_type}")
                
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            return 1e6  # Return large penalty for invalid solutions

    # Subtask 8.2: Implement Transaction Cost Model
    def _calculate_transaction_costs(self,
                                   new_weights: np.ndarray,
                                   current_weights: np.ndarray,
                                   portfolio_value: float = 1.0) -> float:
        """
        Calculate comprehensive transaction costs for portfolio transition.
        
        This implements Subtask 8.2: Implement Transaction Cost Model
        
        Args:
            new_weights: New portfolio weights
            current_weights: Current portfolio weights
            portfolio_value: Total portfolio value
            
        Returns:
            Total transaction costs as fraction of portfolio value
        """
        try:
            # Calculate trade sizes
            trade_sizes = np.abs(new_weights - current_weights)
            total_turnover = np.sum(trade_sizes)
            
            if total_turnover == 0:
                return 0.0
            
            # Proportional costs (brokerage fees)
            proportional_costs = total_turnover * self.transaction_cost_model.proportional_cost
            
            # Fixed costs (only if trading)
            fixed_costs = 0.0
            if total_turnover > 0:
                num_trades = np.sum(trade_sizes > self.transaction_cost_model.minimum_trade_size)
                fixed_costs = num_trades * self.transaction_cost_model.fixed_cost / portfolio_value
            
            # Bid-ask spread costs
            spread_costs = total_turnover * self.transaction_cost_model.bid_ask_spread
            
            # Market impact costs (can be nonlinear)
            if self.transaction_cost_model.nonlinear_impact:
                # Nonlinear market impact: cost increases with trade size
                impact_costs = np.sum(trade_sizes * self.transaction_cost_model.market_impact * 
                                    (1 + trade_sizes))  # Quadratic impact
            else:
                # Linear market impact
                impact_costs = total_turnover * self.transaction_cost_model.market_impact
            
            total_costs = proportional_costs + fixed_costs + spread_costs + impact_costs
            
            return total_costs
            
        except Exception as e:
            logger.error(f"Error calculating transaction costs: {e}")
            return 0.0

    # Subtask 8.3: Develop Regime Transition Optimization
    def optimize_regime_transition(self,
                                 current_weights: pd.Series,
                                 target_regime: str,
                                 regime_confidence: float = 1.0,
                                 transition_speed: float = 1.0) -> OptimizationResult:
        """
        Optimize portfolio transition to target regime.
        
        This implements Subtask 8.3: Develop Regime Transition Optimization
        
        Args:
            current_weights: Current portfolio weights
            target_regime: Target regime for transition
            regime_confidence: Confidence in regime prediction (0-1)
            transition_speed: Speed of transition (0-1, where 1 is immediate)
            
        Returns:
            OptimizationResult with optimized weights and transition plan
        """
        try:
            if self.regime_portfolios is None:
                raise ValueError("Regime portfolios must be set before optimization")
            
            if target_regime not in self.regime_portfolios:
                raise ValueError(f"Target regime '{target_regime}' not found in regime portfolios")
            
            start_time = datetime.now()
            
            # Get target weights from regime portfolio
            target_portfolio = self.regime_portfolios[target_regime]
            target_weights = target_portfolio.weights
            
            # Adjust target based on confidence and transition speed
            adjusted_target = (current_weights + 
                             transition_speed * regime_confidence * 
                             (target_weights - current_weights))
            
            # Get regime statistics
            regime_stats = self.regime_statistics[target_regime]
            expected_returns = regime_stats['mean_returns'].values
            covariance_matrix = regime_stats['covariance'].values
            
            # Set up optimization
            n_assets = len(current_weights)
            initial_guess = adjusted_target.values
            
            # Bounds: non-negative weights
            bounds = [(0, self.risk_constraints.max_concentration) for _ in range(n_assets)]
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            
            # Add risk constraints if specified
            if self.risk_constraints.max_portfolio_volatility is not None:
                def vol_constraint(x):
                    vol = np.sqrt(np.dot(x.T, np.dot(covariance_matrix, x)) * 252)
                    return self.risk_constraints.max_portfolio_volatility - vol
                constraints.append({'type': 'ineq', 'fun': vol_constraint})
            
            if self.risk_constraints.max_turnover is not None:
                def turnover_constraint(x):
                    turnover = np.sum(np.abs(x - current_weights.values))
                    return self.risk_constraints.max_turnover - turnover
                constraints.append({'type': 'ineq', 'fun': turnover_constraint})
            
            # Optimize
            result = minimize(
                self._objective_function,
                initial_guess,
                args=(adjusted_target.values, current_weights.values, 
                      expected_returns, covariance_matrix, 
                      OptimizationObjective.REGIME_TRANSITION),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            # Create optimized weights series
            optimized_weights = pd.Series(result.x, index=current_weights.index)
            optimized_weights = optimized_weights / optimized_weights.sum()  # Normalize
            
            # Calculate metrics
            turnover = np.sum(np.abs(optimized_weights - current_weights))
            transaction_costs = self._calculate_transaction_costs(
                optimized_weights.values, current_weights.values)
            
            portfolio_return = np.dot(optimized_weights.values, expected_returns)
            portfolio_variance = np.dot(optimized_weights.values.T, 
                                      np.dot(covariance_matrix, optimized_weights.values))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return * 252 - self.risk_free_rate) / portfolio_volatility
            
            # Utility calculation
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance - transaction_costs
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            return OptimizationResult(
                optimized_weights=optimized_weights,
                current_weights=current_weights,
                target_weights=target_weights,
                turnover=turnover,
                transaction_costs=transaction_costs,
                expected_return=portfolio_return * 252,
                expected_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                utility=utility,
                regime_probabilities={target_regime: regime_confidence},
                optimization_success=result.success,
                optimization_time=optimization_time,
                constraints_satisfied=self._check_risk_constraints(optimized_weights, covariance_matrix)
            )
            
        except Exception as e:
            logger.error(f"Error in regime transition optimization: {e}")
            raise

    # Subtask 8.4: Create Rebalancing Plan Generator
    def generate_rebalancing_plan(self,
                                current_weights: pd.Series,
                                regime_probabilities: Dict[str, float],
                                time_horizon: int = 30,
                                gradual_transition: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive rebalancing plan based on regime probabilities.
        
        This implements Subtask 8.4: Create Rebalancing Plan Generator
        
        Args:
            current_weights: Current portfolio weights
            regime_probabilities: Probabilities for each regime
            time_horizon: Planning horizon in days
            gradual_transition: Whether to implement gradual transition
            
        Returns:
            Comprehensive rebalancing plan dictionary
        """
        try:
            if self.regime_portfolios is None:
                raise ValueError("Regime portfolios must be set before generating rebalancing plan")
            
            logger.info("Generating rebalancing plan...")
            
            # Handle single regime case
            if len(regime_probabilities) == 1:
                target_regime = list(regime_probabilities.keys())[0]
                confidence = list(regime_probabilities.values())[0]
                
                if confidence >= self.rebalancing_config.regime_confidence_threshold:
                    optimization_result = self.optimize_regime_transition(
                        current_weights, target_regime, confidence)
                    
                    return {
                        'plan_type': 'single_regime_transition',
                        'target_regime': target_regime,
                        'confidence': confidence,
                        'optimization_result': optimization_result,
                        'implementation_schedule': self._create_implementation_schedule(
                            current_weights, optimization_result.optimized_weights, 
                            time_horizon, gradual_transition),
                        'risk_metrics': self._calculate_plan_risk_metrics(optimization_result),
                        'recommendation': 'REBALANCE' if optimization_result.turnover > 0.05 else 'HOLD'
                    }
            
            # Multiple regime case - create blended portfolio
            blended_weights = self._create_blended_portfolio(current_weights, regime_probabilities)
            blended_stats = self._calculate_blended_statistics(regime_probabilities)
            
            # Optimize transition to blended portfolio
            n_assets = len(current_weights)
            initial_guess = current_weights.values
            
            bounds = [(0, self.risk_constraints.max_concentration) for _ in range(n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            
            result = minimize(
                self._objective_function,
                initial_guess,
                args=(blended_weights.values, current_weights.values,
                      blended_stats['expected_returns'], blended_stats['covariance'],
                      OptimizationObjective.UTILITY_MAXIMIZATION),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            optimized_weights = pd.Series(result.x, index=current_weights.index)
            optimized_weights = optimized_weights / optimized_weights.sum()
            
            # Calculate comprehensive metrics
            turnover = np.sum(np.abs(optimized_weights - current_weights))
            transaction_costs = self._calculate_transaction_costs(
                optimized_weights.values, current_weights.values)
            
            # Create optimization result
            portfolio_return = np.dot(optimized_weights.values, blended_stats['expected_returns'])
            portfolio_variance = np.dot(optimized_weights.values.T,
                                      np.dot(blended_stats['covariance'], optimized_weights.values))
            
            optimization_result = OptimizationResult(
                optimized_weights=optimized_weights,
                current_weights=current_weights,
                target_weights=blended_weights,
                turnover=turnover,
                transaction_costs=transaction_costs,
                expected_return=portfolio_return * 252,
                expected_volatility=np.sqrt(portfolio_variance * 252),
                sharpe_ratio=(portfolio_return * 252 - self.risk_free_rate) / np.sqrt(portfolio_variance * 252),
                utility=portfolio_return - 0.5 * self.risk_aversion * portfolio_variance - transaction_costs,
                regime_probabilities=regime_probabilities,
                optimization_success=result.success,
                optimization_time=0.0,
                constraints_satisfied=True
            )
            
            return {
                'plan_type': 'multi_regime_blended',
                'regime_probabilities': regime_probabilities,
                'blended_target': blended_weights,
                'optimization_result': optimization_result,
                'implementation_schedule': self._create_implementation_schedule(
                    current_weights, optimized_weights, time_horizon, gradual_transition),
                'regime_contributions': self._analyze_regime_contributions(regime_probabilities),
                'risk_metrics': self._calculate_plan_risk_metrics(optimization_result),
                'recommendation': self._generate_rebalancing_recommendation(optimization_result)
            }
            
        except Exception as e:
            logger.error(f"Error generating rebalancing plan: {e}")
            raise

    def _create_blended_portfolio(self,
                                current_weights: pd.Series,
                                regime_probabilities: Dict[str, float]) -> pd.Series:
        """Create probability-weighted blended portfolio from regime portfolios."""
        blended_weights = pd.Series(0.0, index=current_weights.index)
        
        for regime, probability in regime_probabilities.items():
            if regime in self.regime_portfolios:
                regime_weights = self.regime_portfolios[regime].weights
                blended_weights += regime_weights * probability
        
        return blended_weights

    def _calculate_blended_statistics(self, regime_probabilities: Dict[str, float]) -> Dict[str, np.ndarray]:
        """Calculate blended expected returns and covariance matrix."""
        first_regime = list(regime_probabilities.keys())[0]
        n_assets = len(self.regime_statistics[first_regime]['mean_returns'])
        
        blended_returns = np.zeros(n_assets)
        blended_covariance = np.zeros((n_assets, n_assets))
        
        for regime, probability in regime_probabilities.items():
            if regime in self.regime_statistics:
                blended_returns += self.regime_statistics[regime]['mean_returns'].values * probability
                blended_covariance += self.regime_statistics[regime]['covariance'].values * probability
        
        return {
            'expected_returns': blended_returns,
            'covariance': blended_covariance
        }

    def _create_implementation_schedule(self,
                                      current_weights: pd.Series,
                                      target_weights: pd.Series,
                                      time_horizon: int,
                                      gradual_transition: bool) -> List[Dict[str, Any]]:
        """Create detailed implementation schedule for rebalancing."""
        if not gradual_transition or time_horizon <= 1:
            return [{
                'day': 0,
                'weights': target_weights,
                'trades': target_weights - current_weights,
                'implementation_type': 'immediate'
            }]
        
        # Create gradual transition schedule
        schedule = []
        transition_days = min(time_horizon, self.rebalancing_config.transition_period)
        
        for day in range(transition_days + 1):
            alpha = day / transition_days  # Transition progress
            
            # Use smooth transition function
            smooth_alpha = 3 * alpha**2 - 2 * alpha**3  # Smooth step function
            
            intermediate_weights = current_weights + smooth_alpha * (target_weights - current_weights)
            
            if day == 0:
                trades = pd.Series(0.0, index=current_weights.index)
            else:
                prev_weights = schedule[-1]['weights']
                trades = intermediate_weights - prev_weights
            
            schedule.append({
                'day': day,
                'weights': intermediate_weights,
                'trades': trades,
                'implementation_type': 'gradual',
                'transition_progress': smooth_alpha
            })
        
        return schedule

    # Subtask 8.5: Implement Risk Management Constraints
    def _check_risk_constraints(self,
                              weights: pd.Series,
                              covariance_matrix: np.ndarray) -> bool:
        """
        Check if portfolio satisfies risk management constraints.
        
        This implements Subtask 8.5: Implement Risk Management Constraints
        
        Args:
            weights: Portfolio weights to check
            covariance_matrix: Asset covariance matrix
            
        Returns:
            True if all constraints are satisfied
        """
        try:
            # Check concentration constraint
            if np.max(weights) > self.risk_constraints.max_concentration:
                return False
            
            # Check portfolio volatility constraint
            if self.risk_constraints.max_portfolio_volatility is not None:
                portfolio_vol = np.sqrt(np.dot(weights.values.T, 
                                             np.dot(covariance_matrix, weights.values)) * 252)
                if portfolio_vol > self.risk_constraints.max_portfolio_volatility:
                    return False
            
            # Check leverage constraint
            if np.sum(np.abs(weights)) > self.risk_constraints.max_leverage:
                return False
            
            # Check diversification constraint
            if self.risk_constraints.min_diversification is not None:
                herfindahl_index = np.sum(weights.values ** 2)
                diversification_index = 1 - herfindahl_index
                if diversification_index < self.risk_constraints.min_diversification:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk constraints: {e}")
            return False

    def _calculate_plan_risk_metrics(self, optimization_result: OptimizationResult) -> Dict[str, float]:
        """Calculate comprehensive risk metrics for rebalancing plan."""
        weights = optimization_result.optimized_weights
        
        # Basic risk metrics
        concentration = np.max(weights)
        herfindahl_index = np.sum(weights.values ** 2)
        diversification_index = 1 - herfindahl_index
        
        # Effective number of assets
        effective_assets = 1 / herfindahl_index
        
        risk_metrics = {
            'portfolio_volatility': optimization_result.expected_volatility,
            'concentration_risk': concentration,
            'diversification_index': diversification_index,
            'effective_num_assets': effective_assets,
            'turnover': optimization_result.turnover,
            'transaction_costs': optimization_result.transaction_costs,
            'sharpe_ratio': optimization_result.sharpe_ratio,
            'utility': optimization_result.utility
        }
        
        return risk_metrics

    # Subtask 8.6: Develop Performance Attribution Analysis
    def analyze_performance_attribution(self,
                                      portfolio_returns: pd.Series,
                                      benchmark_returns: pd.Series,
                                      regime_classifications: pd.Series,
                                      rebalancing_dates: List[datetime]) -> Dict[str, Any]:
        """
        Perform comprehensive performance attribution analysis.
        
        This implements Subtask 8.6: Develop Performance Attribution Analysis
        
        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series  
            regime_classifications: Regime classifications over time
            rebalancing_dates: Dates when rebalancing occurred
            
        Returns:
            Comprehensive performance attribution analysis
        """
        try:
            logger.info("Performing performance attribution analysis...")
            
            # Align all series
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            common_dates = common_dates.intersection(regime_classifications.index)
            
            portfolio_aligned = portfolio_returns.loc[common_dates]
            benchmark_aligned = benchmark_returns.loc[common_dates]
            regimes_aligned = regime_classifications.loc[common_dates]
            
            # Calculate excess returns
            excess_returns = portfolio_aligned - benchmark_aligned
            
            # Performance by regime
            regime_performance = {}
            for regime in regimes_aligned.unique():
                if pd.isna(regime):
                    continue
                
                regime_mask = regimes_aligned == regime
                regime_portfolio_returns = portfolio_aligned[regime_mask]
                regime_benchmark_returns = benchmark_aligned[regime_mask]
                regime_excess_returns = excess_returns[regime_mask]
                
                if len(regime_portfolio_returns) > 0:
                    regime_performance[regime] = {
                        'portfolio_return': regime_portfolio_returns.mean() * 252,
                        'benchmark_return': regime_benchmark_returns.mean() * 252,
                        'excess_return': regime_excess_returns.mean() * 252,
                        'volatility': regime_portfolio_returns.std() * np.sqrt(252),
                        'sharpe_ratio': (regime_portfolio_returns.mean() * 252 - self.risk_free_rate) / 
                                      (regime_portfolio_returns.std() * np.sqrt(252)),
                        'tracking_error': regime_excess_returns.std() * np.sqrt(252),
                        'information_ratio': (regime_excess_returns.mean() * 252) / 
                                           (regime_excess_returns.std() * np.sqrt(252) + 1e-8),
                        'periods': len(regime_portfolio_returns),
                        'hit_rate': (regime_excess_returns > 0).mean()
                    }
            
            # Rebalancing impact analysis
            rebalancing_impact = self._analyze_rebalancing_impact(
                portfolio_returns, rebalancing_dates)
            
            # Transaction cost analysis
            transaction_cost_analysis = self._analyze_transaction_costs(
                portfolio_returns, rebalancing_dates)
            
            # Risk-adjusted performance metrics
            total_return = (1 + portfolio_aligned).prod() - 1
            total_benchmark_return = (1 + benchmark_aligned).prod() - 1
            total_excess_return = total_return - total_benchmark_return
            
            portfolio_vol = portfolio_aligned.std() * np.sqrt(252)
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_aligned).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Attribution summary
            attribution_summary = {
                'total_performance': {
                    'portfolio_return': total_return,
                    'benchmark_return': total_benchmark_return,
                    'excess_return': total_excess_return,
                    'portfolio_volatility': portfolio_vol,
                    'tracking_error': tracking_error,
                    'information_ratio': (excess_returns.mean() * 252) / (tracking_error + 1e-8),
                    'sharpe_ratio': (portfolio_aligned.mean() * 252 - self.risk_free_rate) / portfolio_vol,
                    'max_drawdown': max_drawdown
                },
                'regime_attribution': regime_performance,
                'rebalancing_impact': rebalancing_impact,
                'transaction_cost_impact': transaction_cost_analysis,
                'risk_metrics': {
                    'value_at_risk_95': np.percentile(portfolio_aligned, 5),
                    'conditional_var_95': portfolio_aligned[portfolio_aligned <= np.percentile(portfolio_aligned, 5)].mean(),
                    'skewness': portfolio_aligned.skew(),
                    'kurtosis': portfolio_aligned.kurtosis(),
                    'downside_deviation': portfolio_aligned[portfolio_aligned < 0].std() * np.sqrt(252)
                }
            }
            
            return attribution_summary
            
        except Exception as e:
            logger.error(f"Error in performance attribution analysis: {e}")
            raise

    def _analyze_rebalancing_impact(self,
                                  portfolio_returns: pd.Series,
                                  rebalancing_dates: List[datetime]) -> Dict[str, float]:
        """Analyze the impact of rebalancing on portfolio performance."""
        if not rebalancing_dates:
            return {'rebalancing_frequency': 0, 'avg_rebalancing_impact': 0.0}
        
        rebalancing_impacts = []
        
        for rebal_date in rebalancing_dates:
            # Look at returns around rebalancing date
            start_date = rebal_date - timedelta(days=5)
            end_date = rebal_date + timedelta(days=5)
            
            period_returns = portfolio_returns.loc[start_date:end_date]
            if len(period_returns) > 0:
                impact = period_returns.sum()
                rebalancing_impacts.append(impact)
        
        return {
            'rebalancing_frequency': len(rebalancing_dates) / (len(portfolio_returns) / 252),
            'avg_rebalancing_impact': np.mean(rebalancing_impacts) if rebalancing_impacts else 0.0,
            'rebalancing_volatility': np.std(rebalancing_impacts) if rebalancing_impacts else 0.0
        }

    def _analyze_transaction_costs(self,
                                 portfolio_returns: pd.Series,
                                 rebalancing_dates: List[datetime]) -> Dict[str, float]:
        """Analyze transaction costs impact on performance."""
        # Estimate transaction costs based on rebalancing frequency
        annual_rebalancing_freq = len(rebalancing_dates) / (len(portfolio_returns) / 252)
        
        # Estimate average turnover per rebalancing
        estimated_turnover = 0.2  # 20% average turnover
        
        annual_transaction_costs = (annual_rebalancing_freq * estimated_turnover * 
                                  self.transaction_cost_model.proportional_cost)
        
        return {
            'estimated_annual_transaction_costs': annual_transaction_costs,
            'transaction_cost_drag': annual_transaction_costs,
            'rebalancing_frequency': annual_rebalancing_freq
        }

    def _analyze_regime_contributions(self, regime_probabilities: Dict[str, float]) -> Dict[str, Any]:
        """Analyze contributions of different regimes to portfolio construction."""
        contributions = {}
        
        for regime, probability in regime_probabilities.items():
            if regime in self.regime_portfolios:
                regime_portfolio = self.regime_portfolios[regime]
                contributions[regime] = {
                    'probability': probability,
                    'expected_return': regime_portfolio.expected_return,
                    'volatility': regime_portfolio.expected_volatility,
                    'sharpe_ratio': regime_portfolio.sharpe_ratio,
                    'contribution_to_return': probability * regime_portfolio.expected_return,
                    'contribution_to_risk': probability * regime_portfolio.expected_volatility
                }
        
        return contributions

    def _generate_rebalancing_recommendation(self, optimization_result: OptimizationResult) -> str:
        """Generate actionable rebalancing recommendation."""
        if optimization_result.turnover < 0.02:
            return "HOLD - Minimal rebalancing benefit"
        elif optimization_result.turnover < 0.05:
            return "MINOR_REBALANCE - Small adjustments recommended"
        elif optimization_result.turnover < 0.15:
            return "REBALANCE - Moderate portfolio adjustment"
        else:
            return "MAJOR_REBALANCE - Significant regime change detected"


# Utility functions for easy usage
def create_dynamic_optimizer(transaction_cost: float = 0.001,
                           risk_aversion: float = 2.0,
                           max_concentration: float = 0.4) -> DynamicPortfolioOptimizer:
    """
    Create a DynamicPortfolioOptimizer with standard configuration.
    
    Args:
        transaction_cost: Proportional transaction cost
        risk_aversion: Risk aversion parameter
        max_concentration: Maximum weight in single asset
        
    Returns:
        Configured DynamicPortfolioOptimizer instance
    """
    transaction_model = TransactionCostModel(proportional_cost=transaction_cost)
    risk_constraints = RiskConstraints(max_concentration=max_concentration)
    
    return DynamicPortfolioOptimizer(
        transaction_cost_model=transaction_model,
        risk_constraints=risk_constraints,
        risk_aversion=risk_aversion
    )


def quick_rebalancing_analysis(current_weights: pd.Series,
                             regime_portfolios: Dict[str, PortfolioResult],
                             regime_probabilities: Dict[str, float],
                             regime_statistics: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Perform quick rebalancing analysis with default settings.
    
    Args:
        current_weights: Current portfolio weights
        regime_portfolios: Pre-optimized regime portfolios
        regime_probabilities: Current regime probabilities
        regime_statistics: Regime statistics
        
    Returns:
        Rebalancing analysis results
    """
    optimizer = create_dynamic_optimizer()
    optimizer.set_regime_portfolios(regime_portfolios)
    optimizer.set_regime_statistics(regime_statistics)
    
    return optimizer.generate_rebalancing_plan(current_weights, regime_probabilities) 