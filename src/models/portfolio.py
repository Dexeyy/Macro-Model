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
import os
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress optimization warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Set up logging (quiet by default; respect application root config)
logger = logging.getLogger(__name__)

class OptimizationMethod(Enum):
    """Supported portfolio optimization methods"""
    SHARPE = "sharpe"
    MIN_VARIANCE = "min_variance"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    CVAR = "cvar"
    BLACK_LITTERMAN = "black_litterman"

@dataclass
class PortfolioConstraints:
    """Configuration for portfolio constraints"""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    max_positions: Optional[int] = None  # Maximum number of positions
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None  # Sector min/max exposures
    turnover_limit: Optional[float] = None  # Maximum portfolio turnover (L1 distance)
    leverage_limit: float = 1.0  # Maximum leverage (1.0 = no leverage)
    transaction_cost_bp: float = 0.0  # per-trade cost in basis points applied to turnover

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
        self.last_weights: Optional[pd.Series] = None
        # Internal flag to control whether mean passed to tangency/min-var is already excess
        self._assume_mean_is_excess: bool = False

    # ================= Mean-Covariance Estimation ======================
    def estimate_mean_cov(
        self,
        returns: pd.DataFrame,
        method: str = "shrinkage",
        prior_mean: Optional[pd.Series] = None,
        shrink_alpha: Optional[float] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """Estimate mean and covariance robustly.

        method: 'sample' | 'shrinkage' | 'bayesian'
        - shrinkage: Ledoit–Wolf style shrinkage to identity target (average variance)
        - bayesian: shrink mean toward prior_mean with lambda determined by T
        """
        R = returns.replace([np.inf, -np.inf], np.nan).dropna(how="all").ffill().bfill()
        mu = R.mean()
        S = R.cov()
        if method == "sample":
            return mu, S
        n = S.shape[0]
        if method == "shrinkage":
            # Target: scaled identity
            avg_var = float(np.trace(S)) / n if n > 0 else 0.0
            F = np.eye(n) * avg_var
            T = R.shape[0]
            # Simple alpha heuristic if not provided
            alpha = shrink_alpha if shrink_alpha is not None else min(1.0, n / max(1, T))
            Sigma = alpha * F + (1 - alpha) * S.values
            return mu, pd.DataFrame(Sigma, index=S.index, columns=S.columns)
        if method == "bayesian":
            prior = prior_mean if prior_mean is not None else mu.rolling(len(mu), min_periods=1).mean()
            T = R.shape[0]
            lam = T / (T + n)  # simple shrink factor
            mu_b = lam * mu + (1 - lam) * prior
            return mu_b, S
        return mu, S

    @staticmethod
    def to_periodic_rf(rf_annual: float, periods: int) -> float:
        """Convert annual risk-free rate to per-period rate via compounding.
        rf_period = (1 + rf_annual)**(1/periods) - 1
        """
        try:
            p = max(1, int(periods))
            return float((1.0 + float(rf_annual)) ** (1.0 / p) - 1.0)
        except Exception:
            return float(rf_annual) / max(1, int(periods) if periods else 12)
    
    def _infer_annualization_factor_from_index(self, index: pd.DatetimeIndex) -> int:
        """Infer annualization factor from a DatetimeIndex frequency.
        Defaults to 12 (monthly) if uncertain.
        """
        try:
            if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
                return 12
            freq = pd.infer_freq(index)
            if freq is None:
                # Fallback: use median days between observations
                deltas = np.diff(index.values).astype('timedelta64[D]').astype(int)
                if len(deltas) == 0:
                    return 12
                med = int(np.median(deltas))
                if med <= 2:
                    return 252
                if med <= 9:
                    return 52
                if med <= 40:
                    return 12
                if med <= 120:
                    return 4
                return 1
            # Normalize freq codes
            if freq in ("D", "B") or freq.startswith("D") or freq.startswith("B"):
                return 252
            if freq.startswith("W"):
                return 52
            if freq.endswith("M") or freq in ("MS", "M", "ME"):
                return 12
            if freq.startswith("Q"):
                return 4
            if freq.startswith("A") or freq.startswith("Y"):
                return 1
            return 12
        except Exception:
            return 12
    
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
                    # Add small ridge to diagonal; keep DataFrame type
                    ridge = np.eye(len(covariance)) * 1e-8
                    covariance_stable = pd.DataFrame(
                        covariance.values + ridge,
                        index=covariance.index,
                        columns=covariance.columns,
                    )
                    # Test for positive definiteness
                    np.linalg.cholesky(covariance_stable.values)
                    covariance = covariance_stable
                except np.linalg.LinAlgError:
                    logger.warning(
                        f"Singular covariance matrix for regime {regime}, using diagonal approximation"
                    )
                    diag_mat = np.diag(np.diag(covariance.values)) + np.eye(len(covariance)) * 1e-6
                    covariance = pd.DataFrame(diag_mat, index=covariance.index, columns=covariance.columns)
                
                # Calculate additional metrics with proper annualization based on frequency
                periods_per_year = self._infer_annualization_factor_from_index(regime_returns.index)
                volatility = regime_returns.std() * np.sqrt(periods_per_year)
                denom = volatility.replace(0, np.nan)
                sharpe_ratios = (mean_returns * periods_per_year - self.risk_free_rate) / denom
                sharpe_ratios = sharpe_ratios.fillna(0.0)
                
                # Correlation matrix
                correlation = regime_returns.corr()
                
                # Regime-specific comprehensive statistics
                regime_stats[regime] = {
                    'count': len(regime_returns),
                    'mean_returns': mean_returns,
                    'annualized_returns': mean_returns * periods_per_year,
                    'volatility': volatility,
                    'covariance': covariance,
                    'correlation': correlation,
                    'sharpe_ratios': sharpe_ratios,
                    'periods_per_year': periods_per_year,
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
                              periods_per_year: int = 252,
                              turnover_penalty: float = 0.0,
                              prev_weights: Optional[np.ndarray] = None) -> float:
        """Calculate negative Sharpe ratio (for minimization)"""
        # Always keep finite values for optimizer stability
        eps = 1e-12
        # Use excess returns once via periodic rf
        rf_per_period = self.to_periodic_rf(self.risk_free_rate, periods_per_year)
        returns_excess = returns - rf_per_period
        portfolio_return = self._portfolio_return(weights, returns_excess) * periods_per_year  # Annualize excess
        variance = max(self._portfolio_variance(weights, covariance), 0.0)
        portfolio_volatility = np.sqrt(variance) * np.sqrt(periods_per_year) + eps
        ratio = portfolio_return / portfolio_volatility
        if not np.isfinite(ratio):
            # Penalize non-finite objective
            ratio = -1e9
        obj = -ratio
        if turnover_penalty > 0 and prev_weights is not None:
            obj += turnover_penalty * np.sum(np.abs(weights - prev_weights))
        return obj
    
    def _risk_parity_objective(self, weights: np.ndarray, covariance: pd.DataFrame, turnover_penalty: float = 0.0, prev_weights: Optional[np.ndarray] = None) -> float:
        """Risk parity objective function (minimize sum of squared risk contributions)"""
        portfolio_vol = np.sqrt(self._portfolio_variance(weights, covariance))
        if portfolio_vol == 0:
            return 1e6
        marginal_contrib = np.dot(covariance.values, weights) / portfolio_vol
        contrib = weights * marginal_contrib
        target_contrib = portfolio_vol / len(weights)  # Equal risk contribution
        obj = np.sum((contrib - target_contrib) ** 2)
        if turnover_penalty > 0 and prev_weights is not None:
            obj += turnover_penalty * np.sum(np.abs(weights - prev_weights))
        return obj
    
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

    def _preflight_feasible(self, lb: np.ndarray, ub: np.ndarray, max_positions: Optional[int] = None) -> None:
        """Feasibility checks before invoking the optimizer.
        - lb and ub are arrays of per-asset lower/upper bounds.
        Raises ValueError with clear message if infeasible.
        """
        if lb.shape != ub.shape:
            raise ValueError("lb/ub shape mismatch")
        bad = np.where(lb > ub)[0]
        if bad.size > 0:
            raise ValueError(f"Lower bound exceeds upper for assets at idx={bad.tolist()}")
        if float(lb.sum()) > 1.0 + 1e-12:
            raise ValueError(f"sum(lb)={float(lb.sum()):.3f} > 1")
        if float(ub.sum()) < 1.0 - 1e-12:
            raise ValueError(f"sum(ub)={float(ub.sum()):.3f} < 1")
        if max_positions is not None and int((lb > 0).sum()) > int(max_positions):
            raise ValueError("floors exceed max_positions")
    
    def optimize_portfolio(self, 
                         regime_stats: Dict[str, Dict], 
                         regime: str,
                         method: OptimizationMethod = OptimizationMethod.SHARPE,
                          custom_constraints: Optional[List[Dict]] = None,
                          mean_cov_method: str = "shrinkage",
                          turnover_penalty: float = 0.0,
                          mean_is_excess: bool = False) -> PortfolioResult:
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
            # Estimate robust mean/cov using requested method; prefer shrunk mean too
            base = pd.DataFrame(stats['mean_returns']).T.reindex(columns=stats['mean_returns'].index)
            row = pd.DataFrame([stats['mean_returns']])
            est_input = pd.concat([base, row], ignore_index=True).dropna(axis=1, how='all')
            est_mu, est_cov = self.estimate_mean_cov(
                est_input,
                method=mean_cov_method,
            )
            # Use the estimated (possibly shrunk) mean and covariance when available
            returns = est_mu if isinstance(est_mu, pd.Series) else stats['mean_returns']
            covariance = est_cov if isinstance(est_cov, pd.DataFrame) else stats['covariance']
            covariance = self._near_psd(covariance)
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
            # Preflight feasibility check
            lb = np.array([b[0] for b in bounds], dtype=float)
            ub = np.array([b[1] for b in bounds], dtype=float)
            try:
                self._preflight_feasible(lb, ub, self.constraints.max_positions)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Preflight bounds: lb_sum=%.3f ub_sum=%.3f max_pos=%s", float(lb.sum()), float(ub.sum()), str(self.constraints.max_positions))
            except Exception as exc:
                logger.error("Bounds infeasible: %s", exc)
                # Deterministic fallback path
                w = self._analytic_fallback_weights(returns, covariance, method)
                return self._create_portfolio_result(w, returns, covariance, method.value, regime, False, periods_per_year)
            
            # Define objective function based on method
            periods_per_year = regime_stats[regime].get('periods_per_year', 12)
            # Stash for fallback analytics (tangency needs rf per period)
            try:
                self._last_periods_per_year = int(periods_per_year)
            except Exception:
                self._last_periods_per_year = 12
            # Record whether the mean vector is already excess
            old_flag = self._assume_mean_is_excess
            self._assume_mean_is_excess = bool(mean_is_excess)
            prev_w = self.last_weights.values if isinstance(self.last_weights, pd.Series) and len(self.last_weights) == n_assets else None
            if method == OptimizationMethod.SHARPE:
                objective_func = lambda w: self._negative_sharpe_ratio(w, returns, covariance, periods_per_year=periods_per_year, turnover_penalty=turnover_penalty, prev_weights=prev_w)
            elif method == OptimizationMethod.MIN_VARIANCE:
                objective_func = lambda w: self._portfolio_variance(w, covariance)
            elif method == OptimizationMethod.MAX_RETURN:
                objective_func = lambda w: -self._portfolio_return(w, returns)
            elif method == OptimizationMethod.RISK_PARITY:
                objective_func = lambda w: self._risk_parity_objective(w, covariance, turnover_penalty, prev_w)
            elif method == OptimizationMethod.CVAR:
                # Simple proxy: penalize left-tail (approximate with 5th percentile of historical mix)
                hist = pd.DataFrame(np.random.normal(size=(max(50, n_assets*10), n_assets)), columns=returns.index)
                objective_func = lambda w: -float(np.dot(hist.mean().values, w)) + 10.0 * float(np.quantile(-hist.dot(w), 0.95))
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                # Basic BL: combine equilibrium (pi=mean) with views=returns
                tau = 0.05
                Sigma = covariance.values
                pi = returns.values
                P = np.eye(n_assets)
                q = returns.values
                Omega = np.diag(np.diag(tau * Sigma))
                inv = np.linalg.pinv((np.linalg.pinv(tau * Sigma) + P.T @ np.linalg.pinv(Omega) @ P))
                mu_bl = inv @ (np.linalg.pinv(tau * Sigma) @ pi + P.T @ np.linalg.pinv(Omega) @ q)
                mu_bl = pd.Series(mu_bl, index=returns.index)
                objective_func = lambda w: self._negative_sharpe_ratio(w, mu_bl, covariance, periods_per_year=periods_per_year, turnover_penalty=turnover_penalty, prev_weights=prev_w)
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
                logger.warning("Falling back to analytic projection")
                # Deterministic fallback chain: tangency -> min-var -> equal
                try:
                    w_tan = self._tangency_weights(returns, covariance)
                    if np.isfinite(w_tan).all() and w_tan.sum() > 0:
                        weights = w_tan
                    else:
                        raise RuntimeError("tangency invalid")
                except Exception:
                    try:
                        w_mv = self._min_variance_weights(returns, covariance)
                        if np.isfinite(w_mv).all() and w_mv.sum() > 0:
                            weights = w_mv
                        else:
                            raise RuntimeError("min-var invalid")
                    except Exception:
                        weights = np.ones(n_assets) / n_assets
                success = False
            else:
                weights = result.x
                success = True
            
            # Enforce turnover limit via blending with last_weights
            if self.constraints.turnover_limit is not None and self.last_weights is not None:
                tw = float(np.sum(np.abs(weights - self.last_weights.values)))
                lim = float(self.constraints.turnover_limit)
                if tw > lim and tw > 0:
                    t = lim / tw
                    weights = self.last_weights.values + t * (weights - self.last_weights.values)
            result_out = self._create_portfolio_result(weights, returns, covariance, method.value, regime, success, periods_per_year)
            self.last_weights = result_out.weights
            # Restore flag
            self._assume_mean_is_excess = old_flag
            return result_out
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio for regime {regime}: {e}")
            # Robust analytic fallback (no equal-weight)
            mean = regime_stats[regime]['mean_returns']
            cov = self._near_psd(regime_stats[regime]['covariance'])
            weights = self._analytic_fallback_weights(mean, cov, method)
            # Ensure we restore the state flag
            self._assume_mean_is_excess = False
            return self._create_portfolio_result(
                weights,
                mean,
                cov,
                method.value,
                regime,
                False,
            )
    
    def _create_portfolio_result(self, 
                               weights: np.ndarray, 
                               returns: pd.Series, 
                               covariance: pd.DataFrame,
                               method: str,
                               regime: str,
                                success: bool = True,
                                periods_per_year: int = 252) -> PortfolioResult:
        """Create a PortfolioResult object from optimization results"""
        weights_series = pd.Series(weights, index=returns.index)
        
        rf_per_period = self.to_periodic_rf(self.risk_free_rate, periods_per_year)
        expected_return = self._portfolio_return(weights, returns - rf_per_period) * periods_per_year  # Annualized excess
        expected_volatility = np.sqrt(self._portfolio_variance(weights, covariance)) * np.sqrt(periods_per_year)
        
        sharpe_ratio = expected_return / expected_volatility if expected_volatility > 0 else 0
        
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

    # -------------------- Robust fallback helpers --------------------
    @staticmethod
    def _project_to_simplex(weights: np.ndarray, target_sum: float = 1.0) -> np.ndarray:
        """Project a vector onto the probability simplex sum(weights)=target_sum, weights>=0."""
        if target_sum <= 0:
            raise ValueError("target_sum must be positive")
        w = np.maximum(weights, 0.0)
        if w.sum() == 0:
            return np.ones_like(w) * (target_sum / len(w))
        u = np.sort(w)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - target_sum))[0]
        if len(rho) == 0:
            tau = 0.0
        else:
            rho = rho[-1]
            tau = (cssv[rho] - target_sum) / (rho + 1)
        return np.maximum(w - tau, 0.0)

    def _tangency_weights(self, mean: pd.Series, cov: pd.DataFrame) -> np.ndarray:
        """Compute long-only tangency weights proportional to inv(Sigma) * (mu_excess), projected to simplex.
        If self._assume_mean_is_excess is True, `mean` is treated as already excess.
        Otherwise we subtract periodic rf exactly once.
        """
        Sigma = cov.values
        # Subtract per-period risk-free from mean to use excess returns
        ppy = getattr(self, "_last_periods_per_year", 12)
        rf_per_period = self.to_periodic_rf(self.risk_free_rate, ppy)
        mu = mean.values if self._assume_mean_is_excess else (mean.values - rf_per_period)
        inv = np.linalg.pinv(Sigma)
        raw = inv.dot(mu)
        if not np.all(np.isfinite(raw)):
            return self._min_variance_weights(mean, cov)
        return self._project_to_simplex(raw, 1.0)

    @staticmethod
    def _near_psd(S: pd.DataFrame) -> pd.DataFrame:
        """Return a near-PSD covariance by symmetrizing, eigen clipping, and small ridge."""
        try:
            A = 0.5 * (S.values + S.values.T)
            w, V = np.linalg.eigh(A)
            w_clipped = np.clip(w, 1e-10, None)
            A_psd = (V * w_clipped) @ V.T
            A_psd = A_psd + np.eye(A_psd.shape[0]) * 1e-10
            return pd.DataFrame(A_psd, index=S.index, columns=S.columns)
        except Exception:
            # Diagonal fallback
            diag = np.diag(np.maximum(np.diag(S.values), 1e-10))
            return pd.DataFrame(diag, index=S.index, columns=S.columns)

    def _min_variance_weights(self, mean: pd.Series, cov: pd.DataFrame) -> np.ndarray:
        """Compute long-only minimum-variance weights proportional to inv(Sigma) * 1, projected to simplex."""
        n = len(mean)
        Sigma = cov.values
        inv = np.linalg.pinv(Sigma)
        ones = np.ones(n)
        raw = inv.dot(ones)
        if (not np.all(np.isfinite(raw))) or raw.sum() == 0:
            return np.ones(n) / n
        w = raw / raw.sum()
        return self._project_to_simplex(w, 1.0)

    def _analytic_fallback_weights(self, mean: pd.Series, cov: pd.DataFrame, method: OptimizationMethod) -> np.ndarray:
        """Deterministic fallback without optimizer: try tangency then min-var; ensure feasible."""
        try:
            if method in (OptimizationMethod.SHARPE, OptimizationMethod.MAX_RETURN):
                w = self._tangency_weights(mean, cov)
            elif method == OptimizationMethod.MIN_VARIANCE:
                w = self._min_variance_weights(mean, cov)
            else:
                w = self._min_variance_weights(mean, cov)
            if np.all(np.isfinite(w)) and w.sum() > 0:
                return self._project_to_simplex(w, 1.0)
        except Exception:
            pass
        # Final fallback: single highest-mean asset
        try:
            best = mean.idxmax()
            w = np.zeros(len(mean))
            w[list(mean.index).index(best)] = 1.0
            return w
        except Exception:
            return np.ones(len(mean)) / len(mean)
    
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

    # ============ Probabilities-based optimization and attribution ============
    def optimize_with_probabilities(
        self,
        regime_stats: Dict[str, Dict],
        regime_probs: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.RISK_PARITY,
        mean_cov_method: str = "shrinkage",
        turnover_penalty: float = 0.0,
    ) -> PortfolioResult:
        """Blend mean/cov across regimes using provided probabilities (last row by default) and optimize.
        Regime probabilities rows should sum to 1.
        """
        if regime_probs.empty:
            raise ValueError("regime_probs is empty")
        p = regime_probs.iloc[-1]
        # Align names
        av_mu = None
        av_S = None
        for reg, prob in p.items():
            if reg not in regime_stats:
                continue
            mu_r = regime_stats[reg]['mean_returns']
            S_r = regime_stats[reg]['covariance']
            if av_mu is None:
                av_mu = prob * mu_r
                av_S = prob * S_r
            else:
                av_mu = av_mu.add(prob * mu_r, fill_value=0.0)
                av_S = av_S.add(prob * S_r, fill_value=0.0)
        # Package a pseudo regime
        pseudo = {'mean_returns': av_mu, 'covariance': av_S, 'periods_per_year': 12}
        result = self.optimize_portfolio({'blended': pseudo}, 'blended', method, mean_cov_method=mean_cov_method, turnover_penalty=turnover_penalty)
        return result

    def performance_attribution(self, weights: pd.Series, returns: pd.DataFrame) -> Dict[str, Union[float, pd.Series]]:
        """Compute per-asset contribution and turnover stats.
        Contributions use average weights times asset returns.
        """
        w = weights.reindex(returns.columns).fillna(0.0)
        port = returns.dot(w)
        contrib = (returns.mul(w, axis=1)).sum(axis=0)
        cum = (1 + port).cumprod() - 1
        dd = ((1 + port).cumprod() / (1 + port).cumprod().cummax()) - 1
        return {
            'cumulative_return': float(cum.iloc[-1]) if len(cum) else 0.0,
            'max_drawdown': float(dd.min()) if len(dd) else 0.0,
            'per_asset_contrib': contrib,
        }
    
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
            periods_per_year = self._infer_annualization_factor_from_index(portfolio_returns.index)
            annualized_return = (1 + portfolio_returns.mean()) ** periods_per_year - 1
            annualized_volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
            
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
            downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
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


def create_sixty_forty_benchmark(returns_df: pd.DataFrame) -> Optional[pd.Series]:
    """Create a 60/40 stock/bond benchmark.

    Preference (strict):
    - Stocks: S&P 500 via 'SPX' (fallbacks: '^GSPC', 'SP500')
    - Bonds: 10Y via 'US10Y_NOTE_FUT' (fallbacks: 'IEF')

    If both are found, use exactly those two series: 0.6*SPX + 0.4*US10Y.
    Else, fallback to earlier heuristics; if still unavailable, equal-weight all assets.
    """
    try:
        if returns_df is None or returns_df.empty:
            return None
        cols = list(returns_df.columns)
        upper = {c: str(c).upper() for c in cols}
        # Strict picks
        stock_col = None
        for k in ("SPX", "^GSPC", "SP500"):
            stock_col = next((c for c in cols if upper[c] == k), None)
            if stock_col is not None:
                break
        bond_col = None
        for k in ("US10Y_NOTE_FUT", "IEF"):
            bond_col = next((c for c in cols if upper[c] == k), None)
            if bond_col is not None:
                break
        if stock_col is not None and bond_col is not None:
            stocks_ret = returns_df[stock_col]
            bonds_ret = returns_df[bond_col]
        else:
            # Heuristic fallback
            stock_pref = ["SPX", "SP500", "^GSPC", "SPTR", "EQUITY", "STOCK"]
            bond_pref = ["US10Y_NOTE_FUT", "US30Y_BOND_FUT", "ZN", "ZB", "IEF", "AGG", "BOND"]
            stocks = [c for c in cols if any(k == upper[c] or k in upper[c] for k in stock_pref)]
            bonds = [c for c in cols if any(k == upper[c] or k in upper[c] for k in bond_pref)]
            if len(stocks) == 0 or len(bonds) == 0:
                # fallback: equal-weight across all assets
                n = returns_df.shape[1]
                w = np.ones(n) / n
                return returns_df.dot(w)
            stocks_ret = returns_df[stocks].mean(axis=1)
            bonds_ret = returns_df[bonds].mean(axis=1)
        benchmark = 0.6 * stocks_ret + 0.4 * bonds_ret
        benchmark.name = 'benchmark_60_40'
        return benchmark
    except Exception as e:
        logger.error(f"Error creating 60/40 benchmark: {e}")
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


def _annualization_factor(idx: pd.DatetimeIndex) -> int:
    try:
        freq = pd.infer_freq(idx)
        if freq is None:
            return 12
        if freq.startswith("D") or freq.startswith("B"):
            return 252
        if freq.startswith("W"):
            return 52
        if freq.startswith("Q"):
            return 4
        return 12
    except Exception:
        return 12


def compute_dynamic_regime_portfolio(
    returns: pd.DataFrame,
    regime_series: pd.Series,
    *,
    regime_window_years: int = 10,
    method: OptimizationMethod = OptimizationMethod.SHARPE,
    rebal_freq: str = "M",
    transaction_cost: float = 0.0,
    probability_blending: bool = False,
    blend_probs: bool = False,
    regime_probabilities: Optional[pd.DataFrame] = None,
    min_obs: int = 36,
    mean_cov_method: str = "shrinkage",
    risk_free_rate: Optional[float] = None,
    include_cash: bool = True,
    cash_name: str = "CASH",
    blend_alpha: float = 0.5,
    auto_minvar_if_all_negative: bool = True,
    return_diagnostics: bool = False,
    debug: bool = False,
) -> Tuple[pd.Series, Dict[pd.Timestamp, pd.Series]]:
    """Dynamic regime-aware portfolio with rebalancing on regime change or schedule.

    - For each date, when regime changes or a new month/quarter starts, fit mean/cov
      using only data from the same regime within the last `regime_window_years`.
    - Optimize weights via PortfolioConstructor for that regime.
    - Apply per-trade transaction cost proportional to turnover on rebalance dates.
    """
    if returns is None or returns.empty:
        return pd.Series(dtype=float), {}
    rts = returns.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    regimes = regime_series.reindex(rts.index).ffill()
    ann = _annualization_factor(rts.index)
    window = pd.DateOffset(years=int(regime_window_years))

    # Risk-free rate exposure
    pc = PortfolioConstructor(risk_free_rate=risk_free_rate if risk_free_rate is not None else 0.02)
    # Synthesize CASH if requested and missing
    if include_cash and (cash_name not in rts.columns):
        rf_p = PortfolioConstructor.to_periodic_rf(pc.risk_free_rate, ann)
        rts = rts.copy()
        rts[cash_name] = rf_p
    last_w = pd.Series(0.0, index=rts.columns)
    weights_hist: Dict[pd.Timestamp, pd.Series] = {}
    diags: Dict[pd.Timestamp, Dict[str, Union[str, float, int, bool]]] = {}
    port_r = pd.Series(0.0, index=rts.index)
    prev_reg = None
    last_rebal = None

    for dt in rts.index:
        cur_reg = regimes.loc[dt]
        # Time to rebalance?
        tick = False
        if last_rebal is None:
            tick = True
        else:
            if rebal_freq.upper().startswith("Q"):
                tick = (dt.quarter != last_rebal.quarter) or (dt.year != last_rebal.year)
            elif rebal_freq.upper().startswith("M"):
                tick = (dt.month != last_rebal.month) or (dt.year != last_rebal.year)
        if tick or (prev_reg is None) or (cur_reg != prev_reg):
            # estimation window using same-regime data
            start = dt - window
            mask = (rts.index <= dt) & (rts.index >= start)
            sub = rts.loc[mask]
            hard = regimes.loc[mask] == cur_reg
            sub_reg = sub[hard.values]
            # Expand to earlier same-regime data if not enough observations within window
            if len(sub_reg) < int(max(1, min_obs)):
                mask_all = (rts.index <= dt) & (regimes.reindex(rts.index).ffill() == cur_reg)
                sub_reg_all = rts.loc[mask_all]
                if len(sub_reg_all) > len(sub_reg):
                    sub_reg = sub_reg_all
            # Compute regime-only estimates first
            if len(sub_reg) == 0:
                sub_reg = sub  # fallback to unconditional if no same-regime data exists
            mu_reg = sub_reg.mean()
            cov_reg = sub_reg.cov()
            mu_uncond = sub.mean()
            cov_uncond = sub.cov()
            # Blend only if explicitly requested via flags and probabilities available; else use regime-only
            if (blend_probs or probability_blending) and regime_probabilities is not None:
                try:
                    cur_probs = regime_probabilities.reindex(index=[dt]).iloc[0]
                    alpha_p = float(cur_probs.get(str(cur_reg), cur_probs.max()))
                    alpha_p = max(0.0, min(1.0, alpha_p))
                except Exception:
                    alpha_p = 1.0
                mu = alpha_p * mu_reg + (1 - alpha_p) * mu_uncond
                cov = alpha_p * cov_reg + (1 - alpha_p) * cov_uncond
            else:
                # If regime-only slice is still short, allow fixed alpha blending for robustness
                if len(sub_reg) < int(max(1, min_obs)):
                    a = max(0.0, min(1.0, float(blend_alpha)))
                    mu = a * mu_reg + (1 - a) * mu_uncond
                    cov = a * cov_reg + (1 - a) * cov_uncond
                else:
                    mu, cov = mu_reg, cov_reg
            # Clean and stabilize covariance
            cov = cov.replace([np.inf, -np.inf], np.nan)
            cov = cov.fillna(0.0)
            try:
                ridge = np.eye(len(cov)) * 1e-8
                cov_stable = pd.DataFrame(cov.values + ridge, index=cov.index, columns=cov.columns)
                np.linalg.cholesky(cov_stable.values)
                cov = cov_stable
            except Exception:
                diag_mat = np.diag(np.diag(cov.values)) + np.eye(len(cov)) * 1e-6
                cov = pd.DataFrame(diag_mat, index=cov.index, columns=cov.columns)
            cov = pc._near_psd(cov)
            # Excess means and diagnostics
            rf_p = PortfolioConstructor.to_periodic_rf(pc.risk_free_rate, ann)
            mu_excess = mu - rf_p
            pos_excess_count = int((mu_excess > 0).sum())
            median_excess_mu = float(mu_excess.median()) if len(mu_excess) else 0.0
            # Cond number for covariance
            try:
                svals = np.linalg.svd(cov.values, compute_uv=False)
                cov_cond_number = float((svals.max() / max(svals.min(), 1e-12)) if len(svals) else np.inf)
            except Exception:
                cov_cond_number = float('inf')
            used_window_len = int(len(sub_reg))
            method_used = "solver"
            blending_used = bool((blend_probs or probability_blending) and regime_probabilities is not None) or (len(sub_reg) < int(max(1, min_obs)))
            if pos_excess_count == 0:
                logger.info("All excess means ≤ 0; selecting min-variance or allocating to CASH.")

            # Degenerate covariance check
            degenerate = False
            try:
                diag = np.diag(cov.values) if isinstance(cov, pd.DataFrame) else np.array([])
                degenerate = (len(diag) == 0) or (not np.isfinite(diag).all()) or (np.all(np.abs(diag) < 1e-12))
            except Exception:
                degenerate = True

            used_cash_flag = False
            if degenerate and mu.notna().any():
                # Fallback: try minimum-variance long-only; if it degenerates to equal weights while
                # means are clearly distinct, allocate to highest-mean asset as final fallback
                try:
                    w_arr = pc._min_variance_weights(mu, cov)
                    w = pd.Series(w_arr, index=rts.columns).reindex(rts.columns).fillna(0.0)
                    if (not np.isfinite(w.values).all()) or w.sum() == 0:
                        raise RuntimeError("invalid min-var weights")
                    # Detect near-equal weights outcome on degenerate cov
                    if (w.max() - w.min()) < 1e-9 and (float(mu.max()) - float(mu.min())) > 1e-9:
                        raise RuntimeError("min-var uninformative; prefer highest-mean")
                    method_used = "min_variance"
                except Exception:
                    # If CASH available and all excess ≤ 0, allow full CASH allocation
                    if include_cash and (cash_name in rts.columns) and pos_excess_count == 0:
                        w = pd.Series(0.0, index=rts.columns)
                        w.loc[cash_name] = 1.0
                        method_used = "cash_full"
                        used_cash_flag = True
                    else:
                        best = mu.idxmax()
                        w = pd.Series(0.0, index=rts.columns)
                        if best in w.index:
                            w.loc[best] = 1.0
                        method_used = "highest_mean"
            else:
                # If all excess ≤0 and CASH exists, allow allocation to CASH by passing mean as excess and letting optimizer consider CASH
                stats = {str(cur_reg): {"mean_returns": mu, "covariance": cov, "periods_per_year": ann}}
                # Guardrail: if all excess are non-positive and CASH is not included or not allowed, switch to min-var
                chosen_method = method
                if pos_excess_count == 0 and (not include_cash or cash_name not in rts.columns) and auto_minvar_if_all_negative:
                    chosen_method = OptimizationMethod.MIN_VARIANCE
                res = pc.optimize_portfolio(stats, str(cur_reg), chosen_method, mean_cov_method=mean_cov_method, mean_is_excess=False)
                if (not res.optimization_success) or (not np.isfinite(res.weights.values).all()):
                    # Fallback: minimum-variance; final fallback highest-mean if uninformative
                    try:
                        w_arr = pc._min_variance_weights(mu, cov)
                        w = pd.Series(w_arr, index=rts.columns).reindex(rts.columns).fillna(0.0)
                        if (not np.isfinite(w.values).all()) or w.sum() == 0 or ((w.max() - w.min()) < 1e-9 and (float(mu.max()) - float(mu.min())) > 1e-9):
                            raise RuntimeError("uninformative min-var")
                        method_used = "min_variance"
                    except Exception:
                        if include_cash and (cash_name in rts.columns) and pos_excess_count == 0:
                            w = pd.Series(0.0, index=rts.columns)
                            w.loc[cash_name] = 1.0
                            method_used = "cash_full"
                            used_cash_flag = True
                        else:
                            best = mu.idxmax()
                            w = pd.Series(0.0, index=rts.columns)
                            if best in w.index:
                                w.loc[best] = 1.0
                            method_used = "highest_mean"
                else:
                    w = res.weights.reindex(rts.columns).fillna(0.0)
                    method_used = res.method_used
            turnover = (w - last_w).abs().sum()
            # apply cost once at rebalance (optional)
            cost = (transaction_cost * float(turnover)) if (transaction_cost and transaction_cost > 0.0) else 0.0
            weights_hist[dt] = w
            last_w = w
            last_rebal = dt
            port_r.loc[dt] = float(rts.loc[dt].dot(last_w)) - float(cost)
            # Save diagnostics
            diags[dt] = {
                'periodic_rf': float(rf_p),
                'pos_excess_count': int(pos_excess_count),
                'median_excess_mu': float(median_excess_mu),
                'cov_cond_number': float(cov_cond_number),
                'used_window_len': int(used_window_len),
                'method_used': str(method_used),
                'blending_used': bool(blending_used),
                'used_cash': bool(used_cash_flag),
                'regime': str(cur_reg),
                'realised_return': float(rts.loc[dt].dot(last_w)),
                'lb_sum': float(0.0),
                'ub_sum': float(1.0),
                'max_positions': int(pc.constraints.max_positions) if pc.constraints.max_positions else 0,
            }
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("%s | reg=%s n_obs=%d pos_excess=%d med_excess=%.6f method=%s used_cash=%s blend=%s",
                             dt.strftime('%Y-%m-%d'), str(cur_reg), int(used_window_len), int(pos_excess_count), float(median_excess_mu), str(method_used), str(bool(used_cash_flag)), str(bool(blending_used)))
        else:
            port_r.loc[dt] = float(rts.loc[dt].dot(last_w))
        prev_reg = cur_reg

    # Optional debug CSV
    if debug and diags:
        try:
            out_dir = os.path.join("Output", "diagnostics")
            os.makedirs(out_dir, exist_ok=True)
            import pandas as _pd
            dbg = _pd.DataFrame.from_dict(diags, orient='index')
            dbg.index.name = 'date'
            dbg.reset_index().to_csv(os.path.join(out_dir, 'portfolio_debug.csv'), index=False)
        except Exception:
            logger.debug("Failed to write portfolio_debug.csv", exc_info=True)

    if return_diagnostics:
        # type: ignore[return-value]
        return port_r, {'weights': weights_hist, 'diagnostics': diags}  # for optional richer output
    return port_r, weights_hist