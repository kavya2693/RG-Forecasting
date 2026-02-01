#!/usr/bin/env python3
"""
Croston's Method for Intermittent Demand Forecasting
=====================================================

Implementation of Croston's method (1972) for forecasting intermittent demand
patterns commonly seen in slow-moving items (C-segment SKUs).

Croston's method separately estimates:
1. Demand size (when demand occurs)
2. Inter-arrival time (interval between demands)

Final forecast = demand_size / inter_arrival_time

This is particularly effective for:
- Slow-moving items with sporadic sales
- Items with many zero observations
- Long gaps between sales events

References:
    Croston, J.D. (1972). "Forecasting and Stock Control for Intermittent Demands"
    Syntetos, A.A. & Boylan, J.E. (2005). "The accuracy of intermittent demand estimates"
"""

import numpy as np
from typing import Tuple, Optional


class CrostonMethod:
    """
    Croston's method for intermittent demand forecasting.

    This method is designed for items with sporadic, intermittent demand patterns
    where traditional exponential smoothing fails due to many zero observations.

    Attributes:
        alpha_demand: Smoothing parameter for demand size (0 < alpha <= 1)
        alpha_interval: Smoothing parameter for inter-arrival time (0 < alpha <= 1)
        demand_estimate: Current smoothed demand estimate
        interval_estimate: Current smoothed interval estimate
        last_demand_period: Last period when demand occurred
        is_fitted: Whether the model has been fitted

    Example:
        >>> croston = CrostonMethod(alpha_demand=0.1, alpha_interval=0.1)
        >>> historical_sales = [0, 0, 5, 0, 0, 0, 3, 0, 0, 4, 0]
        >>> croston.fit(historical_sales)
        >>> forecast = croston.predict(horizon=14)
    """

    def __init__(self, alpha_demand: float = 0.1, alpha_interval: float = 0.1):
        """
        Initialize Croston's method.

        Args:
            alpha_demand: Smoothing parameter for demand size estimates.
                         Lower values (0.05-0.15) give more stable forecasts.
                         Higher values (0.2-0.4) react faster to changes.
            alpha_interval: Smoothing parameter for inter-arrival time estimates.
                           Similar interpretation as alpha_demand.

        Raises:
            ValueError: If alpha values are not in (0, 1]
        """
        if not (0 < alpha_demand <= 1):
            raise ValueError(f"alpha_demand must be in (0, 1], got {alpha_demand}")
        if not (0 < alpha_interval <= 1):
            raise ValueError(f"alpha_interval must be in (0, 1], got {alpha_interval}")

        self.alpha_demand = alpha_demand
        self.alpha_interval = alpha_interval

        # State variables
        self.demand_estimate: Optional[float] = None
        self.interval_estimate: Optional[float] = None
        self.last_demand_period: int = 0
        self.is_fitted: bool = False

        # Variance tracking for confidence intervals
        self._demand_variance: float = 0.0
        self._interval_variance: float = 0.0
        self._n_demands: int = 0

    def fit(self, series: np.ndarray) -> 'CrostonMethod':
        """
        Fit Croston's method on historical demand data.

        The fitting process:
        1. Identifies non-zero demand periods
        2. Initializes demand estimate with first non-zero demand
        3. Initializes interval estimate with first inter-arrival time
        4. Updates estimates using exponential smoothing for each demand

        Args:
            series: Array of historical demand values (can contain zeros)

        Returns:
            self: Fitted model instance

        Raises:
            ValueError: If series is empty or contains no positive demands
        """
        series = np.asarray(series, dtype=float)

        if len(series) == 0:
            raise ValueError("Cannot fit on empty series")

        # Find non-zero demand periods
        demand_periods = np.where(series > 0)[0]

        if len(demand_periods) == 0:
            # No positive demands - use fallback
            self.demand_estimate = 0.0
            self.interval_estimate = float(len(series))
            self.is_fitted = True
            return self

        # Extract demand sizes and inter-arrival times
        demand_sizes = series[demand_periods]

        if len(demand_periods) == 1:
            # Only one demand - use it as estimate
            self.demand_estimate = float(demand_sizes[0])
            self.interval_estimate = float(demand_periods[0] + 1)
            self.last_demand_period = len(series) - 1
            self.is_fitted = True
            self._n_demands = 1
            return self

        # Calculate inter-arrival times
        intervals = np.diff(demand_periods)

        # Initialize with first values
        self.demand_estimate = float(demand_sizes[0])
        self.interval_estimate = float(intervals[0]) if len(intervals) > 0 else 1.0

        # Store for variance calculation
        demand_list = [demand_sizes[0]]
        interval_list = [intervals[0]] if len(intervals) > 0 else [1.0]

        # Update estimates using exponential smoothing
        for i in range(1, len(demand_sizes)):
            # Update demand estimate
            self.demand_estimate = (
                self.alpha_demand * demand_sizes[i] +
                (1 - self.alpha_demand) * self.demand_estimate
            )
            demand_list.append(demand_sizes[i])

            # Update interval estimate (if we have interval for this demand)
            if i < len(intervals) + 1:
                interval = intervals[i - 1] if i - 1 < len(intervals) else intervals[-1]
                self.interval_estimate = (
                    self.alpha_interval * interval +
                    (1 - self.alpha_interval) * self.interval_estimate
                )
                interval_list.append(interval)

        # Calculate variance for confidence intervals
        if len(demand_list) > 1:
            self._demand_variance = float(np.var(demand_list))
            self._interval_variance = float(np.var(interval_list))

        self._n_demands = len(demand_sizes)
        self.last_demand_period = demand_periods[-1]
        self.is_fitted = True

        return self

    def get_rate_estimate(self) -> float:
        """
        Get the estimated demand rate (demand per period).

        Returns:
            Estimated average demand per period

        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting rate estimate")

        if self.interval_estimate is None or self.interval_estimate == 0:
            return 0.0

        return self.demand_estimate / self.interval_estimate

    def predict(self, horizon: int = 1) -> np.ndarray:
        """
        Generate forecasts for n periods ahead.

        Croston's method produces a flat (constant) forecast since it estimates
        a steady-state demand rate. The forecast value is:
            forecast = demand_estimate / interval_estimate

        Args:
            horizon: Number of periods to forecast

        Returns:
            Array of forecast values with shape (horizon,)

        Raises:
            RuntimeError: If model has not been fitted
            ValueError: If horizon is not positive
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        rate = self.get_rate_estimate()
        return np.full(horizon, rate)

    def predict_with_intervals(
        self,
        horizon: int = 1,
        confidence: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts with confidence intervals.

        Uses a simplified approach based on demand and interval variance.
        For sparse data, intervals may be wide to reflect uncertainty.

        Args:
            horizon: Number of periods to forecast
            confidence: Confidence level for intervals (default 0.95)

        Returns:
            Tuple of (point_forecast, lower_bound, upper_bound)
            Each is an array with shape (horizon,)

        Raises:
            RuntimeError: If model has not been fitted
            ValueError: If horizon is not positive or confidence not in (0, 1)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting")

        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        if not (0 < confidence < 1):
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")

        point_forecast = self.predict(horizon)

        # Calculate standard error of the rate estimate
        # Using delta method approximation for ratio variance
        if self.interval_estimate > 0 and self._n_demands > 2:
            # Coefficient of variation for demand and interval
            cv_demand = np.sqrt(self._demand_variance) / max(self.demand_estimate, 1e-6)
            cv_interval = np.sqrt(self._interval_variance) / max(self.interval_estimate, 1e-6)

            # Approximate CV of the ratio
            cv_rate = np.sqrt(cv_demand**2 + cv_interval**2)

            # Standard error
            std_error = point_forecast[0] * cv_rate
        else:
            # Not enough data - use demand estimate as proxy for uncertainty
            std_error = max(self.demand_estimate * 0.5, 0.5) if self.demand_estimate else 0.5

        # Calculate z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)

        # Calculate bounds
        margin = z_score * std_error
        lower_bound = np.maximum(point_forecast - margin, 0)
        upper_bound = point_forecast + margin

        return point_forecast, lower_bound, upper_bound

    def get_parameters(self) -> dict:
        """
        Get current model parameters and state.

        Returns:
            Dictionary containing:
                - alpha_demand: Demand smoothing parameter
                - alpha_interval: Interval smoothing parameter
                - demand_estimate: Current demand estimate
                - interval_estimate: Current interval estimate
                - rate_estimate: Demand per period
                - n_demands: Number of demand observations used
        """
        return {
            'alpha_demand': self.alpha_demand,
            'alpha_interval': self.alpha_interval,
            'demand_estimate': self.demand_estimate,
            'interval_estimate': self.interval_estimate,
            'rate_estimate': self.get_rate_estimate() if self.is_fitted else None,
            'n_demands': self._n_demands,
            'is_fitted': self.is_fitted
        }

    def __repr__(self) -> str:
        if self.is_fitted:
            return (
                f"CrostonMethod(alpha_demand={self.alpha_demand}, "
                f"alpha_interval={self.alpha_interval}, "
                f"rate={self.get_rate_estimate():.4f})"
            )
        return (
            f"CrostonMethod(alpha_demand={self.alpha_demand}, "
            f"alpha_interval={self.alpha_interval}, fitted=False)"
        )


class SBA(CrostonMethod):
    """
    Syntetos-Boylan Approximation (SBA) - bias-corrected Croston's method.

    The original Croston's method has a slight positive bias. SBA corrects
    this by applying a debiasing factor:
        forecast = (1 - alpha_interval/2) * (demand_estimate / interval_estimate)

    This correction improves accuracy for most intermittent demand patterns.

    Reference:
        Syntetos, A.A. & Boylan, J.E. (2001). "On the bias of intermittent
        demand estimates"
    """

    def get_rate_estimate(self) -> float:
        """Get bias-corrected demand rate."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting rate estimate")

        if self.interval_estimate is None or self.interval_estimate == 0:
            return 0.0

        # Apply SBA bias correction
        bias_correction = 1 - (self.alpha_interval / 2)
        return bias_correction * (self.demand_estimate / self.interval_estimate)


def select_croston_variant(
    series: np.ndarray,
    cv_threshold: float = 0.49,
    adi_threshold: float = 1.32
) -> str:
    """
    Select appropriate Croston variant based on demand characteristics.

    Uses the Syntetos-Boylan-Croston (SBC) classification scheme:
    - CV (Coefficient of Variation of demand size)
    - ADI (Average Demand Interval)

    Args:
        series: Historical demand data
        cv_threshold: CV threshold for classification (default 0.49)
        adi_threshold: ADI threshold for classification (default 1.32)

    Returns:
        Recommended method: 'croston', 'sba', or 'moving_average'
    """
    series = np.asarray(series)
    non_zero = series[series > 0]

    if len(non_zero) < 3:
        return 'moving_average'

    # Calculate CV of demand sizes
    cv = np.std(non_zero) / np.mean(non_zero) if np.mean(non_zero) > 0 else 0

    # Calculate ADI (average demand interval)
    demand_indices = np.where(series > 0)[0]
    if len(demand_indices) < 2:
        return 'moving_average'

    intervals = np.diff(demand_indices)
    adi = np.mean(intervals)

    # SBC classification
    if adi < adi_threshold:
        if cv < cv_threshold:
            return 'moving_average'  # Smooth demand
        else:
            return 'croston'  # Erratic demand
    else:
        if cv < cv_threshold:
            return 'croston'  # Intermittent demand
        else:
            return 'sba'  # Lumpy demand


if __name__ == '__main__':
    # Example usage
    print("Croston's Method Demo")
    print("=" * 50)

    # Simulated intermittent demand
    np.random.seed(42)
    demand = np.zeros(100)
    demand_points = np.random.choice(100, 15, replace=False)
    demand[demand_points] = np.random.poisson(5, 15)

    print(f"Historical data: {len(demand)} periods, {np.sum(demand > 0)} with demand")
    print(f"Mean demand: {np.mean(demand):.2f}")
    print(f"Mean when > 0: {np.mean(demand[demand > 0]):.2f}")

    # Fit and predict
    model = CrostonMethod(alpha_demand=0.1, alpha_interval=0.1)
    model.fit(demand)

    print(f"\nFitted parameters:")
    for k, v in model.get_parameters().items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Forecast with intervals
    forecast, lower, upper = model.predict_with_intervals(horizon=14, confidence=0.95)

    print(f"\n14-day forecast:")
    print(f"  Point estimate: {forecast[0]:.3f} per day")
    print(f"  95% CI: [{lower[0]:.3f}, {upper[0]:.3f}]")

    # SBA variant
    sba_model = SBA(alpha_demand=0.1, alpha_interval=0.1)
    sba_model.fit(demand)
    sba_forecast = sba_model.predict(14)

    print(f"\nSBA forecast: {sba_forecast[0]:.3f} per day")
    print(f"Bias correction applied: {1 - 0.1/2:.2f}")
