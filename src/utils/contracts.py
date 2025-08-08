import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Pattern, Sequence, Tuple, Type

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameContract:
    """Lightweight dataframe contract.

    This intentionally avoids external deps (e.g., pydantic) while providing
    a clear place to declare expected columns and basic shape constraints.
    """

    name: str
    # Fixed columns that should exist when applicable
    required_columns: Tuple[str, ...] = ()
    # Columns that are nice-to-have; missing ones only produce warnings
    optional_columns: Tuple[str, ...] = ()
    # Regex patterns for optional columns (e.g., probability columns)
    optional_patterns: Tuple[Pattern[str], ...] = ()
    # Whether a datetime index should be present instead of a 'date' column
    require_datetime_index: bool = True
    # If True, either datetime index OR a 'date' column is acceptable
    allow_date_column: bool = True
    # Allows contract-specific checks beyond simple column presence
    def extra_checks(self, df: pd.DataFrame) -> List[str]:
        return []


class ProcessedMacroFrame(FrameContract):
    def __init__(self) -> None:
        super().__init__(
            name="ProcessedMacroFrame",
            required_columns=(),  # flexible macro features; index should be datetime
            optional_columns=(
                # Theme composites if/when present (not required yet)
                "F_Growth",
                "F_Inflation",
                "F_Liquidity",
                "F_CreditRisk",
                "F_Housing",
                "F_External",
            ),
            optional_patterns=(),
            require_datetime_index=True,
            allow_date_column=True,
        )


class RegimeFrame(FrameContract):
    def __init__(self) -> None:
        super().__init__(
            name="RegimeFrame",
            # date covered via datetime index; require rule regime column name per spec
            required_columns=("Rule_Regime",),
            optional_columns=("KMeans_Regime", "HMM_Regime", "Regime_Ensemble"),
            optional_patterns=(re.compile(r".*_Prob_.*"),),
            require_datetime_index=True,
            allow_date_column=True,
        )


class PerformanceFrame(FrameContract):
    def __init__(self) -> None:
        super().__init__(
            name="PerformanceFrame",
            # Shape may be time-series (date index + asset return cols + regime label)
            # or aggregated-by-regime (regimes as index, MultiIndex columns per asset/metric).
            required_columns=(),
            optional_columns=(),
            optional_patterns=(),
            # We allow non-datetime index because aggregated outputs are by regime, not by date
            require_datetime_index=False,
            allow_date_column=True,
        )

    def extra_checks(self, df: pd.DataFrame) -> List[str]:
        issues: List[str] = []

        # Case A: Aggregated by regime (e.g., index are regime labels)
        if df.index.dtype == object or df.index.dtype == "O":
            # Likely aggregated table; ensure columns exist and are not empty
            if df.empty:
                issues.append("PerformanceFrame appears aggregated by regime but is empty.")
            return issues

        # Case B: Time series performance (date-like index or date column)
        # - Expect at least one regime label column and at least one numeric asset return column
        regime_like = [c for c in df.columns if "Regime" in str(c)]
        asset_like = [c for c in df.columns if c not in regime_like]
        asset_like = [c for c in asset_like if pd.api.types.is_numeric_dtype(df[c])]

        if len(regime_like) == 0:
            issues.append("No regime label columns found (expected a column containing 'Regime').")
        if len(asset_like) == 0:
            issues.append("No numeric asset return columns found.")

        return issues


def _missing_fixed_columns(
    df_columns: Sequence[str], expected: Iterable[str]
) -> List[str]:
    df_cols = set(map(str, df_columns))
    return [col for col in expected if str(col) not in df_cols]


def _present_optional_patterns(df_columns: Sequence[str], patterns: Tuple[Pattern[str], ...]) -> List[str]:
    df_cols = list(map(str, df_columns))
    found: List[str] = []
    for pat in patterns:
        found.extend([c for c in df_cols if pat.search(c)])
    return found


def validate_frame(
    df: pd.DataFrame,
    model_cls: Type[FrameContract],
    *,
    validate: bool = True,
    where: Optional[str] = None,
) -> None:
    """Validate a DataFrame against a declared frame contract.

    Args:
        df: The frame to validate
        model_cls: A subclass of FrameContract (e.g., ProcessedMacroFrame)
        validate: If True, raise on errors; if False, only log warnings
        where: Optional context string to include in messages

    Behavior:
        - Checks datetime index vs date column depending on the contract
        - Reports missing required columns
        - Notes optional columns/patterns not present (as warnings)
        - Runs contract-specific extra checks
    """
    contract = model_cls()
    context = f" in {where}" if where else ""

    problems: List[str] = []
    warnings_list: List[str] = []

    # Date / index expectations
    if contract.require_datetime_index:
        has_dt_index = isinstance(df.index, pd.DatetimeIndex)
        has_date_col = "date" in df.columns
        if not has_dt_index:
            if contract.allow_date_column and has_date_col:
                # best-effort check type
                if not pd.api.types.is_datetime64_any_dtype(df["date"]) :
                    warnings_list.append("Column 'date' exists but is not datetime-like.")
            else:
                problems.append("Expected DatetimeIndex or a 'date' column.")

    # Fixed required columns (ignore 'date' if index satisfies requirement)
    missing_required = _missing_fixed_columns(df.columns, contract.required_columns)
    if missing_required:
        problems.append(
            f"Missing required columns: {missing_required}. Present: {list(df.columns)[:15]}..."
        )

    # Optional columns and patterns - warn when none of them exist
    opt_missing = _missing_fixed_columns(df.columns, contract.optional_columns)
    if len(opt_missing) == len(contract.optional_columns) and len(contract.optional_columns) > 0:
        warnings_list.append(
            f"Optional thematic columns not found (not required yet): {list(contract.optional_columns)}"
        )

    matched_optional = _present_optional_patterns(df.columns, contract.optional_patterns)
    if contract.optional_patterns and not matched_optional:
        pattern_strs = [p.pattern for p in contract.optional_patterns]
        warnings_list.append(
            f"No columns matched optional patterns {pattern_strs}."
        )

    # Contract-specific checks
    problems.extend(contract.extra_checks(df))

    if problems:
        message = f"{contract.name}{context} validation failed: " + "; ".join(problems)
        if validate:
            raise ValueError(message)
        logger.warning(message)

    for w in warnings_list:
        logger.warning("%s%s: %s", contract.name, context, w)

    # No return (side-effect logging / raising only)
    return None


__all__ = [
    "FrameContract",
    "ProcessedMacroFrame",
    "RegimeFrame",
    "PerformanceFrame",
    "validate_frame",
]


