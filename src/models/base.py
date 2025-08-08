from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Optional

import pandas as pd


@dataclass
class RegimeResult:
    labels: pd.Series  # int labels indexed by date
    proba: Optional[pd.DataFrame]  # columns = state_i probabilities
    diagnostics: Dict


class RegimeModel(Protocol):
    def fit(self, X: pd.DataFrame, **kwargs) -> RegimeResult: ...


