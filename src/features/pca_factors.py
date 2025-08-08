from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def fit_theme_pca(
    df: pd.DataFrame, theme_cols: List[str], n_components: int = 1
) -> Tuple[pd.Series, Dict[str, object]]:
    """Fit PCA on given theme columns and return PC1 and model params.

    - Standardizes inputs using column means/standard deviations
    - Fills missing values with column means (computed over available data)
    - Returns the first principal component as a pd.Series aligned to df.index
    - Returns a dict of model parameters (components, explained variance, means, scales)
    - Silently attempts to persist parameters via FeatureStore if available
    """
    from sklearn.decomposition import PCA

    cols = [c for c in theme_cols if c in df.columns]
    if not cols:
        return pd.Series(index=df.index, dtype=float, name="PC1"), {}

    X = df[cols].astype(float).copy()
    # compute column means/std ignoring NaN
    col_means = X.mean(axis=0)
    col_stds = X.std(axis=0, ddof=0).replace(0.0, np.nan)
    # fill missing with means, then standardize
    X_filled = X.fillna(col_means)
    X_std = (X_filled - col_means) / col_stds
    X_std = X_std.fillna(0.0)

    n_components = max(1, int(n_components))
    pca = PCA(n_components=n_components, random_state=42)
    try:
        T = pca.fit_transform(X_std.values)
    except Exception as exc:
        logger.warning("PCA failed for columns %s: %s", cols, exc)
        return pd.Series(index=df.index, dtype=float, name="PC1"), {}

    pc1 = pd.Series(T[:, 0], index=df.index, name="PC1")

    params: Dict[str, object] = {
        "components_": getattr(pca, "components_", None),
        "explained_variance_ratio_": getattr(pca, "explained_variance_ratio_", None),
        "mean_": col_means.to_dict(),
        "scale_": col_stds.to_dict(),
        "columns": cols,
        "n_components": n_components,
    }

    # Best-effort persistence via FeatureStore if available
    try:
        from src.features.feature_store import create_file_feature_store, FeatureType

        store = create_file_feature_store()
        store.save_feature(
            name="theme_pca_model_params",
            data=params,
            description="Parameters for theme PCA transformation",
            feature_type=FeatureType.OBJECT,
            tags=["pca", "theme"],
        )
    except Exception:
        # Optional; do not fail pipeline if the store is not configured
        pass

    return pc1, params


