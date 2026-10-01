"""
SkyGuard AI — explainability helpers

Turns raw SHAP feature contributions into the plain-English "WHY?" reasons
shown on the dashboard alert card.

Usage:
    from src.explain import humanize_reason, explain_row
    reasons = explain_row(row_features_df, explainer, feature_cols)
"""

import numpy as np
import pandas as pd
from src.config import LABEL_MAP


def humanize_reason(feature_name: str, actual_value: float):
    """Turn a raw feature name + value into a plain-English reason, or None
    if the value isn't meaningful enough to show (e.g. trivial short streaks)."""
    if 'max_abs_zscore' in feature_name:
        return f"Reading is {actual_value:.1f}\u03c3 away from this station's own recent history"
    if 'max_abs_spatial_zscore' in feature_name:
        return f"Reading disagrees sharply with nearby stations ({actual_value:.1f}\u03c3)"
    if 'unchanged_streak' in feature_name:
        if actual_value < 3:
            return None
        return f"Value has stayed identical for {int(actual_value)} consecutive readings \u2014 possible frozen sensor"
    if 'roll_std3' in feature_name:
        return f"Almost no variation in the last 3 readings ({actual_value:.2f}) \u2014 possible frozen sensor"
    if 'zscore' in feature_name:
        base = feature_name.replace('_zscore', '')
        label = LABEL_MAP.get(base, base)
        return f"{label.capitalize()} is {actual_value:.1f} standard deviations from normal for this station"
    if 'spatial_dev' in feature_name:
        base = feature_name.replace('_spatial_dev', '').replace('_zscore', '')
        label = LABEL_MAP.get(base, base)
        direction = "higher than" if actual_value > 0 else "lower than"
        return f"{label.capitalize()} is {direction} nearby stations by {abs(actual_value):.1f}"
    if 'diff1' in feature_name:
        base = feature_name.replace('_diff1', '')
        label = LABEL_MAP.get(base, base)
        return f"{label.capitalize()} changed by {actual_value:.1f} from the previous reading"
    if 'swing' in feature_name:
        return f"Unusual {feature_name.replace('_', ' ')}: {actual_value:.1f}"
    return f"{feature_name}: {actual_value:.2f}"


def explain_row(X_row: pd.DataFrame, explainer, feature_cols, top_n: int = 5):
    """
    X_row: a single-row DataFrame with exactly `feature_cols` as columns.
    Returns a list of human-readable reason strings, strongest first.
    """
    shap_values = explainer.shap_values(X_row)

    if isinstance(shap_values, list):
        shap_row = shap_values[1][0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_row = shap_values[0, :, 1]
    else:
        shap_row = shap_values[0]

    contributions = pd.DataFrame({
        'feature': feature_cols,
        'actual_value': X_row.iloc[0].values,
        'shap_value': shap_row,
    })
    top = contributions[contributions['shap_value'] > 0].sort_values(
        'shap_value', ascending=False
    ).head(top_n)

    reasons = []
    for _, r in top.iterrows():
        reason = humanize_reason(r['feature'], r['actual_value'])
        if reason:
            reasons.append(reason)
    return reasons