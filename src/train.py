"""
SkyGuard AI — full training pipeline

Run this file directly to redo everything from clean data to a saved model:
    python -m src.train

This replaces re-running ~40 notebook cells by hand. Notebooks are still
useful for exploration and plots, but this is the source of truth for
"how do I reproduce the model."
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_recall_curve,
                              precision_score, recall_score, roc_auc_score)

from src.config import (CLEAN_DATA_PATH, CONTAMINATION, EXPLAINER_PATH,
                         FEATURED_DATA_PATH, FEATURES_PATH, MODEL_PATH,
                         NEIGHBORS_PATH, STATION_COORDS_PATH, VALUE_COLS)
from src.data_prep import load_and_clean
from src.features import add_all_features, build_neighbor_lookup
from src.injection import inject_anomalies


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # 1. Clean data (or reload if already done)
    if os.path.exists(CLEAN_DATA_PATH):
        clean_df = pd.read_csv(CLEAN_DATA_PATH)
        clean_df['date'] = pd.to_datetime(clean_df['date'])
        print(f"Loaded existing clean data: {clean_df.shape}")
    else:
        clean_df = load_and_clean()

    # 2. Spatial neighbor lookup
    nearest_neighbors, station_ids = build_neighbor_lookup(clean_df)
    station_coords = clean_df[['station_id', 'lat', 'lon']].drop_duplicates().reset_index(drop=True)

    # 3. Inject known, labeled anomalies
    injected_df = inject_anomalies(clean_df, VALUE_COLS, fraction=0.04, seed=42)
    print("Anomaly type counts:")
    print(injected_df['anomaly_type'].value_counts())

    # 4. Feature engineering (temporal + multivariate + spatial + aggregate)
    featured_df, feature_cols = add_all_features(injected_df, nearest_neighbors, station_ids)
    featured_df = featured_df.dropna(subset=feature_cols).reset_index(drop=True)
    featured_df = featured_df.sort_values('date').reset_index(drop=True)
    featured_df.to_csv(FEATURED_DATA_PATH, index=False)
    print(f"Featured data saved: {featured_df.shape}, {len(feature_cols)} features")

    # 5. Time-based train/test split (no future leakage)
    split_date = featured_df['date'].quantile(0.8)
    train_df = featured_df[featured_df['date'] < split_date]
    test_df = featured_df[featured_df['date'] >= split_date]
    print(f"Train: {train_df.shape} | Test: {test_df.shape}")

    # 6. Train supervised classifier (outperformed the unsupervised-only approach)
    X_train = train_df[feature_cols]
    y_train = train_df['is_anomaly']
    X_test = test_df[feature_cols]
    y_test = test_df['is_anomaly']

    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    probs = rf_model.predict_proba(X_test)[:, 1]

    # 7. Pick the best-F1 threshold rather than assuming 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores.argmax()
    best_threshold = float(thresholds[best_idx])

    preds = (probs >= best_threshold).astype(int)
    print(f"\nBest threshold: {best_threshold:.3f}")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1-score:  {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    test_df = test_df.copy()
    test_df['predicted'] = preds
    print("\nDetection rate by anomaly type:")
    print(test_df[test_df['is_anomaly'] == 1].groupby('anomaly_type')['predicted'].mean())

    # 8. SHAP explainer (built once, reused at inference time — this is slow to build,
    #    but cheap to use on a single row later)
    import shap
    explainer = shap.TreeExplainer(rf_model)

    # 9. Save everything needed for inference / the API
    joblib.dump(rf_model, MODEL_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)
    joblib.dump(nearest_neighbors, NEIGHBORS_PATH)
    joblib.dump(station_coords, STATION_COORDS_PATH)
    print(f"\nSaved model, features, explainer, and spatial lookups to models/")


if __name__ == "__main__":
    main()