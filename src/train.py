"""
Model Training & Evaluation Module for Speech Disorder Detection
B.Tech Major Project - Dept. of Information Technology

Pipeline:
1. Load acoustic feature matrix (or build if not present)
2. Stratified 80/20 Train-Test split
3. Feature scaling via StandardScaler
4. Train 3 Classifiers:
   - Support Vector Machine (SVM with RBF Kernel & Probability Calibration)
   - Random Forest Classifier
   - Logistic Regression
5. Comprehensive Evaluation:
   - Accuracy, Precision, Recall, F1-Score (Macro & Weighted)
   - Confusion Matrix
6. Automated Best Model Selection & Persistence:
   - models/best_model.joblib
   - models/scaler.joblib
   - models/label_encoder.joblib
   - models/model_metadata.json
"""

from __future__ import annotations
import os
import sys
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    from src.feature_extraction import build_feature_dataset, FEATURE_NAMES
except ImportError:
    from feature_extraction import build_feature_dataset, FEATURE_NAMES


def train_models(
    data_dir: Union[str, Path] = "data",
    features_csv: Union[str, Path] = "data/features.csv",
    models_dir: Union[str, Path] = "models",
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes end-to-end model training, evaluation, comparison, and persistence.
    
    Returns:
        dict containing model benchmarks, best model name, and evaluation matrices.
    """
    models_path = Path(models_dir)
    features_path = Path(features_csv)
    data_path = Path(data_dir)

    models_path.mkdir(parents=True, exist_ok=True)

    # 1. Ensure features CSV exists; build if missing
    if not features_path.exists():
        print(f"Features file {features_path} not found. Building from {data_path}...")
        df = build_feature_dataset(data_dir=data_path, output_csv=features_path)
    else:
        df = pd.read_csv(features_path)

    if df.empty or "label" not in df.columns:
        raise ValueError(f"Feature dataset in {features_path} is empty or missing 'label' column.")

    # Check class distribution
    class_counts = df["label"].value_counts()
    print("\nDataset Class Distribution:")
    for cls_name, count in class_counts.items():
        print(f"  - {cls_name}: {count} samples")

    # Extract X (features) and y (labels)
    feature_cols = [col for col in FEATURE_NAMES if col in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].astype(str).str.lower().values

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(c) for c in label_encoder.classes_]

    # 2. Train/Test Split (Stratified if viable)
    can_stratify = (class_counts.min() >= 2) and (len(class_counts) > 1)
    stratify_param = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    print(f"\nSplit: Train={len(X_train)} samples | Test={len(X_test)} samples (Stratified={can_stratify})")

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save fitted scaler and label encoder
    scaler_path = models_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    encoder_path = models_path / "label_encoder.joblib"
    joblib.dump(label_encoder, encoder_path)

    # 4. Define candidate models
    classifiers = {
        "Support Vector Machine (SVM)": CalibratedClassifierCV(
            estimator=SVC(kernel="rbf", C=2.0, random_state=random_state),
            ensemble=False
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state
        )
    }

    results = {}
    best_model_name = None
    best_f1_score = -1.0
    best_model_obj = None

    print("\n" + "=" * 72)
    print(f"{'Model':<32} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("=" * 72)

    for name, clf in classifiers.items():
        # Train
        clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Compute metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cm = [[int(val) for val in row] for row in confusion_matrix(y_test, y_pred)]

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
        }

        print(f"{name:<32} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")

        # Track best model (primary metric: F1-score, secondary: Accuracy)
        if f1 > best_f1_score:
            best_f1_score = f1
            best_model_name = name
            best_model_obj = clf

    print("=" * 72)
    print(f"\n>> Best performing model: {best_model_name} (Weighted F1 = {best_f1_score:.4f})")

    # 5. Persist Best Model
    best_model_path = models_path / "best_model.joblib"
    joblib.dump(best_model_obj, best_model_path)
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print(f"Saved encoder to: {encoder_path}")

    # 6. Save Metadata & Metrics JSON
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "best_model": best_model_name,
        "classes": class_names,
        "feature_names": feature_cols,
        "num_features": len(feature_cols),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "models_benchmark": results
    }

    metadata_path = models_path / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    train_models()
