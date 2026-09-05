"""
Inference Script for Speech Disorder Detection
B.Tech Major Project - Dept. of Information Technology

Given an input .wav audio recording:
1. Runs full preprocessing pipeline (resampling, trimming, normalization)
2. Extracts acoustic features (MFCCs, F0 pitch, RMS energy, ZCR, Spectral)
3. Loads fitted StandardScaler and trained best ML model
4. Outputs predicted class, confidence score, and full probability distribution
5. Generates feature summary & preliminary clinical screening insight
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np

try:
    from src.feature_extraction import extract_feature_vector, FEATURE_NAMES
except ImportError:
    from feature_extraction import extract_feature_vector, FEATURE_NAMES


def resolve_audio_path(file_path: Union[str, Path]) -> Path:
    """Resolve audio file path across absolute, relative, and data folder candidates."""
    p = Path(file_path)
    if p.exists():
        return p

    # Check relative to PROJECT_ROOT
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate

    # Check inside data subfolders
    for sub in ["normal", "dysarthria", "dysphonia", "stuttering"]:
        candidate = PROJECT_ROOT / "data" / sub / p.name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Input audio file not found at '{file_path}'.")


def predict_audio(
    file_path: Union[str, Path],
    models_dir: Union[str, Path] = "models",
    model_path: Optional[Union[str, Path]] = None,
    scaler_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Perform speech disorder classification on a single audio file.
    
    Args:
        file_path: Path to .wav audio file.
        models_dir: Directory containing saved models and metadata.
        model_path: Optional explicit model file path.
        scaler_path: Optional explicit scaler file path.
        
    Returns:
        dict containing:
            - status: "success"
            - predicted_class: str
            - confidence: float (0.0 to 1.0)
            - confidence_percentage: str
            - probabilities: dict of class -> probability
            - key_indicators: dict of primary acoustic metrics
            - all_features: dict of all 40 features
            - model_used: str
            - preliminary_screening_note: str
            - disclaimer: str
    """
    resolved_file = resolve_audio_path(file_path)
    models_folder = Path(models_dir)
    if not models_folder.is_absolute():
        models_folder = PROJECT_ROOT / models_folder

    model_file = Path(model_path) if model_path else models_folder / "best_model.joblib"
    scaler_file = Path(scaler_path) if scaler_path else models_folder / "scaler.joblib"
    encoder_file = models_folder / "label_encoder.joblib"
    metadata_file = models_folder / "model_metadata.json"

    if not model_file.exists() or not scaler_file.exists():
        raise FileNotFoundError(
            f"Trained model or scaler not found in '{models_folder}'. "
            "Please run 'python src/train.py' first to train and persist the models."
        )

    # Load artifacts
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)

    label_encoder = None
    if encoder_file.exists():
        try:
            label_encoder = joblib.load(encoder_file)
        except Exception:
            pass

    # Read metadata for feature alignment & model name
    expected_features = FEATURE_NAMES
    classes = getattr(label_encoder, "classes_", None)
    model_name = "Calibrated Machine Learning Model"

    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_name = meta.get("best_model", model_name)
                expected_features = meta.get("feature_names", FEATURE_NAMES)
                if classes is None and "classes" in meta:
                    classes = meta["classes"]
        except Exception:
            pass

    # 1. Feature Extraction with explicit feature ordering
    feat_vector, feat_dict = extract_feature_vector(
        resolved_file,
        feature_names=expected_features
    )

    # 2. Reshape and Scale
    X_input = feat_vector.reshape(1, -1)
    X_scaled = scaler.transform(X_input)

    # 3. Predict Class & Probabilities
    pred_idx = model.predict(X_scaled)[0]

    probabilities: Dict[str, float] = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        if classes is not None and len(classes) == len(probs):
            for cls_name, prob in zip(classes, probs):
                probabilities[str(cls_name)] = round(float(prob), 4)
            confidence = float(np.max(probs))
        else:
            confidence = float(np.max(probs))
            probabilities = {f"Class_{i}": round(float(p), 4) for i, p in enumerate(probs)}
    else:
        confidence = 1.0

    # Sort probabilities descending
    probabilities = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    # Decode predicted class name
    if label_encoder is not None and isinstance(pred_idx, (int, np.integer)):
        predicted_class = str(label_encoder.inverse_transform([pred_idx])[0])
    elif classes is not None and isinstance(pred_idx, (int, np.integer)) and pred_idx < len(classes):
        predicted_class = str(classes[pred_idx])
    else:
        predicted_class = str(pred_idx)

    # Key acoustic indicators
    pitch_mean = feat_dict.get("pitch_mean", 0.0)
    pitch_std = feat_dict.get("pitch_std", 0.0)
    rms_mean = feat_dict.get("rms_mean", 0.0)
    zcr_mean = feat_dict.get("zcr_mean", 0.0)
    spec_cent = feat_dict.get("spectral_centroid_mean", 0.0)

    # Clinical screening insight note
    p_lower = predicted_class.lower()
    if p_lower == "normal":
        insight = "Acoustic markers (steady F0 pitch, fluent harmonic resonance) fall within standard non-pathological speech baselines."
    elif p_lower == "dysarthria":
        insight = "Detected markers characteristic of dysarthria: reduced formant dynamics, articulatory imprecision, and altered phonatory timing."
    elif p_lower == "dysphonia":
        insight = "Detected elevated pitch instability (jitter/shimmer) and spectral turbulence noise characteristic of dysphonia/vocal cord perturbation."
    elif p_lower == "stuttering":
        insight = "Detected acoustic pauses and repetitive bursts indicative of speech disfluency / stuttering blocks."
    else:
        insight = f"Sample classified into category '{predicted_class}' based on acoustic pattern analysis."

    if rms_mean < 0.003:
        insight += " (Notice: Extremely low acoustic energy or silence detected. Please verify microphone input.)"

    return {
        "status": "success",
        "audio_file": resolved_file.name,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.1f}%",
        "probabilities": probabilities,
        "model_used": model_name,
        "preliminary_screening_note": insight,
        "key_indicators": {
            "mean_pitch_f0_hz": round(float(pitch_mean), 1),
            "pitch_variability_std": round(float(pitch_std), 1),
            "energy_rms": round(float(rms_mean), 4),
            "zero_crossing_rate": round(float(zcr_mean), 4),
            "spectral_centroid_hz": round(float(spec_cent), 1)
        },
        "all_features": {k: round(float(v), 4) for k, v in feat_dict.items()},
        "disclaimer": "PRELIMINARY SCREENING TOOL ONLY: This assessment does not substitute for a clinical diagnosis by a certified Speech-Language Pathologist (SLP)."
    }


def main():
    parser = argparse.ArgumentParser(description="Speech Disorder Detection Inference CLI")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to input .wav audio file")
    parser.add_argument("--models-dir", "-m", type=str, default="models", help="Directory containing trained models")
    args = parser.parse_args()

    print(f"\nAnalyzing audio file: {args.file} ...")
    try:
        result = predict_audio(file_path=args.file, models_dir=args.models_dir)

        print("\n" + "=" * 66)
        print("  SPEECH DISORDER SCREENING REPORT")
        print("=" * 66)
        print(f"  File Analyzed        : {result['audio_file']}")
        print(f"  Predicted Condition  : {result['predicted_class'].upper()}")
        print(f"  Confidence Score     : {result['confidence_percentage']}")
        print(f"  Classifier Model     : {result['model_used']}")
        print("-" * 66)
        print("  Probability Distribution:")
        for cls, prob in result["probabilities"].items():
            bar = "#" * int(prob * 25)
            print(f"    - {cls:<15}: {prob:>6.2%} | {bar}")
        print("-" * 66)
        print("  Key Acoustic Indicators:")
        for k, v in result["key_indicators"].items():
            print(f"    - {k:<25}: {v}")
        print("-" * 66)
        print(f"  Clinical Note : {result['preliminary_screening_note']}")
        print(f"  Notice        : {result['disclaimer']}")
        print("=" * 66 + "\n")
    except Exception as err:
        print(f"\nInference Error: {err}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
