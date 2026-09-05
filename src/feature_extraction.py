"""
Feature Extraction Module for Speech Disorder Detection
B.Tech Major Project - Dept. of Information Technology

Extracts comprehensive acoustic and clinical speech features:
1. Mel-Frequency Cepstral Coefficients (MFCCs): 13 coefficients (mean & std across frames = 26 features)
2. Pitch / Fundamental Frequency (F0): mean, std, min, max
3. Energy / RMS (Root Mean Square): mean, std
4. Zero-Crossing Rate (ZCR): mean, std
5. Spectral Centroid: mean, std
6. Spectral Rolloff: mean, std
7. Spectral Bandwidth: mean, std

Exports:
- Dictionary of named features
- Feature vectors for ML inference
- Full dataset feature matrix CSV builder
"""

from __future__ import annotations
import os
import sys
import glob
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import librosa

try:
    from src.preprocessing import preprocess_audio
except ImportError:
    from preprocessing import preprocess_audio


# Canonical ordered list of 40 extracted feature column names
FEATURE_NAMES: List[str] = []
for i in range(1, 14):
    FEATURE_NAMES.extend([f"mfcc_{i}_mean", f"mfcc_{i}_std"])
FEATURE_NAMES.extend([
    "pitch_mean", "pitch_std", "pitch_min", "pitch_max",
    "rms_mean", "rms_std",
    "zcr_mean", "zcr_std",
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_rolloff_mean", "spectral_rolloff_std",
    "spectral_bandwidth_mean", "spectral_bandwidth_std"
])


def extract_pitch_f0(
    y: np.ndarray,
    sr: int = 16000,
    fmin: float = 65.0,
    fmax: float = 400.0
) -> Dict[str, float]:
    """
    Extract fundamental frequency (F0/pitch) using STFT parabolic peak interpolation (piptrack).
    Accurately identifies prominent harmonic frequencies across speech frames.
    """
    default_res = {
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "pitch_min": 0.0,
        "pitch_max": 0.0
    }

    if len(y) < 512:
        return default_res

    try:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax)
        if pitches.shape[1] == 0:
            return default_res

        # Select pitch corresponding to the peak magnitude per frame
        pitch_indices = magnitudes.argmax(axis=0)
        pitch_values = pitches[pitch_indices, np.arange(pitches.shape[1])]
        valid_f0 = pitch_values[(pitch_values >= fmin) & (pitch_values <= fmax)]
        
        if len(valid_f0) > 0:
            return {
                "pitch_mean": float(np.mean(valid_f0)),
                "pitch_std": float(np.std(valid_f0)),
                "pitch_min": float(np.min(valid_f0)),
                "pitch_max": float(np.max(valid_f0))
            }
    except Exception:
        pass

    return default_res


def extract_features_from_audio(y: np.ndarray, sr: int = 16000) -> Dict[str, float]:
    """
    Extract 40 acoustic features from a preprocessed audio signal.
    
    Args:
        y: Normalized, trimmed 1D audio array.
        sr: Sampling rate (default 16000).
        
    Returns:
        dict mapping feature_name -> float value
    """
    features: Dict[str, float] = {}

    # Safe FFT size for short signals
    n_fft = min(len(y), 2048)
    hop_length = 512

    # 1. Mel-Frequency Cepstral Coefficients (MFCCs: 13 coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
    for idx in range(13):
        coeff_num = idx + 1
        features[f"mfcc_{coeff_num}_mean"] = float(np.mean(mfccs[idx]))
        features[f"mfcc_{coeff_num}_std"] = float(np.std(mfccs[idx]))

    # 2. Pitch / Fundamental Frequency (F0)
    pitch_dict = extract_pitch_f0(y, sr=sr)
    features.update(pitch_dict)

    # 3. Energy / Root Mean Square (RMS)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"] = float(np.std(rms))

    # 4. Zero-Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"] = float(np.std(zcr))

    # 5. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features["spectral_centroid_mean"] = float(np.mean(spec_cent))
    features["spectral_centroid_std"] = float(np.std(spec_cent))

    # 6. Spectral Rolloff (85% energy roll-off)
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)
    features["spectral_rolloff_mean"] = float(np.mean(spec_rolloff))
    features["spectral_rolloff_std"] = float(np.std(spec_rolloff))

    # 7. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features["spectral_bandwidth_mean"] = float(np.mean(spec_bw))
    features["spectral_bandwidth_std"] = float(np.std(spec_bw))

    # Clean any accidental NaNs or Infs
    for k, v in features.items():
        if np.isnan(v) or np.isinf(v):
            features[k] = 0.0

    return features


def extract_feature_vector(
    file_path: Union[str, Path],
    target_sr: int = 16000,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Given a path to a .wav file, preprocess it and return the ordered 1D numpy feature vector
    along with the feature dictionary.
    
    Args:
        file_path: Path to .wav audio.
        target_sr: Target sampling rate.
        feature_names: Optional explicit list of feature names to order the vector.
        
    Returns:
        tuple of (vector_1d_numpy, features_dict)
    """
    y, sr = preprocess_audio(file_path, target_sr=target_sr)
    feat_dict = extract_features_from_audio(y, sr=sr)
    
    order = feature_names if feature_names is not None else FEATURE_NAMES
    vector = np.array([feat_dict.get(name, 0.0) for name in order], dtype=np.float32)
    return vector, feat_dict


def build_feature_dataset(data_dir: Union[str, Path] = "data", output_csv: Union[str, Path] = "data/features.csv") -> pd.DataFrame:
    """
    Scans data_dir for class subdirectories (e.g. normal, dysarthria, dysphonia, stuttering),
    extracts features from all .wav files, and saves the matrix to a CSV file.
    
    Args:
        data_dir: Root dataset folder containing class subdirectories.
        output_csv: Path to save the extracted feature matrix.
        
    Returns:
        pd.DataFrame containing feature columns + 'label' + 'filename'
    """
    data_path = Path(data_dir)
    csv_path = Path(output_csv)
    print(f"Scanning dataset in: {data_path.resolve()}")
    
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"Created empty directory: {data_path.resolve()}")

    records = []
    # Discover class directories sorted alphabetically
    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    for class_folder in class_dirs:
        label = class_folder.name.lower()
        # Find all .wav and .WAV files
        wav_files = [f for f in class_folder.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]
        print(f"Processing class '{label}': {len(wav_files)} files found.")
        
        for wav_path in wav_files:
            try:
                _, feat_dict = extract_feature_vector(wav_path)
                feat_dict["label"] = label
                feat_dict["filename"] = wav_path.name
                records.append(feat_dict)
            except Exception as err:
                print(f"Warning: Error extracting features from {wav_path}: {err}")

    df = pd.DataFrame(records)
    if not df.empty:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Feature matrix saved successfully: {csv_path} ({len(df)} samples, {len(FEATURE_NAMES)} features)")
    else:
        print("No audio samples were processed. Feature dataset is empty.")

    return df


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        data_p = sys.argv[2] if len(sys.argv) > 2 else "data"
        build_feature_dataset(data_dir=data_p)
    elif len(sys.argv) > 1:
        test_wav = sys.argv[1]
        print(f"Extracting features from: {test_wav}")
        vec, feats = extract_feature_vector(test_wav)
        print(f"Extracted {len(vec)} features successfully.")
        print("Sample features:", {k: round(v, 4) for k, v in list(feats.items())[:6]})
    else:
        print("Feature Extraction module ready.")
        print("Usage:")
        print("  python feature_extraction.py <sample.wav>")
        print("  python feature_extraction.py --build [data_dir]")
