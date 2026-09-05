"""
Audio Preprocessing Module for Speech Disorder Detection
B.Tech Major Project - Dept. of Information Technology

Handles:
- Loading audio files (.wav) with format/channel validation
- Resampling to standard sampling rate (default: 16,000 Hz)
- Silence trimming using energy threshold
- Peak amplitude normalization
- Padding short audio signals for safe STFT/spectral processing
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Union
import numpy as np
import soundfile as sf
import librosa


import subprocess


def convert_to_pcm_wav(input_path: Union[str, Path], output_path: Union[str, Path], target_sr: int = 16000) -> bool:
    """
    Convert any non-standard or compressed audio format (WebM, Opus, Ogg, MP3, AAC, etc.)
    into a standardized 16kHz mono 16-bit linear PCM WAV container.
    """
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe, "-y", "-i", str(input_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(target_sr),
            str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return res.returncode == 0 and os.path.exists(str(output_path)) and os.path.getsize(str(output_path)) > 44
    except Exception:
        return False


def load_audio(file_path: Union[str, Path], target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, and resample to target_sr.
    Automatically handles WebM, Opus, Ogg, MP3, or raw WAV audio.
    
    Args:
        file_path: Path to the audio file.
        target_sr: Desired sampling rate (default: 16000 Hz).
        
    Returns:
        tuple of (audio_array, sampling_rate)
    """
    str_path = str(file_path)
    if not os.path.exists(str_path):
        raise FileNotFoundError(f"Audio file not found at: {str_path}")

    # Check if file has a valid RIFF WAV header
    is_riff_wav = False
    try:
        with open(str_path, 'rb') as f:
            header = f.read(12)
            if header.startswith(b'RIFF') and b'WAVE' in header:
                is_riff_wav = True
    except Exception:
        pass

    # If not a standard RIFF WAV, or if corrupted container, convert using ffmpeg
    if not is_riff_wav:
        converted_wav = str_path + ".converted.wav"
        success = convert_to_pcm_wav(str_path, converted_wav, target_sr=target_sr)
        if success:
            str_path = converted_wav

    try:
        # Load with soundfile
        data, sr_orig = sf.read(str_path)
        if data.ndim > 1:
            data = np.mean(data, axis=-1)
        if sr_orig != target_sr:
            data = librosa.resample(data.astype(np.float32), orig_sr=sr_orig, target_sr=target_sr)
        y, sr = data, target_sr
    except Exception:
        # Fallback to librosa.load
        try:
            y, sr = librosa.load(str_path, sr=target_sr, mono=True)
        except Exception:
            # Last resort conversion attempt
            converted_wav = str_path + ".fallback.wav"
            if convert_to_pcm_wav(str_path, converted_wav, target_sr=target_sr):
                data, sr_orig = sf.read(converted_wav)
                if data.ndim > 1:
                    data = np.mean(data, axis=-1)
                y, sr = data, target_sr
            else:
                raise RuntimeError(f"Failed to decode audio format for {str_path}")

    # Ensure float32 dtype
    y = np.asarray(y, dtype=np.float32)

    # If audio is excessively short (< 2048 samples = 0.128s at 16kHz), zero-pad for STFT stability
    if len(y) < 2048:
        pad_width = 2048 - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')

    return y, sr


def trim_silence(y: np.ndarray, top_db: float = 25.0) -> np.ndarray:
    """
    Trim leading and trailing silence from an audio signal.
    
    Args:
        y: 1D audio signal array.
        top_db: The threshold (in decibels) below reference to consider as silence.
        
    Returns:
        Trimmed audio array.
    """
    if len(y) == 0:
        return y

    try:
        trimmed_y, _ = librosa.effects.trim(y, top_db=top_db)
        # If trimming resulted in an empty or excessively short signal (< 2048 samples), retain original
        if len(trimmed_y) < 2048:
            return y
        return trimmed_y
    except Exception:
        return y


def normalize_amplitude(y: np.ndarray) -> np.ndarray:
    """
    Normalize audio signal amplitude to peak range [-1.0, 1.0].
    
    Args:
        y: 1D audio signal array.
        
    Returns:
        Normalized audio array.
    """
    if len(y) == 0:
        return y

    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        return y / max_val
    return y


def preprocess_audio(file_path: Union[str, Path], target_sr: int = 16000, top_db: float = 25.0) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing pipeline: load, resample, trim silence, and normalize.
    
    Args:
        file_path: Path to .wav audio.
        target_sr: Target sample rate in Hz.
        top_db: Silence trimming threshold in dB.
        
    Returns:
        tuple of (preprocessed_signal, sample_rate)
    """
    y, sr = load_audio(file_path, target_sr=target_sr)
    y_trimmed = trim_silence(y, top_db=top_db)
    y_norm = normalize_amplitude(y_trimmed)
    return y_norm, sr


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Preprocessing test on: {test_file}")
        audio, rate = preprocess_audio(test_file)
        print(f"Success: Sample Rate={rate}Hz, Duration={len(audio)/rate:.2f}s, Peak={np.max(np.abs(audio)):.2f}")
    else:
        print("Preprocessing module ready. Usage: python preprocessing.py <audio_path.wav>")
