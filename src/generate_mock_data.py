"""
Synthetic Audio Dataset Generator for Speech Disorder Detection
B.Tech Major Project - Dept. of Information Technology

Generates realistic mock acoustic speech waveforms (.wav) for:
1. normal: Harmonic vowels, stable pitch F0 (120-220 Hz), low jitter, fluent cadence
2. dysarthria: Monotone/erratic pitch, slower speech rate, reduced formant dynamics
3. dysphonia: Hoarse/breathy acoustics, elevated pitch jitter, amplitude shimmer, noise
4. stuttering: Syllable repetition bursts, sudden phonatory blocks, silence interruptions

Used to validate the entire preprocessing, feature extraction, training,
and inference pipeline immediately before plugging in clinical or benchmark datasets
(such as Saarbruecken Voice Database, TORGO, or UASpeech).
"""

import os
import numpy as np
import soundfile as sf


def generate_vowel_formants(t: np.ndarray, f0: float, formants: list[float], bandwidths: list[float]) -> np.ndarray:
    """Simulate vocal tract resonance using a source-filter model."""
    # Glottal pulse train approximation
    source = np.sin(2 * np.pi * f0 * t)
    for harmonic in range(2, 6):
        source += (1.0 / harmonic) * np.sin(2 * np.pi * (harmonic * f0) * t)

    # Apply formant filtering via damped resonances
    filtered = np.zeros_like(t)
    for f, bw in zip(formants, bandwidths):
        decay = np.exp(-np.pi * bw * (t % (1.0 / max(f0, 1.0))))
        resonance = np.sin(2 * np.pi * f * t) * decay
        filtered += resonance

    combined = 0.5 * source + 0.5 * filtered
    return combined


def generate_normal_sample(duration: float = 3.0, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate healthy, fluent speech sample with steady harmonics."""
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Healthy base pitch with gentle human prosody contour
    f0_base = np.random.uniform(130, 200)
    pitch_contour = f0_base + 12 * np.sin(2 * np.pi * 1.5 * t)
    
    # Speech envelope (vowel syllables with natural pauses)
    envelope = np.maximum(0.0, np.sin(2 * np.pi * 2.5 * t)) ** 2
    
    audio = np.sin(2 * np.pi * pitch_contour * t)
    for h in [2, 3, 4]:
        audio += (0.6 / h) * np.sin(2 * np.pi * (h * pitch_contour) * t)
    
    # Formants for English vowel /a/ or /i/
    f1, f2 = 700.0, 1200.0
    audio += 0.4 * np.sin(2 * np.pi * f1 * t) + 0.2 * np.sin(2 * np.pi * f2 * t)
    
    # Apply envelope and light acoustic room noise
    audio = audio * envelope
    noise = np.random.normal(0, 0.015, len(t))
    signal = audio + noise
    
    # Normalize
    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_dysarthria_sample(duration: float = 3.5, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate dysarthric speech sample: slow, slurred, monotone/strained."""
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Flat, monotone pitch with sudden tremors
    f0_base = np.random.uniform(110, 150)
    tremor = 4.0 * np.sin(2 * np.pi * 6.0 * t)  # 6 Hz neurological tremor
    pitch_contour = f0_base + tremor
    
    # Prolonged, slow envelope articulation (0.9 Hz vs normal 2.5 Hz)
    envelope = np.maximum(0.0, np.sin(2 * np.pi * 0.9 * t)) ** 1.5
    
    audio = np.sin(2 * np.pi * pitch_contour * t)
    for h in [2, 3]:
        audio += (0.4 / h) * np.sin(2 * np.pi * (h * pitch_contour) * t)
        
    # Imprecise formants with low spectral contrast
    audio += 0.25 * np.sin(2 * np.pi * 500.0 * t)
    
    signal = (audio * envelope) + np.random.normal(0, 0.02, len(t))
    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_dysphonia_sample(duration: float = 3.0, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate dysphonic speech sample: hoarse, breathy, high jitter & shimmer."""
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Severe pitch jitter (frequency instability)
    f0_base = np.random.uniform(140, 210)
    jitter = np.random.normal(0, 28.0, len(t))  # high jitter
    pitch_contour = np.clip(f0_base + jitter, 80, 350)
    
    # Amplitude shimmer (erratic amplitude fluctuations)
    shimmer = 0.5 + 0.5 * np.random.uniform(0.4, 1.0, len(t))
    
    envelope = np.maximum(0.0, np.sin(2 * np.pi * 2.0 * t))
    
    harmonics = np.sin(2 * np.pi * pitch_contour * t) + 0.3 * np.sin(2 * np.pi * 2 * pitch_contour * t)
    
    # High breathiness / unvoiced turbulent noise (elevates ZCR and bandwidth)
    breath_noise = np.random.normal(0, 0.18, len(t))
    
    signal = (harmonics * envelope * shimmer) + breath_noise
    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_stuttering_sample(duration: float = 3.5, sr: int = 16000, seed: int = 42) -> np.ndarray:
    """Generate stuttering sample: syllable repetitions and block silences."""
    np.random.seed(seed)
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    f0_base = np.random.uniform(130, 190)
    pitch = f0_base + 10 * np.sin(2 * np.pi * 1.5 * t)
    
    # Create rapid repetitive bursts (e.g. 7 Hz repetition bursts) followed by silent blocks
    burst_pattern = (np.sin(2 * np.pi * 8.0 * t) > 0.3).astype(float)
    # Silent block during second second
    block_mask = np.ones_like(t)
    block_start = int(sr * 1.2)
    block_end = int(sr * 2.0)
    block_mask[block_start:block_end] = 0.05  # Blocked silence
    
    audio = np.sin(2 * np.pi * pitch * t) + 0.5 * np.sin(2 * np.pi * 2 * pitch * t)
    signal = audio * burst_pattern * block_mask + np.random.normal(0, 0.02, len(t))
    return signal / (np.max(np.abs(signal)) + 1e-8)


def generate_dataset(output_dir: str = "data", samples_per_class: int = 20, sr: int = 16000):
    """
    Generate synthetic dataset across 4 clinical categories.
    """
    generators = {
        "normal": generate_normal_sample,
        "dysarthria": generate_dysarthria_sample,
        "dysphonia": generate_dysphonia_sample,
        "stuttering": generate_stuttering_sample
    }
    
    total_files = 0
    for class_name, gen_func in generators.items():
        class_folder = os.path.join(output_dir, class_name)
        os.makedirs(class_folder, exist_ok=True)
        print(f"Generating {samples_per_class} audio samples for class '{class_name}'...")
        
        for i in range(1, samples_per_class + 1):
            seed = 1000 + i * 37 + (hash(class_name) % 500)
            duration = np.random.uniform(2.5, 4.0)
            audio = gen_func(duration=duration, sr=sr, seed=seed)
            
            filename = f"sample_{class_name}_{i:02d}.wav"
            file_path = os.path.join(class_folder, filename)
            sf.write(file_path, audio, sr, subtype='PCM_16')
            total_files += 1
            
    print(f"\nCompleted! Generated {total_files} synthetic audio files in '{output_dir}'.")
    return total_files


if __name__ == "__main__":
    generate_dataset()
