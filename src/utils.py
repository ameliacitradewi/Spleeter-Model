"""Utility functions for audio I/O and path handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SPLEETER_5S_STEMS = ("vocals", "drums", "bass", "piano", "other")
MODEL_NAME = "spleeter_5stems"
SPLEETER_PRESET = "spleeter:5stems"
SAMPLE_RATE = 44100


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio as float32 array shaped [channels, samples]."""
    data, sr = sf.read(str(path), always_2d=True)
    audio = data.T.astype(np.float32)
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    if sr != sample_rate:
        from scipy import signal as scipy_signal

        num_samples = int(audio.shape[1] * sample_rate / sr)
        audio = scipy_signal.resample(audio, num_samples, axis=1).astype(np.float32)
        sr = sample_rate
    return audio, sr


def save_stem(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Save stem audio shaped [channels, samples] or [samples, channels]."""
    ensure_dir(path.parent)
    if audio.ndim == 1:
        audio_out = audio
    elif audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        audio_out = audio.T
    else:
        audio_out = audio
    sf.write(str(path), audio_out, sample_rate, subtype="FLOAT")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def stem_output_dir(base_output: Path, track_name: str) -> Path:
    return ensure_dir(base_output / MODEL_NAME / track_name)


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        return audio.mean(axis=0)
    return audio.mean(axis=1)
