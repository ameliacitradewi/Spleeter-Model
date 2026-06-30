"""Evaluation metrics for separated stems."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mir_eval
import numpy as np
from scipy import signal

from src.utils import (
    MODEL_NAME,
    SPLEETER_5S_STEMS,
    load_audio,
    save_json,
    to_mono,
)

logger = logging.getLogger(__name__)


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Scale-invariant SDR in dB."""
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    if ref.ndim > 1:
        ref = to_mono(ref)
    if est.ndim > 1:
        est = to_mono(est)

    min_len = min(len(ref), len(est))
    ref, est = ref[:min_len], est[:min_len]

    dot = np.dot(est, ref)
    ref_energy = np.dot(ref, ref) + 1e-8
    scale = dot / ref_energy
    target = scale * ref
    noise = est - target
    return float(10 * np.log10((np.dot(target, target) + 1e-8) / (np.dot(noise, noise) + 1e-8)))


def spectral_centroid(audio: np.ndarray, sr: int) -> float:
    mono = to_mono(audio)
    freqs, _, spec = signal.spectrogram(mono, fs=sr, nperseg=2048, noverlap=1024)
    mag = np.abs(spec)
    power = mag.sum(axis=1) + 1e-8
    return float(np.sum(freqs * power) / np.sum(power))


def spectral_bandwidth(audio: np.ndarray, sr: int) -> float:
    mono = to_mono(audio)
    freqs, _, spec = signal.spectrogram(mono, fs=sr, nperseg=2048, noverlap=1024)
    mag = np.abs(spec)
    power = mag.sum(axis=1) + 1e-8
    centroid = np.sum(freqs * power) / np.sum(power)
    return float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / np.sum(power)))


def stem_energy_ratio(stem: np.ndarray, mixture: np.ndarray) -> float:
    stem_rms = float(np.sqrt(np.mean(to_mono(stem) ** 2)) + 1e-8)
    mix_rms = float(np.sqrt(np.mean(to_mono(mixture) ** 2)) + 1e-8)
    return stem_rms / mix_rms


def per_stem_proxy_metrics(stem: np.ndarray, mixture: np.ndarray, sr: int) -> dict[str, float]:
    mono = to_mono(stem)
    return {
        "rms": float(np.sqrt(np.mean(mono**2))),
        "peak": float(np.max(np.abs(mono))),
        "crest_factor_db": float(20 * np.log10((np.max(np.abs(mono)) + 1e-8) / (np.sqrt(np.mean(mono**2)) + 1e-8))),
        "energy_ratio_vs_mix": stem_energy_ratio(stem, mixture),
        "spectral_centroid_hz": spectral_centroid(stem, sr),
        "spectral_bandwidth_hz": spectral_bandwidth(stem, sr),
    }


def reconstruction_metrics(mixture: np.ndarray, stems: dict[str, np.ndarray]) -> dict[str, float]:
    """Compare original mixture against sum of separated stems."""
    min_len = min(mixture.shape[-1], *(s.shape[-1] for s in stems.values()))
    mix = mixture[..., :min_len]
    recon = sum(s[..., :min_len] for s in stems.values())

    mix_mono = to_mono(mix)
    recon_mono = to_mono(recon)

    return {
        "mix_reconstruction_si_sdr_db": si_sdr(mix_mono, recon_mono),
        "mix_reconstruction_correlation": float(np.corrcoef(mix_mono, recon_mono)[0, 1]),
        "mix_reconstruction_mae": float(np.mean(np.abs(mix_mono - recon_mono))),
    }


def bss_metrics_with_references(
    estimates: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    sr: int,
) -> dict[str, dict[str, float]]:
    """Compute SDR/SIR/SAR when ground-truth stems are available."""
    common = sorted(set(estimates) & set(references))
    if not common:
        return {}

    est_list = [to_mono(estimates[name]) for name in common]
    ref_list = [to_mono(references[name]) for name in common]

    min_len = min(len(e) for e in est_list + ref_list)
    est_list = [e[:min_len] for e in est_list]
    ref_list = [r[:min_len] for r in ref_list]

    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
        np.stack(ref_list),
        np.stack(est_list),
        compute_permutation=False,
    )

    per_source: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(common):
        per_source[name] = {
            "sdr_db": float(sdr[idx]),
            "sir_db": float(sir[idx]),
            "sar_db": float(sar[idx]),
            "si_sdr_db": si_sdr(ref_list[idx], est_list[idx]),
        }

    per_source["overall"] = {
        "sdr_db": float(np.mean(sdr)),
        "sir_db": float(np.mean(sir)),
        "sar_db": float(np.mean(sar)),
        "si_sdr_db": float(np.mean([per_source[n]["si_sdr_db"] for n in common])),
    }
    return per_source


def load_reference_stems(reference_dir: Path) -> dict[str, np.ndarray]:
    refs: dict[str, np.ndarray] = {}
    for stem in SPLEETER_5S_STEMS:
        path = reference_dir / f"{stem}.wav"
        if path.exists():
            audio, _ = load_audio(path)
            refs[stem] = audio
    return refs


def evaluate_separation(
    mixture_path: Path,
    stems_dir: Path,
    *,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate separated stems and return a metrics dictionary."""
    mixture, sr = load_audio(mixture_path)

    stems: dict[str, np.ndarray] = {}
    for stem in SPLEETER_5S_STEMS:
        stem_path = stems_dir / f"{stem}.wav"
        if stem_path.exists():
            stems[stem], stem_sr = load_audio(stem_path)
            if stem_sr != sr:
                logger.warning("Sample rate mismatch for %s: %s vs %s", stem, stem_sr, sr)

    if not stems:
        raise FileNotFoundError(f"No stem .wav files found in {stems_dir}")

    metrics: dict[str, Any] = {
        "input": str(mixture_path),
        "stems_dir": str(stems_dir),
        "sample_rate": sr,
        "model": MODEL_NAME,
        "stems_found": sorted(stems.keys()),
        "reconstruction": reconstruction_metrics(mixture, stems),
        "per_stem": {
            name: per_stem_proxy_metrics(audio, mixture, sr) for name, audio in stems.items()
        },
    }

    if reference_dir is not None:
        references = load_reference_stems(reference_dir)
        if references:
            metrics["reference_dir"] = str(reference_dir)
            metrics["bss_eval"] = bss_metrics_with_references(stems, references, sr)
        else:
            logger.warning("No reference stems found in %s", reference_dir)

    return metrics


def evaluate_and_save(
    mixture_path: Path,
    stems_dir: Path,
    metrics_output: Path,
    *,
    reference_dir: Path | None = None,
) -> dict[str, Any]:
    metrics = evaluate_separation(mixture_path, stems_dir, reference_dir=reference_dir)
    save_json(metrics_output, metrics)
    logger.info("Metrics saved to %s", metrics_output)
    return metrics
