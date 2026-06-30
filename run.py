#!/usr/bin/env python3
"""CLI for Spleeter 5-stem source separation and evaluation."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Suppress verbose TensorFlow logs before importing spleeter.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.evaluate import evaluate_and_save
from src.separate import separate_batch, separate_track
from src.utils import ensure_dir, stem_output_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pisahkan instrumen dari lagu menggunakan Spleeter (5 stems).",
    )
    parser.add_argument(
        "input",
        nargs="+",
        type=Path,
        help="Path ke file audio input (.wav, .mp3, .flac, dll.)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Folder output untuk stem .wav (default: outputs)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Durasi maksimum audio yang diproses dalam detik (default: 600)",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Folder berisi ground-truth stems untuk metrik SDR/SIR/SAR",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Hanya pisahkan, tanpa evaluasi metrik",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    input_paths = [p.resolve() for p in args.input]
    for path in input_paths:
        if not path.exists():
            logger.error("File tidak ditemukan: %s", path)
            return 1

    kwargs = {"duration": args.duration}

    if len(input_paths) == 1:
        separate_track(input_paths[0], output_dir, **kwargs)
    else:
        separate_batch(input_paths, output_dir, **kwargs)

    if args.skip_eval:
        logger.info("Evaluasi dilewati.")
        return 0

    for input_path in input_paths:
        stems_dir = stem_output_dir(output_dir, input_path.stem)
        metrics_path = stems_dir / "metrics.json"
        metrics = evaluate_and_save(
            input_path,
            stems_dir,
            metrics_path,
            reference_dir=args.reference_dir,
        )
        _print_metrics_summary(input_path.stem, metrics)

    return 0


def _print_metrics_summary(track_name: str, metrics: dict) -> None:
    recon = metrics["reconstruction"]
    print(f"\n=== Metrik Evaluasi: {track_name} ===")
    print(f"  Mix reconstruction SI-SDR : {recon['mix_reconstruction_si_sdr_db']:.2f} dB")
    print(f"  Mix reconstruction corr : {recon['mix_reconstruction_correlation']:.4f}")
    print(f"  Mix reconstruction MAE  : {recon['mix_reconstruction_mae']:.6f}")

    if "bss_eval" in metrics:
        overall = metrics["bss_eval"]["overall"]
        print("  --- Dengan ground-truth ---")
        print(f"  Overall SDR  : {overall['sdr_db']:.2f} dB")
        print(f"  Overall SIR  : {overall['sir_db']:.2f} dB")
        print(f"  Overall SAR  : {overall['sar_db']:.2f} dB")
        print(f"  Overall SI-SDR: {overall['si_sdr_db']:.2f} dB")

    print("  --- Per stem (proxy) ---")
    for stem, values in metrics["per_stem"].items():
        print(
            f"  {stem:8s} | RMS={values['rms']:.4f} | "
            f"energy_ratio={values['energy_ratio_vs_mix']:.3f} | "
            f"centroid={values['spectral_centroid_hz']:.0f} Hz"
        )


if __name__ == "__main__":
    sys.exit(main())
