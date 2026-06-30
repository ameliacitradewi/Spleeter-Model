"""Separate instruments from a song using Spleeter 5-stems."""

from __future__ import annotations

import logging
from pathlib import Path

from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator
from tqdm import tqdm

from src.utils import SPLEETER_5S_STEMS, SPLEETER_PRESET, save_stem, stem_output_dir

logger = logging.getLogger(__name__)

_separator: Separator | None = None


def get_separator() -> Separator:
    global _separator
    if _separator is None:
        logger.info("Loading model %s", SPLEETER_PRESET)
        _separator = Separator(SPLEETER_PRESET)
    return _separator


def load_track(path: Path, sample_rate: int, duration: float = 600.0) -> tuple:
    adapter = AudioAdapter.default()
    waveform, _ = adapter.load(
        str(path),
        offset=0,
        duration=duration,
        sample_rate=sample_rate,
    )
    return waveform, sample_rate


def separate_track(
    input_path: Path,
    output_dir: Path,
    *,
    duration: float = 600.0,
) -> dict[str, Path]:
    """
    Separate a single audio file into 5 stems with Spleeter.

    Returns mapping of stem name -> output .wav path.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    separator = get_separator()
    adapter = AudioAdapter.default()

    logger.info("Separating %s ...", input_path.name)
    waveform, _ = adapter.load(
        str(input_path),
        offset=0,
        duration=duration,
        sample_rate=separator._sample_rate,
    )
    sources = separator.separate(waveform, str(input_path))

    track_name = input_path.stem
    out_dir = stem_output_dir(output_dir, track_name)
    stem_paths: dict[str, Path] = {}

    for source_name in SPLEETER_5S_STEMS:
        if source_name not in sources:
            logger.warning("Stem %s not found in model output", source_name)
            continue
        out_path = out_dir / f"{source_name}.wav"
        save_stem(out_path, sources[source_name], separator._sample_rate)
        stem_paths[source_name] = out_path
        logger.info("Saved %s", out_path)

    expected = set(SPLEETER_5S_STEMS)
    if set(stem_paths) != expected:
        logger.warning("Missing stems: %s", expected - set(stem_paths))

    return stem_paths


def separate_batch(
    input_paths: list[Path],
    output_dir: Path,
    **kwargs,
) -> dict[str, dict[str, Path]]:
    get_separator()
    results: dict[str, dict[str, Path]] = {}
    for path in tqdm(input_paths, desc="Separating tracks"):
        results[path.stem] = separate_track(path, output_dir, **kwargs)
    return results
