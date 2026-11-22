# Spleeter Model

A Python CLI for separating a music mix into five WAV stems with Deezer's pretrained `spleeter:5stems` model. It processes one or more tracks and can write a JSON report for each track. The report measures how well the stems reconstruct the mix and, when reference stems are supplied, how closely each estimate matches its source. [Upstream Spleeter project](https://github.com/deezer/spleeter)

## Setup

Use Python 3.11 for the TensorFlow versions pinned in this repository. Install FFmpeg (including `ffprobe`) and libsndfile. The first separation downloads the pretrained model weights.

```sh
git clone https://github.com/ameliacitradewi/Spleeter-Model.git
cd Spleeter-Model
```

### macOS on Apple silicon

With Conda installed, create the environment and install the TensorFlow and Metal versions listed in [`requirements-macos.txt`](requirements-macos.txt).

```sh
conda create --prefix ./.venv -c conda-forge python=3.11 ffmpeg libsndfile pip -y
conda activate ./.venv
python -m pip install tensorflow-macos==2.12.0 tensorflow-metal==1.0.0 -r requirements.txt
python -m pip install --no-deps spleeter==2.4.2
```

The manual commands install TensorFlow's dependencies before installing Spleeter with `--no-deps`. This lets the project use its Apple silicon TensorFlow build while keeping Spleeter's upstream TensorFlow pin from replacing it. [Apple TensorFlow setup](https://developer.apple.com/metal/tensorflow-plugin/) · [Spleeter package requirements](https://github.com/deezer/spleeter/blob/master/pyproject.toml)

### Linux and Windows

Install FFmpeg and libsndfile for your platform and make sure the FFmpeg tools are on `PATH`. Create a Python 3.11 environment and follow the commands for your platform. [Spleeter system dependencies](https://github.com/deezer/spleeter/wiki/1.-Installation)

**Linux**

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-linux.txt -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install tensorflow==2.12.1 -r requirements.txt
.\.venv\Scripts\python.exe -m pip install --no-deps spleeter==2.4.2
```

Use `.\.venv\Scripts\python.exe` in place of `python` in the examples below if the Windows environment is inactive. The separate Spleeter install bypasses its exact TensorFlow I/O pin, which has no Windows x64 wheel for Python 3.11.

## Separate audio

Run the CLI with one input path or a list of paths. A 44.1 kHz WAV is a convenient starting point when evaluation is enabled.

```sh
python run.py inputs/track.wav
python run.py inputs/track-a.wav inputs/track-b.wav --output-dir results
python run.py inputs/track.mp3 --duration 120 --skip-eval
python run.py inputs/track.wav --reference-dir references/track
```

| Argument | Behavior |
| --- | --- |
| `input [input ...]` | One or more input audio files. Use unique base names for separate output folders. |
| `-o`, `--output-dir` | Output root. Defaults to `outputs`. |
| `--duration` | Maximum seconds to process from the start of each track. Defaults to `600`. |
| `--reference-dir` | Folder of reference WAVs for one track. For separate reference sets, evaluate each track in its own invocation. |
| `--skip-eval` | Saves the separated WAVs and skips metric computation. Useful when FFmpeg can decode the input but the evaluation reader cannot. |

A successful separation of `track.wav` writes the following files. The model outputs are saved as floating-point WAVs. `metrics.json` is written when evaluation completes.

```text
outputs/spleeter_5stems/track/
├── vocals.wav
├── drums.wav
├── bass.wav
├── piano.wav
├── other.wav
└── metrics.json
```

## Evaluate separation

| Report section | What it measures |
| --- | --- |
| `reconstruction` | SI-SDR, correlation, and mean absolute error between the original mix and the sum of the estimated stems |
| `per_stem` | RMS, peak, crest factor, RMS ratio relative to the mix, spectral centroid, and spectral bandwidth |
| `bss_eval` | SDR, SIR, SAR, and SI-SDR for matched reference stems, plus an unweighted mean across those stems |

Name reference files after the corresponding output stems, such as `vocals.wav` or `drums.wav`. Evaluation loads audio at 44.1 kHz, computes comparisons in mono, and truncates signals to a shared length. Align references to the input's starting point before evaluating. Reconstruction scores show how well the stems add up to the mix. Reference-based scores measure how well they reproduce individual sources.

## Project files

| File | Responsibility |
| --- | --- |
| [`run.py`](run.py) | CLI arguments, single-track and batch orchestration, and metric summaries |
| [`src/separate.py`](src/separate.py) | Loads and reuses the Spleeter separator, then saves the five model outputs |
| [`src/evaluate.py`](src/evaluate.py) | Reconstruction, per-stem statistics, and reference-based evaluation |
| [`src/utils.py`](src/utils.py) | Audio I/O, resampling, and output paths |
