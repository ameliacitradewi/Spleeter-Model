# Spleeter Stem Separation (5 Stems)

Pemisahan instrumen dari lagu menggunakan model **5 stems** dari [deezer/spleeter](https://github.com/deezer/spleeter).

Model ini memisahkan audio menjadi **5 stem**:

| Stem | Deskripsi |
|------|-----------|
| `vocals` | Vokal |
| `drums` | Drum/percussion |
| `bass` | Bass |
| `piano` | Piano |
| `other` | Instrumen lainnya |

## Prasyarat

Spleeter membutuhkan **ffmpeg** dan **libsndfile**:

```bash
# macOS
brew install ffmpeg libsndfile
```

> **Catatan Python:** Spleeter membutuhkan Python 3.9–3.11 (TensorFlow belum mendukung Python 3.14). Gunakan conda jika perlu:
> ```bash
> conda create -n spleeter python=3.11 -y
> conda activate spleeter
> conda install -c conda-forge ffmpeg libsndfile -y
> ```

## Setup

Spleeter membutuhkan **Python 3.9–3.11** (TensorFlow belum mendukung Python 3.12+).

### macOS (Apple Silicon) — direkomendasikan

```bash
chmod +x setup-macos.sh
./setup-macos.sh
conda activate /path/to/Spleeter/.venv
```

Atau manual:

```bash
conda create -p .venv python=3.11 ffmpeg libsndfile -y -c conda-forge
conda activate .venv
pip install -r requirements-macos.txt --no-deps
pip install -r requirements.txt
```

### Linux / Windows

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-linux.txt --no-deps
pip install -r requirements.txt
```

> Model pretrained (`5stems`) akan diunduh otomatis saat pertama kali dijalankan (~250 MB).

## Penggunaan

Letakkan file lagu di folder `inputs/`, lalu jalankan:

```bash
python run.py inputs/lagu.mp3
```

Output disimpan di:

```
outputs/spleeter_5stems/<nama_lagu>/
├── vocals.wav
├── drums.wav
├── bass.wav
├── piano.wav
├── other.wav
└── metrics.json
```

### Opsi tambahan

```bash
# Tentukan folder output
python run.py inputs/lagu.wav -o outputs

# Proses audio lebih panjang (default max 600 detik)
python run.py inputs/lagu.mp3 --duration 900

# Evaluasi dengan ground-truth stems (untuk SDR/SIR/SAR)
python run.py inputs/lagu.wav --reference-dir references/lagu/

# Hanya pisahkan tanpa evaluasi
python run.py inputs/lagu.mp3 --skip-eval
```

## Metrik Evaluasi

File `metrics.json` berisi:

### Tanpa ground-truth (selalu dihitung)

| Metrik | Keterangan |
|--------|------------|
| `mix_reconstruction_si_sdr_db` | Seberapa baik jumlah semua stem merekonstruksi mix asli |
| `mix_reconstruction_correlation` | Korelasi antara mix asli vs jumlah stem |
| `mix_reconstruction_mae` | Mean absolute error rekonstruksi |
| `per_stem.*` | RMS, peak, crest factor, energy ratio, spectral centroid/bandwidth per stem |

### Dengan ground-truth (`--reference-dir`)

| Metrik | Keterangan |
|--------|------------|
| `sdr_db` | Signal-to-Distortion Ratio |
| `sir_db` | Signal-to-Interference Ratio |
| `sar_db` | Signal-to-Artifacts Ratio |
| `si_sdr_db` | Scale-Invariant SDR |

## Struktur Project

```
.
├── run.py              # CLI utama
├── requirements.txt
├── inputs/             # Tempat file audio input
├── outputs/            # Hasil pemisahan + metrics.json
└── src/
    ├── separate.py     # Logika pemisahan spleeter:5stems
    ├── evaluate.py     # Perhitungan metrik evaluasi
    └── utils.py        # Helper audio I/O
```

## Referensi

- [Spleeter - Deezer source separation library](https://github.com/deezer/spleeter)
- Preset model: `spleeter:5stems` (vocals / drums / bass / piano / other)
