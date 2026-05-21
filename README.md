# CNN Music Classifier

A convolutional neural network for music genre classification using the GTZAN dataset.

## Setup

### 1. Install a Python version manager

pyenv lets each project pin its own Python version so multiple projects on the
same machine can use different versions without conflict. A `.python-version`
file is already committed to this repo so pyenv picks the right version
automatically once it is installed.

**macOS**

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

brew install pyenv
```

Add to `~/.zshrc` (or `~/.bash_profile` if you use bash):

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload your shell: `source ~/.zshrc`

---

**Ubuntu / Debian**

```bash
# Install build dependencies required to compile Python from source
sudo apt update && sudo apt install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
```

Add to `~/.bashrc`:

```bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

Reload your shell: `source ~/.bashrc`

---

**Windows**

Use [pyenv-win](https://github.com/pyenv-win/pyenv-win). In PowerShell (run as Administrator):

```powershell
winget install pyenv-win.pyenv-win
```

Then close and reopen PowerShell. pyenv-win reads `.python-version` the same
way as pyenv on macOS/Linux.

### 2. Install Python 3.12.2

```bash
pyenv install 3.12.2
```

### 3. Pin the Python version for this project

The `.python-version` file in this repo already specifies `3.12.2`, so if you
cloned the repo this step is only needed to re-pin after a fresh pyenv install:

```bash
cd CNN_Music_Classifier
pyenv local 3.12.2
```

Verify the right interpreter is active:

```bash
python --version   # should print Python 3.12.2
```

### 4. Create a virtual environment

A virtual environment is an isolated folder that holds this project's packages
separately from every other project — installing librosa here will not affect
anything else on your machine.

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt will show `(.venv)` while the environment is active. Run
`deactivate` to exit it.

### 5. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 6. Download training data

Raw audio files are stored in S3 and not tracked in git. After configuring the AWS CLI (see below), run:

```bash
python scripts/download_training_data.py
```

Files land in `training_data/` at the repo root.

**AWS CLI setup (one-time per machine):**

```bash
brew install awscli
aws configure   # enter the access key, secret key, region: us-east-1, output: json
```

Ask Colin for IAM credentials scoped to the `cnn-music-classifier-data` bucket.

## Running tests

```bash
pytest
```

## Linting

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

This checks only for fatal errors (syntax errors and undefined names). The
`.flake8` config file in the repo root ensures `.venv` and build artifacts are
excluded automatically. Clean output with exit code 0 means no issues found.

## Web app (frontend + backend)

This projects uses a React frontend that uploads a file to a Flask backend, which runs inference and returns ranked genre predictions. The frontend lives in `frontend/` and the backend lives in `src/music_classifier/web/`.

The two run as separate servers during development with Flask on port 5000 and the Vite dev server on port 5173. Vite proxies API requests to Flask, so the browser only talks to 5173 and CORS is handled automatically.

### 1. Install Node.js

The frotnend uses npm, which comes bundled wth Node.js. Install Node 18 or newer from [nodejs.org](https://nodejs.org) or via `nvm` and then verfy:

```bash
node --version # should print v18 or higher
```

### 2. Install frontend dependecies

The React app manages its dependencies separately from the Python project.

```bash
cd frontend
npm install
cd ..
```
### 3. Start the backend

The backend exposes the model over HTTP. From the repo root, with your virtual environment active, run the command:

```bash
flask --app src/music_classifier/web/app.py run
```

This serves on `http://localhost:5000`. It loads the model at startup, so the model artifact must be present.

For using the UI without a model, a stub backend that returns fake predictions is available for frontend work that doesn't rely on real inference:

```bash
python scripts/dev_backend_stub.py
```

### 4. Start the frontend

In a second terminal:

```bash
cd frontend
npm run dev
```

Open the URL it generates (default http://localhost:5173). Upload an audio file (`.wav`, `.mp3`, `.au`, `.ogg`, `.flac`, or `.m4a`) and the predicted genres will appear as ranked confidence bars.

### Running frontend tests

```bash
cd frontend
npm test
```

This is separate from the Python `pytest` suite. The frontend tests run with Vitest and live in `frontend/src/test/`.

### Production build

```bash
cd frontend
npm run build
```

Outputs static files to `frontend/dist/`, which can be served by any static host or by Flask directly

> **macOS note:** port 5000 is used by AirPlay Receiver.
> This can intercept requests to the backend and return `403 Forbidden`.
> If the backend won't bind or upoads fail with 403, disable AirPlay Receiver
> in System Settings -> General -> AirDrop & Handoff. 

## Pipeline data flow

```
Raw .au files
    │
    ▼  Stage 1 — Waveform preprocessing  (PreprocessConfig)
    │  load_audio()          resample to 22 050 Hz, mono float32
    │  segment_waveform()    chop into 3 s clips → AudioRecord
    │                        { path, label, sr, segments: (n_segs, n_samples) }
    │
    ▼  Stage 2 — Spectrogram generation  (SpectrogramConfig)
    │  segments_to_mel_spectrograms()   FFT → mel filter bank → dB
    │  normalize_spectrograms()         scale to [0, 1]
    │                        → SpectrogramRecord
    │                           { path, label, sr, spectrograms: (n_segs, 128, 130) }
    │
    ├─ stratified_split()    file-level 80/10/10 train/val/test
    │
    └─ save_dataset()        .npz archive { X: float32, y: int64, label_names }
           │
           └─ load_dataset() → (X, y, label_names) ready for model.fit()
```

## Preprocessing smoke run

Processes the GTZAN dataset end-to-end (load → resample → segment) and prints a summary:

```bash
python scripts/run_preprocess_smoke.py --dataset-root training_data/gtzan_dataset
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset-root` | `training_data/gtzan_dataset` | Root of the GTZAN-style dataset |
| `--target-sr` | `22050` | Target sample rate (Hz) |
| `--segment-seconds` | `3.0` | Duration of each segment |
| `--hop-seconds` | `None` (non-overlapping) | Stride between segments |
| `--limit-files` | `None` | Stop after N files (quick check) |
| `--pad-short` | off | Zero-pad short tail segments |

Example with overlapping windows and a file limit:

```bash
# macOS / Linux
python scripts/run_preprocess_smoke.py \
    --dataset-root training_data/gtzan_dataset \
    --segment-seconds 3.0 \
    --hop-seconds 1.5 \
    --limit-files 20
```

```powershell
# Windows (PowerShell uses ` for line continuation)
python scripts/run_preprocess_smoke.py `
    --dataset-root training_data/gtzan_dataset `
    --segment-seconds 3.0 `
    --hop-seconds 1.5 `
    --limit-files 20
```

## Project layout

```
src/
  music_classifier/
    preprocessing/
      config.py        # PreprocessConfig + SpectrogramConfig dataclasses
      io.py            # File discovery, label parsing, librosa loading
      segment.py       # Fixed-length waveform segmentation
      spectrogram.py   # Mel-spectrogram generation (FFT → mel → dB)
      normalize.py     # Per-spectrogram normalization (minmax / standardize)
      pipeline.py      # Orchestrates both stages; AudioRecord + SpectrogramRecord
      splitter.py      # Stratified file-level train/val/test split
      storage.py       # save_dataset / load_dataset (.npz format)
tests/
  test_io.py
  test_segment.py
  test_spectrogram.py
  test_splitter.py
  test_storage.py
  test_pipeline_smoke.py
  test_gtzan_integration.py  # integration tests against real GTZAN data
scripts/
  run_preprocess_smoke.py
training_data/
  gtzan_dataset/     # 10 genres × 100 .au files
```
