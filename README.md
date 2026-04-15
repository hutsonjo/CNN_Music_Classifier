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
      config.py      # PreprocessConfig dataclass
      io.py          # File discovery, label parsing, librosa loading
      segment.py     # Fixed-length waveform segmentation
      pipeline.py    # Orchestrates load → resample → segment
tests/
  test_io.py
  test_segment.py
  test_pipeline_smoke.py
scripts/
  run_preprocess_smoke.py
training_data/
  gtzan_dataset/     # 10 genres × 100 .au files
```
