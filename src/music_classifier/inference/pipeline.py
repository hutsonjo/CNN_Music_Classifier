from __future__ import annotations

from pathlib import Path

import numpy as np
from tensorflow.keras import Model

from preprocessing import PreprocessConfig, SpectrogramConfig
from preprocessing import preprocess_file, build_spectrogram_record
