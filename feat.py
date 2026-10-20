import numpy as np
import librosa
import soundfile as sf
from math import gcd
from scipy.signal import butter, sosfiltfilt, resample_poly

sr = 24000  # A-SPIDS native rate; Pi recordings get resampled to this
w = sr      # 1 s windows
HP = 600    # high-pass cutoff, kills piezo swing and sound-card mains hum (50 Hz harmonics up to ~550 Hz)
sos = butter(4, HP, "highpass", fs=sr, output="sos")


def load(f):
    y, r = sf.read(f, dtype="float32")
    if y.ndim > 1:
        y = y[:, 0]
    return rs(y, r)


def rs(y, r):
    if r == sr:
        return y
    g = gcd(sr, r)
    return resample_poly(y, sr // g, r // g).astype(np.float32)


def feats(y):
    y = sosfiltfilt(sos, y).astype(np.float32)  # per 1 s window, so train/predict/live see the same thing
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    b = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    z = librosa.feature.zero_crossing_rate(y)
    e = np.sqrt(np.mean(y ** 2))
    k = np.max(np.abs(y)) / (e + 1e-9)  # peakiness: insect clicks are sharp impulses
    return np.hstack([m.mean(1), m.std(1), c.mean(), b.mean(), z.mean(), e, k])


def wins(y):
    return [feats(y[i:i + w]) for i in range(0, len(y) - w + 1, w)]


def model(p):
    import joblib, sys
    d = joblib.load(p)
    if not isinstance(d, dict) or d.get("hp") != HP:
        sys.exit(f"{p} is an old model without the {HP} Hz high-pass - retrain with train.py")
    return d["rf"]
