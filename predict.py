import sys
import numpy as np
import librosa
import joblib

sr = 24000  # A-SPIDS native rate; Pi recordings get resampled to this
w = sr


def feats(y):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    b = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    z = librosa.feature.zero_crossing_rate(y)
    e = np.sqrt(np.mean(y ** 2))
    k = np.max(np.abs(y)) / (e + 1e-9)
    return np.hstack([m.mean(1), m.std(1), c.mean(), b.mean(), z.mean(), e, k])


rf = joblib.load(sys.argv[2] if len(sys.argv) > 2 else "model.pkl")
y, _ = librosa.load(sys.argv[1], sr=sr, mono=True)

n = 0
for i in range(0, len(y) - w + 1, w):
    pr = rf.predict_proba([feats(y[i:i + w])])[0][1]
    if pr > 0.5:
        n += 1
        print(f"{i // sr}s  insect  ({pr:.2f})")

t = len(y) // w
print(f"{n}/{t} windows flagged")
print("INSECTS DETECTED" if n > 0.2 * t else "clean")
