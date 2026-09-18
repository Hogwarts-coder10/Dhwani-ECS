import subprocess, sys, time
import numpy as np
import librosa
import joblib

DEV = "plughw:1,0"   # piezo USB sound card - check with `arecord -l`
MIC = None           # INMP441 for veto, e.g. "plughw:2,0" - None = off
MTH = 0.02           # mic RMS above this = outside noise, skip that second
BUZ = 17             # buzzer GPIO pin (BCM)
T = 5                # seconds per block
sr = 24000


def feats(y):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    b = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    z = librosa.feature.zero_crossing_rate(y)
    e = np.sqrt(np.mean(y ** 2))
    k = np.max(np.abs(y)) / (e + 1e-9)
    return np.hstack([m.mean(1), m.std(1), c.mean(), b.mean(), z.mean(), e, k])


def rec(dev, out):
    return subprocess.Popen(["arecord", "-q", "-D", dev, "-f", "S16_LE", "-r", "48000",
                             "-c", "1", "-d", str(T), out])


rf = joblib.load(sys.argv[1] if len(sys.argv) > 1 else "model.pkl")
try:
    from gpiozero import Buzzer
    bz = Buzzer(BUZ)
except Exception as e:
    print("no buzzer:", e)
    bz = None

print("listening... Ctrl+C to stop")
while True:
    ps = [rec(DEV, "/tmp/p.wav")]
    if MIC:
        ps.append(rec(MIC, "/tmp/m.wav"))
    for p in ps:
        p.wait()

    y, _ = librosa.load("/tmp/p.wav", sr=sr)
    if MIC:
        m, _ = librosa.load("/tmp/m.wav", sr=sr)

    hit = n = v = 0
    for i in range(0, len(y) - sr + 1, sr):
        if MIC:
            me = np.sqrt(np.mean(m[i:i + sr] ** 2))
            if me > MTH:
                v += 1
                continue
        n += 1
        if rf.predict_proba([feats(y[i:i + sr])])[0][1] > 0.5:
            hit += 1

    s = time.strftime("%H:%M:%S")
    print(f"{s}  {hit}/{n} seconds flagged" + (f"  ({v} vetoed by mic)" if MIC else ""))
    if n and hit / n > 0.2:
        print(f"{s}  >>> INSECTS DETECTED")
        if bz:
            bz.beep(0.2, 0.2, n=3, background=True)
