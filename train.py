import glob, sys
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

sr = 24000  # A-SPIDS native rate; Pi recordings get resampled to this
w = sr  # 1 s windows


def feats(y):
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    c = librosa.feature.spectral_centroid(y=y, sr=sr)
    b = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    z = librosa.feature.zero_crossing_rate(y)
    e = np.sqrt(np.mean(y ** 2))
    k = np.max(np.abs(y)) / (e + 1e-9)  # peakiness: insect clicks are sharp impulses
    return np.hstack([m.mean(1), m.std(1), c.mean(), b.mean(), z.mean(), e, k])


X, Y, G, W = [], [], [], []
for lab, d in enumerate(["clean", "insect"]):
    fs = glob.glob(f"data/{d}/*.wav")
    print(d, len(fs), "files")
    for f in fs:
        y, _ = librosa.load(f, sr=sr, mono=True)
        for i in range(0, len(y) - w + 1, w):
            X.append(feats(y[i:i + w]))
            Y.append(lab)
            G.append(f.split("/")[-1].split("_Ch")[0])  # group by session, not file
            W.append("Wheat" in f)

X, Y, G, W = np.array(X), np.array(Y), np.array(G), np.array(W)
print("windows:", len(Y), "| insect:", Y.sum(), "| clean:", len(Y) - Y.sum())

# test on wheat the model never saw; if wheat lacks a class, split by session instead
if len(set(Y[W])) == 2:
    print("test = all wheat recordings (held out), train = other grains")
    tr, te = np.where(~W)[0], np.where(W)[0]
else:
    print("test = random 25% of sessions")
    sp = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(sp.split(X, Y, G))
if len(set(Y[te])) < 2:
    sys.exit("test split has only one class - add more files per folder")

rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
rf.fit(X[tr], Y[tr])
p = rf.predict(X[te])

print(confusion_matrix(Y[te], p))
print(classification_report(Y[te], p, target_names=["clean", "insect"]))

joblib.dump(rf, "model_eval.pkl")  # never saw the test (wheat) data - use this for honest demos
print("saved model_eval.pkl")

rf.fit(X, Y)  # final model on all data
joblib.dump(rf, "model.pkl")
print("saved model.pkl")
