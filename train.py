import glob, os, sys
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from feat import load, wins, HP

T = 5      # live.py block, seconds
FRAC = 0.2
K, M = 3, 6  # live.py alarm: 3 of the last 6 blocks (30 s) must be hits


def key(f):  # (recording, chunk): one log row or one record.py take = one recording
    n = os.path.basename(f)[:-4]
    r, k = n.rsplit("_", 1)
    return r, int(k)


X, Y, R = [], [], []
info = {}  # recording -> (label, session, noise, wheat)
for lab, d in enumerate(["clean", "insect"]):
    fs = sorted(glob.glob(f"data/{d}/*.wav"), key=key)
    print(d, len(fs), "files")
    for f in fs:
        r = key(f)[0]
        s, rest = r.split("_Ch", 1)
        p = rest.split("_")
        noise = p[2] if len(p) == 4 else "Silence"  # old names have no noise field
        if r.startswith("own-"):
            noise = "own-" + noise  # report our sensor separately from A-SPIDS
        info[r] = (lab, s, noise, "Wheat" in rest)
        for x in wins(load(f)):
            X.append(x)
            Y.append(lab)
            R.append(r)

X, Y, R = np.array(X), np.array(Y), np.array(R)
G = np.array([info[r][1] for r in R])  # group by session, never split one
W = np.array([info[r][3] for r in R])
print("windows:", len(Y), "| insect:", Y.sum(), "| clean:", len(Y) - Y.sum(), "| sessions:", len(set(G)))
for nz in sorted({v[2] for v in info.values()}):
    for lab in [0, 1]:
        k = (Y == lab) & np.array([info[r][2] == nz for r in R])
        if k.any():
            print(f"  {['clean', 'insect'][lab]:6} {nz:10} {k.sum() / 60:6.1f} min  {len(set(G[k]))} sessions")


def rf():
    return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)


def rep(pr, rs, th=0.5, out=None, quiet=False):
    det = det2 = npos = ok = nneg = 0
    fa, fa2, hr = {}, {}, {}
    L = ["recording,session,noise,truth,seconds,flagged,verdict,hit_blocks,beeps_3of6,blocks_5s"]
    for r in rs:
        lab, s, nz, _ = info[r]
        f = pr[R == r] > th
        v = f.mean() > FRAC
        nb = len(f) // T
        b = f[:nb * T].reshape(nb, T).mean(1) > FRAC
        a = int(b.sum())
        a2 = sum(b[max(0, j - M + 1):j + 1].sum() >= K for j in range(nb))
        if lab:
            npos += 1
            det += v
            det2 += a2 > 0
        else:
            nneg += 1
            ok += not v
            fa[nz] = fa.get(nz, 0) + a
            fa2[nz] = fa2.get(nz, 0) + a2
            hr[nz] = hr.get(nz, 0) + nb * T / 3600
        L.append(f"{r},{s},{nz},{['clean', 'insect'][lab]},{len(f)},{f.mean():.2f},"
                 f"{'INSECT' if v else 'clean'},{a},{a2},{nb}")
    if out:
        open(out, "w").write("\n".join(L) + "\n")
    h, a, a2 = sum(hr.values()), sum(fa.values()), sum(fa2.values())
    if not quiet:
        print(f"  infested recordings detected: {det}/{npos}   (with {K}-of-{M} rule: {det2}/{npos})")
        print(f"  clean recordings cleared:     {ok}/{nneg}")
        print(f"  false alarms per hour         every hit block | {K}-of-{M} rule")
        for nz in sorted(hr):
            x = hr[nz] or 1
            print(f"    {nz:12} {hr[nz] * 60:6.1f} min   {fa[nz]:4} = {fa[nz] / x:6.1f} /h | {fa2[nz]:4} = {fa2[nz] / x:6.1f} /h")
        x = h or 1
        print(f"    {'total':12} {h * 60:6.1f} min   {a:4} = {a / x:6.1f} /h | {a2:4} = {a2 / x:6.1f} /h")
        if h < 1:
            print("  (under 1 h of clean audio - false alarms/h here is rough)")
    return det2, npos, a2, h


def prob(m, x):
    return m.predict_proba(x)[:, list(m.classes_).index(1)] if 1 in m.classes_ else np.zeros(len(x))


# 1) all wheat held out -> model_eval.pkl
print("\n== test = all wheat (never seen), train = other grains ==")
if len(set(Y[W])) < 2 or len(set(Y[~W])) < 2:
    sys.exit("wheat or non-wheat side lacks a class - check data/")
m = rf().fit(X[~W], Y[~W])
pr = np.zeros(len(Y))
pr[W] = prob(m, X[W])
rep(pr, sorted(set(R[W])), out="eval_wheat.csv")
joblib.dump({"rf": m, "hp": HP}, "model_eval.pkl")
print("  saved model_eval.pkl, eval_wheat.csv")

# 2) every session held out once (5 folds) -> more clean hours for false alarms/h
print("\n== test = each session held out once (5-fold by session, all grains) ==")
pr = np.zeros(len(Y))
for a, b in GroupKFold(n_splits=min(5, len(set(G)))).split(X, Y, G):
    pr[b] = prob(rf().fit(X[a], Y[a]), X[b])
rep(pr, sorted(set(R)), out="eval_cv.csv")
print("  saved eval_cv.csv")
print(f"\n  {K}-of-{M} rule  threshold  detected  false alarms/h")
for th in [0.5, 0.6, 0.7, 0.8]:
    d, n, a, h = rep(pr, sorted(set(R)), th, quiet=True)
    print(f"               {th:.1f}        {d}/{n}     {a / h if h else 0:6.1f}")

m = rf().fit(X, Y)  # final model on all data
joblib.dump({"rf": m, "hp": HP}, "model.pkl")
print("\nsaved model.pkl")
