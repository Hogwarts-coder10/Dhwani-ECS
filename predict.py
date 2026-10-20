import sys
from feat import load, wins, model, sr

rf = model(sys.argv[2] if len(sys.argv) > 2 else "model.pkl")
y = load(sys.argv[1])
X = wins(y)
if not X:
    sys.exit("recording shorter than 1 s")

n = 0
for i, pr in enumerate(rf.predict_proba(X)[:, 1]):
    if pr > 0.5:
        n += 1
        print(f"{i}s  insect  ({pr:.2f})")

t = len(X)
print(f"{n}/{t} windows flagged")
print("INSECTS DETECTED" if n > 0.2 * t else "clean")
