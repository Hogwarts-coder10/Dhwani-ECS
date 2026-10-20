import os, subprocess, sys, time
import numpy as np
import soundfile as sf
from feat import rs, wins, model

BUZ = 17   # buzzer GPIO pin (BCM)
T = 5      # seconds per block
K, M = 3, 6  # alarm only if 3 of the last 6 blocks (30 s) are hits; K, M = 1, 1 = old behaviour


def dev():  # USB card number reshuffles on reboot, so find it each run
    if os.environ.get("DEV"):
        return os.environ["DEV"]
    for l in subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout.splitlines():
        if l.startswith("card") and "USB" in l:
            return "plughw:" + l.split()[1].rstrip(":") + ",0"
    sys.exit("no USB sound card in `arecord -l` - plug it in or run DEV=plughw:N,0 python live.py")


rf = model(sys.argv[1] if len(sys.argv) > 1 else "model.pkl")
try:
    from gpiozero import Buzzer
    bz = Buzzer(BUZ)
except Exception as e:
    print("no buzzer:", e)
    bz = None

D = dev()
H = []
print("listening on", D, "... Ctrl+C to stop")
while True:
    if subprocess.run(["arecord", "-q", "-D", D, "-f", "S16_LE", "-r", "48000",
                       "-c", "1", "-d", str(T), "/tmp/p.wav"]).returncode:
        sys.exit("arecord failed - check `arecord -l`")
    y, r = sf.read("/tmp/p.wav", dtype="float32")
    pk = np.abs(y).max()
    X = wins(rs(y, r))
    if not X:
        continue
    hit = int((rf.predict_proba(X)[:, 1] > 0.5).sum())
    n = len(X)

    H = (H + [hit / n > 0.2])[-M:]
    s = time.strftime("%H:%M:%S")
    print(f"{s}  {hit}/{n} seconds flagged  peak {pk:.2f}  hits {sum(H)}/{len(H)}" + ("  CLIPPING" if pk > 0.98 else ""))
    if sum(H) >= K:
        print(f"{s}  >>> INSECTS DETECTED")
        if bz:
            bz.beep(0.2, 0.2, n=3, background=True)
