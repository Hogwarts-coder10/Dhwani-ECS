import os, subprocess, sys, time
import numpy as np
import soundfile as sf


def dev():
    if os.environ.get("DEV"):
        return os.environ["DEV"]
    for l in subprocess.run(["arecord", "-l"], capture_output=True, text=True).stdout.splitlines():
        if l.startswith("card") and "USB" in l:
            return "plughw:" + l.split()[1].rstrip(":") + ",0"
    sys.exit("no USB sound card in `arecord -l` - plug it in or run DEV=plughw:N,0 python record.py ...")


if len(sys.argv) < 4 or sys.argv[1] not in ("clean", "insect"):
    sys.exit("usage: python record.py clean|insect SESSION SECONDS [NOISE] [NOTE]\n"
             "  python record.py clean s01 300\n"
             "  python record.py clean s02 120 Talking \"talking 1 m from the jar\"\n"
             "  python record.py insect s03 300 Silence \"jar B, 5 cm deep\"")

lab, t = sys.argv[1], int(sys.argv[3])
s = "".join(c if c.isalnum() or c == "-" else "-" for c in sys.argv[2])
nz = "".join(c for c in (sys.argv[4] if len(sys.argv) > 4 else "Silence") if c.isalnum())
note = sys.argv[5] if len(sys.argv) > 5 else ""

# same pattern as get_data.py: <session>_Ch<c>_<material>_<noise>_<take>_<chunk>.wav
os.makedirs(f"data/{lab}", exist_ok=True)
f = f"data/{lab}/own-{s}_Ch0_OwnWheat_{nz}_{time.strftime('%Y%m%d%H%M%S')}_1.wav"
D = dev()
print(f"recording {t} s from {D} -> {f}  (Ctrl+C stops early)")
p = subprocess.Popen(["arecord", "-q", "-D", D, "-f", "S16_LE", "-r", "48000", "-c", "1", "-d", str(t), f])
try:
    p.wait()
except KeyboardInterrupt:
    p.wait()

if not os.path.exists(f) or os.path.getsize(f) < 1000:
    sys.exit("nothing recorded")
y, r = sf.read(f, dtype="float32")
pk = np.abs(y).max()
e = np.sqrt(np.mean(y ** 2))
cl = np.mean(np.abs(y) > 0.98)
print(f"saved {len(y) / r:.0f} s  peak {pk:.3f}  rms {e:.4f}  clipped {cl:.2%}")
if cl > 0.001:
    print("WARNING: clipping - fix the amp before training on this file")
if e < 1e-4:
    print("WARNING: nearly silent - is the probe connected?")

new = not os.path.exists("data/log.csv")
with open("data/log.csv", "a") as o:
    if new:
        o.write("file,label,session,noise,seconds,peak,rms,clipped,note\n")
    o.write(f"{f},{lab},own-{s},{nz},{len(y) / r:.0f},{pk:.3f},{e:.4f},{cl:.4f},\"{note}\"\n")
