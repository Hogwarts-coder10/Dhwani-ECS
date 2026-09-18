import csv, os, sys, zipfile
from datetime import datetime, timedelta
import soundfile as sf
from kaggle.api.kaggle_api_extended import KaggleApi

ds = "dkadyrov/stored-product-insect-database-spidb-aspids"
log = "data/aspids/aspids_log.csv"
CH = [1]                              # which channel(s) to use - set after probe
SKIP = set()                          # sessions to ignore - set after probe
CAP = {"insect": 3, "clean": 10}      # max minutes taken per log row
bugs = {"Tenebrio molitor", "Tenebrio molitor larvae",
        "Tribolium confusum", "Callosobruchus maculatus"}

a = KaggleApi()
a.authenticate()
os.makedirs("cache", exist_ok=True)


def get(f):
    p = os.path.join("cache", os.path.basename(f))
    if os.path.exists(p):
        return p
    a.dataset_download_file(ds, f, path="cache", quiet=True)
    z = p + ".zip"
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zz:
            zz.extractall("cache")
        os.remove(z)
    return p


def st(s):
    return datetime.strptime(s[:23], "%Y_%m_%d_%H_%M_%S.%f")


# full file list, cached so we only page through Kaggle once
if os.path.exists("cache/files.txt"):
    fs = open("cache/files.txt").read().split()
else:
    fs, t = [], None
    while True:
        r = a.dataset_list_files(ds, page_token=t, page_size=200)
        fs += [f.name for f in r.files]
        t = r.next_page_token
        if not t:
            break
    open("cache/files.txt", "w").write("\n".join(fs))
print(len(fs), "files listed")

# session -> {(channel, chunk): path}
ws = {}
for f in fs:
    if not f.endswith(".wav") or "/2023/05/17/2023/" in f:  # skip broken-clock folder
        continue
    s, rest = os.path.basename(f)[:-4].split("_Ch")
    c, k = rest.split("_")
    ws.setdefault(s, {})[(int(c), int(k))] = f

# chunk length: read one full chunk (a chunk 1 that has a chunk 2 after it)
for s, ch in ws.items():
    c = min(c for c, k in ch)
    if (c, 1) in ch and (c, 2) in ch:
        i = sf.info(get(ch[(c, 1)]))
        D = timedelta(seconds=i.duration)
        break
print("chunk:", D, "| sr:", i.samplerate, "| file channels:", i.channels)

if len(sys.argv) > 1 and sys.argv[1] == "probe":
    for s in ["2023_05_24_19_41_19.559_t00100",
              "2023_06_21_17_27_12.324_t00100",
              "2023_06_21_17_27_12.330_t00100"]:
        print("\n==", s, "channels:", sorted({c for c, k in ws.get(s, {})}))
        d = [f for f in fs if os.path.basename(f) == s + "_Data.txt"]
        if d:
            print(open(get(d[0]), errors="ignore").read()[:1500])
    sys.exit()

for d in ["data/insect", "data/clean"]:
    os.makedirs(d, exist_ok=True)

n = {"insect": 0, "clean": 0}
for i, r in enumerate(csv.DictReader(open(log))):
    if r["noise"] != "Silence":
        continue
    if r["target"].startswith("No Insects"):
        lab = "clean"
    elif r["target"] in bugs:
        lab = "insect"
    else:
        continue
    rs = datetime.fromisoformat(r["start"])
    re = min(datetime.fromisoformat(r["end"]), rs + timedelta(minutes=CAP[lab]))
    mat = (r["material"] or "Unknown").replace(" ", "")

    for s, ch in ws.items():
        if s in SKIP:
            continue
        s0 = st(s)
        if s0 > re:
            continue
        for (c, k), f in ch.items():
            if c not in CH:
                continue
            cs = s0 + (k - 1) * D
            a0, a1 = max(cs, rs), min(cs + D, re)
            if (a1 - a0).total_seconds() < 2:
                continue
            p = get(f)
            sr = sf.info(p).samplerate
            x0 = int((a0 - cs).total_seconds() * sr)
            x1 = int((a1 - cs).total_seconds() * sr)
            y, _ = sf.read(p, start=x0, stop=x1)
            if y.ndim > 1:
                y = y[:, 0]
            sf.write(f"data/{lab}/{s}_Ch{c}_{mat}_{i}_{k}.wav", y, sr)
            n[lab] += (a1 - a0).total_seconds() / 60
            print(f"{lab:6} {mat:12} {s} Ch{c} chunk {k}  {(a1 - a0).total_seconds():.0f}s")

print(f"\ninsect: {n['insect']:.1f} min | clean: {n['clean']:.1f} min")
