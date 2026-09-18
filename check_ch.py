import os, zipfile
import numpy as np
import soundfile as sf
from kaggle.api.kaggle_api_extended import KaggleApi

ds = "dkadyrov/stored-product-insect-database-spidb-aspids"
s = "2023_11_01_18_09_46.378_t00100"   # log: 18:10-18:11 silence, 18:13-18:14 talking

a = KaggleApi()
a.authenticate()
fs = open("cache/files.txt").read().split()


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


def rms(y):
    return np.sqrt(np.mean(y ** 2)) + 1e-12


# session starts 18:09:46.378, so offsets in seconds:
q = (13.6, 73.6)     # 18:10:00-18:11:00 silence
t = (193.6, 253.6)   # 18:13:00-18:14:00 talking

print("ch   silence     talking     jump")
for c in range(8):
    f = [x for x in fs if os.path.basename(x) == f"{s}_Ch{c}_00001.wav"]
    if not f:
        print(c, "missing")
        continue
    y, sr = sf.read(get(f[0]))
    a0 = rms(y[int(q[0] * sr):int(q[1] * sr)])
    a1 = rms(y[int(t[0] * sr):int(t[1] * sr)])
    print(f"{c}   {a0:.2e}   {a1:.2e}   {20 * np.log10(a1 / a0):+.1f} dB")
