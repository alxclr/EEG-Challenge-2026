"""Sleep-EDF Expanded as extra data for Track 03 (pre-training only).

Sleep-EDF Expanded (PhysioNet ``sleep-edfx`` 1.0.0, open, CC BY 4.0) holds
197 whole-night polysomnograms with hypnograms scored in 30-s epochs:
153 "sleep cassette" (SC, 78 healthy people, ambulatory) and 44 "sleep
telemetry" (ST, 22 people, in hospital). Its EEG is two bipolar channels at
100 Hz: Fpz-Cz (frontal, the nearest to Muse's AF7/AF8, which Muse
references to Fpz) and Pz-Oz (posterior, alpha-rich; no Muse equivalent,
Muse's TP9/TP10 behind the ears see some of it).

``prepare`` turns each night into Muse-like windows: 5 s, resampled to
128 Hz (640 samples), in µV, unfiltered beyond the original recording. It
keeps, per night, the stretch from ``lead_min`` minutes before the first
sleep epoch (any of N1, N2, N3, REM) to ``after_min`` minutes after the
first N2 epoch (SC nights start in the afternoon, hours before bed; Muse
recordings start around bedtime). Each window gets:

- ``latency``: seconds from its start to the first N2 epoch, as the Muse
  target, NOT capped, negative after the first N2 (cap it with
  ``np.clip(latency, 0, 600)``; Muse has no window after its first N2);
- ``stage``: the stage of the 30-s epoch it starts in: 0 W, 1 N1, 2 N2,
  3 N3 (R&K 3 and 4), 4 REM, -1 movement or unscored (usable for an
  auxiliary sleep-staging task, including after the first N2).

Usage::

    python -m benchmark_utils.sleepedf prepare --out DIR   # from the track
    # then in a solver's fit() only (never at module top: the competition
    # workers have neither this file's data nor network):
    from benchmark_utils.sleepedf import load_sleepedf
    nights = load_sleepedf()   # reads $SLEEPEDF_DIR

Pre-training on it is allowed (Terms, Rule 4: public, redistributable data)
and must be declared in the final method description.
"""

import argparse
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BASE = "https://physionet.org/files/sleep-edfx/1.0.0/"
SUBSETS = {"sc": "sleep-cassette/", "st": "sleep-telemetry/"}
CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
SFREQ_OUT = 128
WINDOW_S = 5
EPOCH_S = 30
STAGES = {"Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
          "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4}


# ------------------------------------------------------------ download ---

def list_records(subset):
    """``[(night id, psg url, hypnogram url)]`` of one subset."""
    url = BASE + SUBSETS[subset]
    html = urllib.request.urlopen(url, timeout=60).read().decode()
    psg = sorted(set(re.findall(r"(S[CT]\d{4}[A-Z]\d-PSG\.edf)", html)))
    hyp = sorted(set(re.findall(r"(S[CT]\d{4}[A-Z]{2}-Hypnogram\.edf)",
                                html)))
    # PSG SC4001E0 pairs with hypnogram SC4001EC: same first 7 characters.
    hyp_of = {h[:7]: h for h in hyp}
    return [(p[:7], url + p, url + hyp_of[p[:7]]) for p in psg
            if p[:7] in hyp_of]


def download(url, path):
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


# ------------------------------------------------------------- windows ---

def night_windows(psg_path, hyp_path, lead_min=60.0, after_min=30.0):
    """Muse-like windows of one night, or ``None`` if it never reaches N2."""
    import mne
    from scipy.signal import resample_poly

    raw = mne.io.read_raw_edf(psg_path, include=list(CHANNELS),
                              preload=True, verbose="error")
    sfreq = raw.info["sfreq"]
    data = raw.get_data(picks=list(CHANNELS)) * 1e6  # V -> µV, (2, n)

    # Hypnogram: one stage per 30-s epoch from the recording start.
    annot = mne.read_annotations(hyp_path)
    n_epochs = int(data.shape[1] / sfreq // EPOCH_S)
    stage = np.full(n_epochs, -1, dtype=np.int8)
    for onset, duration, desc in zip(annot.onset, annot.duration,
                                     annot.description):
        if desc in STAGES:
            first = int(round(onset / EPOCH_S))
            last = int(round((onset + duration) / EPOCH_S))
            stage[first:min(last, n_epochs)] = STAGES[desc]
    sleep = np.flatnonzero(stage >= 1)
    n2 = np.flatnonzero(stage == 2)
    if not len(sleep) or not len(n2):
        return None
    first_sleep_s = sleep[0] * EPOCH_S
    first_n2_s = n2[0] * EPOCH_S

    start_s = max(0.0, first_sleep_s - lead_min * 60)
    stop_s = min(n_epochs * EPOCH_S, first_n2_s + after_min * 60)
    lo, hi = int(start_s * sfreq), int(stop_s * sfreq)
    # 100 Hz -> 128 Hz (resample_poly(x, 32, 25)); generic for other rates.
    up, down = SFREQ_OUT, int(round(sfreq))
    g = np.gcd(up, down)
    seg = resample_poly(data[:, lo:hi], up // g, down // g, axis=1)

    win = WINDOW_S * SFREQ_OUT
    n_win = seg.shape[1] // win
    X = seg[:, :n_win * win].reshape(2, n_win, win).transpose(1, 0, 2)
    onset = start_s + WINDOW_S * np.arange(n_win)
    return dict(
        X=X.astype(np.float16),
        latency=(first_n2_s - onset).astype(np.float32),
        stage=stage[(onset // EPOCH_S).astype(int)],
        onset=onset.astype(np.float32),        # s from the PSG's start
        first_sleep_s=np.float32(first_sleep_s),
        first_n2_s=np.float32(first_n2_s),
    )


def prepare(out, subsets=("sc", "st"), lead_min=60.0, after_min=30.0,
            limit=None, workers=8, keep_edf=False):
    """Download Sleep-EDF and write one ``.npz`` of windows per night."""
    out = Path(out)
    raw_dir = out / "edf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = [(s, *r) for s in subsets for r in list_records(s)]
    if limit:
        records = records[:limit]
    print(f"[sleepedf] {len(records)} nights -> {out}", flush=True)

    def one(rec):
        subset, night, psg_url, hyp_url = rec
        target = out / f"{night}.npz"
        if target.exists():
            return "cached"
        psg = download(psg_url, raw_dir / psg_url.rsplit("/", 1)[1])
        hyp = download(hyp_url, raw_dir / hyp_url.rsplit("/", 1)[1])
        w = night_windows(psg, hyp, lead_min, after_min)
        if w is None:
            return "no N2"
        # SC4ssN: subject ss, night N; ST7ssN likewise.
        np.savez(target, **w, subset=subset, subject=int(night[3:5]),
                 night=int(night[5]), ch_names=np.array(CHANNELS))
        if not keep_edf:
            psg.unlink()
            hyp.unlink()
        return f"{len(w['X'])} windows, N2 after " \
               f"{(w['first_n2_s'] - w['onset'][0]) / 60:.0f} min"

    with ThreadPoolExecutor(workers) as pool:
        for rec, status in zip(records, pool.map(one, records)):
            print(f"[sleepedf] {rec[0]} {rec[1]}: {status}", flush=True)


# --------------------------------------------------------------- loader ---

def load_sleepedf(path=None, channels=("Fpz-Cz", "Pz-Oz"), up_to_n2=True,
                  max_lead_min=None, subsets=("sc", "st")):
    """The prepared nights, as a list of dicts of numpy arrays.

    Parameters
    ----------
    path : path-like, default ``$SLEEPEDF_DIR``
    channels : names among ``"Fpz-Cz"``, ``"Pz-Oz"`` (order kept).
    up_to_n2 : bool
        Keep only the windows before the first N2, like Muse (the target is
        then ``np.clip(latency, 0, 600)``); False keeps the post-N2 windows
        too (sleep staging).
    max_lead_min : float, optional
        Drop the windows more than this many minutes before the first N2
        (Muse latencies rarely exceed ~1 h; the cap is 600 s anyway).

    Returns
    -------
    nights : list of dict
        ``X`` float32 ``(n, len(channels), 640)`` µV, ``latency`` float32
        ``(n,)`` s, ``stage`` int8 ``(n,)``, ``onset`` float32 ``(n,)``,
        ``subject`` (int, for subject-wise splits), ``subset``, ``night``.
    """
    path = Path(path or os.environ.get("SLEEPEDF_DIR", ""))
    files = sorted(path.glob("S*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No prepared Sleep-EDF nights in {path!s}: run "
            "`python -m benchmark_utils.sleepedf prepare --out DIR` and set "
            "SLEEPEDF_DIR=DIR.")
    nights = []
    for f in files:
        z = np.load(f)
        if str(z["subset"]) not in subsets:
            continue
        names = [str(c).replace("EEG ", "") for c in z["ch_names"]]
        idx = [names.index(c) for c in channels]
        keep = np.ones(len(z["latency"]), dtype=bool)
        if up_to_n2:
            keep &= z["latency"] > 0
        if max_lead_min is not None:
            keep &= z["latency"] <= max_lead_min * 60
        nights.append(dict(
            X=z["X"][keep][:, idx].astype(np.float32),
            latency=z["latency"][keep], stage=z["stage"][keep],
            onset=z["onset"][keep], subject=int(z["subject"]),
            # Subject ids overlap between SC and ST: keep the subset.
            subset=str(z["subset"]), night=int(z["night"]), name=f.stem,
        ))
    return nights


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare Sleep-EDF windows.")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--out", required=True)
    q.add_argument("--subsets", default="sc,st")
    q.add_argument("--lead-min", type=float, default=60.0)
    q.add_argument("--after-min", type=float, default=30.0)
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--keep-edf", action="store_true")
    args = p.parse_args(argv)
    prepare(args.out, tuple(args.subsets.split(",")), args.lead_min,
            args.after_min, args.limit, args.workers, args.keep_edf)


if __name__ == "__main__":
    sys.exit(main())
