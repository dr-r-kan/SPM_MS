#!/usr/bin/env python3
"""Convert FIF manifest rows to EEGLAB .set plus chanloc sidecars."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))

import mne
import numpy as np
import scipy.io as sio


FILE_COLUMNS = ("file_path", "file path", "file", "path", "filename")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest_csv")
    p.add_argument("output_dir")
    p.add_argument("--output-manifest", default="")
    p.add_argument("--file-column", default="")
    p.add_argument("--input-dir", default="", help="Directory containing files when manifest paths are stale")
    p.add_argument("--montage", default="", help="MNE builtin montage name or montage file to apply before export")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    manifest = Path(args.manifest_csv).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else manifest.parent
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = Path(args.output_manifest).expanduser().resolve() if args.output_manifest else out_dir / "converted_manifest.csv"

    with manifest.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not rows or not fieldnames:
        raise SystemExit(f"No rows found in {manifest}")

    file_col = args.file_column or first_file_column(fieldnames)
    if not file_col:
        raise SystemExit(f"No file column found. Tried: {', '.join(FILE_COLUMNS)}")

    reference_positions = build_reference_positions(rows, file_col, input_dir, args.montage)
    print(f"reference montage positions: {len(reference_positions)}")

    for row in rows:
        src = resolve_input_path(row[file_col], input_dir)
        stem = src.stem
        out_set = out_dir / f"{stem}.set"
        raw = mne.io.read_raw_fif(src, preload=True, verbose="ERROR")
        raw.pick("eeg")
        if args.montage:
            raw.set_montage(load_montage(args.montage), on_missing="warn")

        raw.export(out_set, fmt="eeglab", overwrite=args.overwrite, verbose="ERROR")
        finite = write_chanloc_sidecars(raw, out_dir / stem, reference_positions)
        row[file_col] = str(out_set)
        print(f"{src.name} -> {out_set.name} ({len(raw.ch_names)} EEG channels, {finite} finite positions)")

    with out_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"manifest -> {out_manifest}")


def first_file_column(fieldnames):
    by_norm = {norm_name(x): x for x in fieldnames}
    for name in FILE_COLUMNS:
        hit = by_norm.get(norm_name(name))
        if hit:
            return hit
    return ""


def norm_name(name):
    return "".join(ch for ch in name.lower() if ch.isalnum())


def resolve_input_path(value, manifest_dir):
    p = Path(value).expanduser()
    if p.exists():
        return p.resolve()
    local = manifest_dir / Path(str(value).replace("\\", "/")).name
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(value)


def load_montage(value):
    p = Path(value).expanduser()
    if p.exists():
        return mne.channels.read_custom_montage(p)
    return mne.channels.make_standard_montage(value)


def build_reference_positions(rows, file_col, input_dir, montage_name):
    ref = {}
    for row in rows:
        src = resolve_input_path(row[file_col], input_dir)
        raw = mne.io.read_raw_fif(src, preload=False, verbose="ERROR")
        raw.pick("eeg")
        if montage_name:
            raw.set_montage(load_montage(montage_name), on_missing="warn")
        xyz = channel_positions(raw)
        for label, pos in zip(raw.ch_names, xyz):
            key = normalise_channel_label(label)
            if key and key not in ref and usable_position(pos):
                ref[key] = pos
    return ref


def channel_positions(raw, reference_positions=None):
    reference_positions = reference_positions or {}
    montage_pos = {}
    montage = raw.get_montage()
    if montage is not None:
        montage_pos = montage.get_positions().get("ch_pos", {})
    montage_pos_by_key = {normalise_channel_label(k): v for k, v in montage_pos.items()}
    xyz = np.full((len(raw.ch_names), 3), np.nan)
    for i, (name, ch) in enumerate(zip(raw.ch_names, raw.info["chs"])):
        key = normalise_channel_label(name)
        candidates = (
            montage_pos.get(name),
            montage_pos_by_key.get(key),
            ch.get("loc", np.full(12, np.nan))[:3],
            reference_positions.get(key),
        )
        for pos in candidates:
            if usable_position(pos):
                xyz[i] = np.asarray(pos, dtype=float)[:3]
                break
    return xyz


def usable_position(pos):
    if pos is None:
        return False
    pos = np.asarray(pos, dtype=float)
    return pos.size >= 3 and np.all(np.isfinite(pos[:3])) and np.linalg.norm(pos[:3]) > 0


def normalise_channel_label(label):
    s = "".join(ch for ch in str(label).strip().lower() if ch.isalnum())
    for prefix in ("eeg", "channel", "chan"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.isdigit():
        s = s.lstrip("0") or "0"
    return s


def write_chanloc_sidecars(raw, stem, reference_positions=None):
    labels = raw.ch_names
    xyz = channel_positions(raw, reference_positions)
    chanlocs = np.empty((1, len(labels)), dtype=[("labels", "O"), ("X", "O"), ("Y", "O"), ("Z", "O")])
    for i, label in enumerate(labels):
        chanlocs[0, i] = (label, float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]))

    sio.savemat(str(stem) + "_chanlocs.mat", {"chanlocs": chanlocs, "labels": np.array(labels, dtype=object), "pos": xyz})
    with open(str(stem) + "_electrodes.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["name", "x", "y", "z"])
        for label, pos in zip(labels, xyz):
            w.writerow([label, *pos])
    return int(np.sum(np.all(np.isfinite(xyz), axis=1)))


if __name__ == "__main__":
    main()
