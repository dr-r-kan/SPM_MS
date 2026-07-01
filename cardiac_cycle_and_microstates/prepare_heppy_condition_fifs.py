#!/usr/bin/env python3
"""
Prepare condition-specific HEPPy FIF files from the current raw_fif layout.

The current HEPPy layout stores all usable files in heppy/raw_fif:

    suj_1_pp_raw.fif              preprocessed EEG with CFA removed
    suj_1_pp_raw_keepcfa.fif      preprocessed EEG before CFA removal
    suj_1_ica.fif                 ICA solution for the CFA-removed raw
    suj_1_keepcfa_ica.fif         ICA solution for the keep-CFA raw

This helper deliberately uses only the exact *_pp_raw.fif files as EEG input for
microstate fitting. It excludes *_keepcfa* and *_ica.fif files from the input set.
It crops/concatenates condition-specific snippets using the annotations/event
codes described in the manifest and writes a manifest for the MATLAB SPM-VB
wrapper.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "numba_cache"))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(Path(tempfile.gettempdir()) / "pycache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl_cache"))

try:
    import mne
except Exception as exc:  # pragma: no cover
    raise RuntimeError("mne is required to prepare condition-specific FIF files") from exc


DEFAULT_CONDITION_MARKERS: Dict[str, Tuple[str, ...]] = {
    "extero": ("101", "102"),
    "intero": ("103", "104", "HBT_b1", "HBT_b2"),
    "feedback": ("105", "HBT_c"),
    "intero_fb": ("106", "107", "HBT_d1", "HBT_d2"),
}


def norm_sid(x: object) -> str:
    return str(x).strip().lower()


def infer_group_and_study_from_sid(sid: str) -> Tuple[str, str]:
    m = re.search(r"(\d+)", str(sid))
    if not m:
        return "unknown", "unknown"
    n = int(m.group(1))
    if 1 <= n < 100:
        return "ANX", "ANX"
    if 100 <= n < 200:
        return "NANX", "ANX"
    if 700 <= n < 800:
        return "HTN", "HTN"
    if 800 <= n < 900:
        return "NHTN", "HTN"
    return "unknown", "unknown"


def is_exact_cfa_removed_raw(path: Path) -> bool:
    """Return True only for suj_N_pp_raw.fif, not keep-CFA or ICA files."""
    name = path.name.lower()
    return re.fullmatch(r"suj_\d+_pp_raw\.fif", name) is not None


def subject_from_raw_path(path: Path) -> str:
    m = re.match(r"^(suj_\d+)_pp_raw\.fif$", path.name.lower())
    if not m:
        raise ValueError(f"Could not infer participant from {path.name}")
    return m.group(1)


def annotation_matches(description: str, markers: Sequence[str]) -> bool:
    desc = str(description)
    desc_low = desc.lower()
    for marker in markers:
        marker = str(marker).strip()
        if not marker:
            continue
        marker_low = marker.lower()
        if marker_low.startswith("hbt_"):
            if marker_low in desc_low:
                return True
            continue
        if marker_low.isdigit():
            # Match bare numeric event codes and common annotation forms such as
            # "(103)", "Stimulus/S 103", or "HBT_b1/103" without allowing 1103.
            if re.search(rf"(?<!\d){re.escape(marker_low)}(?!\d)", desc_low):
                return True
            continue
        if marker_low in desc_low:
            return True
    return False


def intervals_from_condition_annotations(
    raw: "mne.io.BaseRaw",
    markers: Sequence[str],
    *,
    padding_s: float,
    max_marker_gap_s: float,
    min_duration_s: float,
) -> List[Tuple[float, float]]:
    onsets: List[float] = []
    durations: List[float] = []
    for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
        if annotation_matches(str(desc), markers):
            onsets.append(float(onset))
            durations.append(float(duration) if np.isfinite(duration) else 0.0)

    if not onsets:
        return []

    first_time = float(raw.times[0]) if raw.n_times else 0.0
    last_time = float(raw.times[-1]) if raw.n_times else 0.0
    padding_s = max(0.0, float(padding_s))

    # If annotations carry meaningful duration, trust them as block intervals.
    dur_intervals = []
    for onset, duration in sorted(zip(onsets, durations), key=lambda x: x[0]):
        if duration >= min_duration_s:
            start = max(first_time, onset - padding_s)
            stop = min(last_time, onset + duration + padding_s)
            if stop > start:
                dur_intervals.append((start, stop))
    if dur_intervals:
        return merge_intervals(dur_intervals, max_gap_s=padding_s)

    # Otherwise numeric/HBT annotations are sparse event markers. Group nearby
    # markers into task intervals and concatenate intervals for the same condition.
    onsets_sorted = sorted(onsets)
    intervals: List[Tuple[float, float]] = []
    block_start = onsets_sorted[0]
    previous = onsets_sorted[0]
    for onset in onsets_sorted[1:]:
        if onset - previous > max_marker_gap_s:
            start = max(first_time, block_start - padding_s)
            stop = min(last_time, previous + padding_s)
            if stop > start:
                intervals.append((start, stop))
            block_start = onset
        previous = onset
    start = max(first_time, block_start - padding_s)
    stop = min(last_time, previous + padding_s)
    if stop > start:
        intervals.append((start, stop))
    return merge_intervals(intervals, max_gap_s=padding_s)


def merge_intervals(intervals: Sequence[Tuple[float, float]], *, max_gap_s: float = 0.0) -> List[Tuple[float, float]]:
    intervals = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, stop in intervals[1:]:
        last_start, last_stop = merged[-1]
        if start <= last_stop + max(0.0, float(max_gap_s)):
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def crop_and_concatenate(raw: "mne.io.BaseRaw", intervals: Sequence[Tuple[float, float]]) -> "mne.io.BaseRaw":
    parts = []
    for start, stop in intervals:
        stop = min(float(stop), float(raw.times[-1]))
        start = max(float(start), 0.0)
        if stop <= start:
            continue
        # include_tmax=False avoids duplicating boundary samples during concatenation.
        part = raw.copy().crop(tmin=start, tmax=stop, include_tmax=False)
        parts.append(part)
    if not parts:
        raise RuntimeError("No non-empty intervals after cropping")
    if len(parts) == 1:
        return parts[0]
    return mne.concatenate_raws(parts, on_mismatch="warn")


def read_group_override(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    df = pd.read_csv(path)
    norm_cols = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    pcol = next((norm_cols[k] for k in ("participant", "subject", "subjectid", "id") if k in norm_cols), None)
    gcol = next((norm_cols[k] for k in ("group", "condition", "diagnosis") if k in norm_cols), None)
    if pcol is None or gcol is None:
        return {}
    out = {}
    for _, row in df[[pcol, gcol]].dropna().iterrows():
        out[norm_sid(row[pcol])] = str(row[gcol]).strip()
    return out


def study_from_group(group: str) -> str:
    g = str(group).strip().upper()
    if g in {"ANX", "NANX", "ANXIETY", "NONANXIOUS", "NON_ANXIOUS"}:
        return "ANX"
    if g in {"HTN", "NHTN", "HYPERTENSION", "NONHYPERTENSION", "NON_HTN"}:
        return "HTN"
    return "unknown"


def load_marker_map(path: Optional[Path]) -> Dict[str, Tuple[str, ...]]:
    if path is None:
        return dict(DEFAULT_CONDITION_MARKERS)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    out: Dict[str, Tuple[str, ...]] = {}
    for condition, markers in payload.items():
        if isinstance(markers, str):
            markers = [markers]
        out[str(condition).strip().lower()] = tuple(str(x) for x in markers)
    return out


def prepare_condition_fifs(
    *,
    raw_dir: Path,
    out_dir: Path,
    manifest_path: Path,
    conditions: Sequence[str],
    marker_map: Dict[str, Tuple[str, ...]],
    group_override: Dict[str, str],
    padding_s: float,
    max_marker_gap_s: float,
    min_duration_s: float,
    overwrite: bool,
    include_full_record: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []

    raw_files = [p for p in sorted(raw_dir.glob("suj_*_pp_raw.fif")) if is_exact_cfa_removed_raw(p)]
    if not raw_files:
        raise FileNotFoundError(f"No exact suj_N_pp_raw.fif files found in {raw_dir}")

    for raw_path in raw_files:
        sid = subject_from_raw_path(raw_path)
        group, study = infer_group_and_study_from_sid(sid)
        if sid in group_override:
            group = group_override[sid]
            study = study_from_group(group)

        keepcfa_fif = raw_dir / f"{sid}_pp_raw_keepcfa.fif"
        ica_fif = raw_dir / f"{sid}_ica.fif"
        keepcfa_ica_fif = raw_dir / f"{sid}_keepcfa_ica.fif"

        logger.info("Reading %s", raw_path.name)
        try:
            raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
        except Exception as exc:
            logger.warning("Skipping %s: could not read file (%s)", raw_path.name, exc)
            continue

        if include_full_record:
            full_out = out_dir / f"{sid}_full_pp_raw.fif"
            if overwrite or not full_out.exists():
                raw.save(full_out, overwrite=True, verbose="ERROR")
            rows.append({
                "participant": sid,
                "condition": "full",
                "group": group,
                "study": study,
                "file_path": str(full_out.resolve()),
                "raw_fif": str(raw_path.resolve()),
                "cfa_removed_fif": str(raw_path.resolve()),
                "keepcfa_fif": str(keepcfa_fif.resolve()) if keepcfa_fif.exists() else "",
                "ica_fif": str(ica_fif.resolve()) if ica_fif.exists() else "",
                "keepcfa_ica_fif": str(keepcfa_ica_fif.resolve()) if keepcfa_ica_fif.exists() else "",
                "source_intervals_s": json.dumps([[0.0, float(raw.times[-1])]]),
                "source_start_s": 0.0,
                "source_stop_s": float(raw.times[-1]),
                "n_source_intervals": 1,
                "n_samples": int(raw.n_times),
                "sfreq": float(raw.info["sfreq"]),
            })

        for condition in conditions:
            condition = str(condition).strip().lower()
            markers = marker_map.get(condition, ())
            if not markers:
                logger.warning("No markers configured for condition=%s; skipping %s", condition, sid)
                continue
            intervals = intervals_from_condition_annotations(
                raw,
                markers,
                padding_s=padding_s,
                max_marker_gap_s=max_marker_gap_s,
                min_duration_s=min_duration_s,
            )
            if not intervals:
                logger.warning("%s: no annotation intervals found for %s using markers %s", sid, condition, markers)
                continue
            try:
                cond_raw = crop_and_concatenate(raw, intervals)
            except Exception as exc:
                logger.warning("%s %s: crop/concat failed (%s)", sid, condition, exc)
                continue

            out_fif = out_dir / f"{sid}_{condition}_pp_raw.fif"
            if overwrite or not out_fif.exists():
                cond_raw.save(out_fif, overwrite=True, verbose="ERROR")
            rows.append({
                "participant": sid,
                "condition": condition,
                "group": group,
                "study": study,
                "file_path": str(out_fif.resolve()),
                "raw_fif": str(raw_path.resolve()),
                "cfa_removed_fif": str(raw_path.resolve()),
                "keepcfa_fif": str(keepcfa_fif.resolve()) if keepcfa_fif.exists() else "",
                "ica_fif": str(ica_fif.resolve()) if ica_fif.exists() else "",
                "keepcfa_ica_fif": str(keepcfa_ica_fif.resolve()) if keepcfa_ica_fif.exists() else "",
                "source_intervals_s": json.dumps([[round(a, 6), round(b, 6)] for a, b in intervals]),
                "source_start_s": float(intervals[0][0]),
                "source_stop_s": float(intervals[-1][1]),
                "n_source_intervals": int(len(intervals)),
                "n_samples": int(cond_raw.n_times),
                "sfreq": float(cond_raw.info["sfreq"]),
            })
            logger.info("%s %s: wrote %s from %d interval(s)", sid, condition, out_fif.name, len(intervals))

    if not rows:
        raise RuntimeError("No condition-specific files were created; check annotation labels and marker map")

    manifest = pd.DataFrame(rows)
    condition_order = {cond: i for i, cond in enumerate([*conditions, "full"])}
    manifest["condition_order"] = manifest["condition"].map(lambda x: condition_order.get(str(x), 999))
    manifest = manifest.sort_values(["study", "group", "participant", "condition_order"]).drop(columns=["condition_order"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("prepare_heppy_condition_fifs")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare condition-specific HEPPy FIF files from heppy/raw_fif.")
    p.add_argument("--raw-dir", default='/Users/rohan/EEG/Microstates_and_Interoception/heppy/raw_fif', type=Path, help="Directory containing suj_N_pp_raw.fif files")
    p.add_argument("--out-dir", default='/Users/rohan/EEG/Microstates_and_Interoception/heppy/condition_fifs', type=Path, help="Output directory for condition-specific FIFs")
    p.add_argument("--manifest", default='/Users/rohan/EEG/Microstates_and_Interoception/heppy/condition_fifs/manifest.csv', type=Path, help="CSV manifest to write")
    p.add_argument("--conditions", nargs="+", default=["intero", "extero", "feedback", "intero_fb"])
    p.add_argument("--condition-markers-json", type=Path, default=None)
    p.add_argument("--group-table", type=Path, default=None)
    p.add_argument("--padding-s", type=float, default=1.0)
    p.add_argument("--max-marker-gap-s", type=float, default=5.0)
    p.add_argument("--min-duration-s", type=float, default=2.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-full-record", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logger = setup_logger(args.manifest.parent / "prepare_heppy_condition_fifs.log")
    marker_map = load_marker_map(args.condition_markers_json)
    group_override = read_group_override(args.group_table)
    manifest = prepare_condition_fifs(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        conditions=tuple(str(c).strip().lower() for c in args.conditions),
        marker_map=marker_map,
        group_override=group_override,
        padding_s=float(args.padding_s),
        max_marker_gap_s=float(args.max_marker_gap_s),
        min_duration_s=float(args.min_duration_s),
        overwrite=bool(args.overwrite),
        include_full_record=bool(args.include_full_record),
        logger=logger,
    )
    logger.info("Wrote manifest with %d rows: %s", len(manifest), args.manifest)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
