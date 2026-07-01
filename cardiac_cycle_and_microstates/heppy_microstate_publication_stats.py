#!/usr/bin/env python3
"""
Publication-quality downstream statistics for the HEPPy microstate pipeline.

This script consumes the MATLAB output produced by
``fit_microstates_heppy_global_group_condition.m`` and the original HEPPy FIF
files in heppy/raw_fif.  It does not refit microstates.  It performs the requested downstream
analyses:

1. effects of group and condition on microstate coverage and GFP;
2. circular cardiac-cycle modulation of each microstate;
3. heartbeat-tapping hit/miss rates by condition and their association with
   microstate variables;
4. beat-by-beat prediction of tapping success from microstate E occurrence in
   current/previous systole and diastole windows.

The inferential unit is the participant or participant-condition wherever
possible.  Sample-level circular summaries are collapsed before inferential
statistics to avoid treating autocorrelated EEG samples as independent.
Beat-level success models include pooled/study GEE with participant-clustered
working correlation and empirical-Bayes shrinkage of systole/diastole GEE estimates.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import traceback
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
from scipy.signal import find_peaks
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from statsmodels.genmod.cov_struct import Exchangeable, Independence
import mne
import patsy

SCRIPT_VERSION = "heppy_microstate_publication_stats_2026_06_30_systole_diastole"

INFERENTIAL_CONDITIONS = ("intero", "extero", "feedback", "intero_fb")
ACCURACY_CONDITIONS = ("intero", "feedback", "intero_fb")
DEFAULT_TAP_HIT_WINDOW_S = (0.350, 0.650)
SYSTOLE_WINDOW_S = (0.350, 0.600)
TAP_MARKERS_BY_CONDITION = {
    "intero": ("HBT_b1", "HBT_b2"),
    "feedback": ("HBT_c",),
    "intero_fb": ("HBT_d1", "HBT_d2"),
}
FALLBACK_TAP_MARKERS_BY_CONDITION = {
    "intero": ("103", "104"),
    "feedback": ("105",),
    "intero_fb": ("106", "107"),
}
WINDOWS_E = (
    ("E_m100_100", -0.100, 0.100, "-100 to 100 ms"),
    ("E_100_400", 0.100, 0.400, "100 to 400 ms"),
    ("E_400_600", 0.400, 0.600, "400 to 600 ms"),
)
CASE_CONTROL_CONTRASTS = (
    ("NANX", "ANX", "anxiety"),
    ("NHTN", "HTN", "hypertension"),
)
CURRENT_PHASE_LAG = "current_beat"
PREVIOUS_PHASE_LAG = "previous_beat"


@dataclass(frozen=True)
class Config:
    matlab_root: Path
    heppy_root: Path
    output_dir: Path
    conditions: Tuple[str, ...] = INFERENTIAL_CONDITIONS
    accuracy_conditions: Tuple[str, ...] = ACCURACY_CONDITIONS
    primary_state: str = "E"
    n_phase_bins: int = 24
    valid_tap_window_s: Tuple[float, float] = DEFAULT_TAP_HIT_WINDOW_S
    min_participants: int = 6
    min_beats_per_cell: int = 8
    max_beats_per_participant_condition: int = 2000
    n_circular_permutations: int = 0
    random_seed: int = 42
    quick: bool = False
    force_rebuild: bool = False

    @property
    def tables_dir(self) -> Path:
        return self.output_dir / "tables"

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def cache_dir(self) -> Path:
        return self.output_dir / "cache"


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("heppy_microstate_publication_stats")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "heppy_microstate_publication_stats.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def ensure_dirs(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.tables_dir.mkdir(exist_ok=True)
    cfg.figures_dir.mkdir(exist_ok=True)
    cfg.cache_dir.mkdir(exist_ok=True)


def norm_sid(x: object) -> str:
    return str(x).strip().lower()


def flatten_state_name(x: object) -> str:
    s = str(x).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "", s)
    if s.startswith("MS") and len(s) > 2:
        s = s[2:]
    if s.startswith("STATE"):
        s = s.replace("STATE", "")
    return s


def condition_from_name(path_or_name: object) -> str:
    name = Path(str(path_or_name)).name.lower()
    if "intero_fb" in name or "intero-fb" in name or "interofb" in name:
        return "intero_fb"
    if "feedback" in name or "hbt_c" in name:
        return "feedback"
    if "extero" in name:
        return "extero"
    if "intero" in name:
        return "intero"
    if "full" in name or re.fullmatch(r"suj_\d+_pp_raw\.fif", name):
        return "full"
    return "unknown"


def resolve_heppy_root(path: Path) -> Path:
    """Accept either the project root or the heppy directory."""
    path = Path(path)
    if (path / "raw_fif").exists():
        return path
    if (path / "heppy" / "raw_fif").exists():
        return path / "heppy"
    return path


def is_exact_cfa_removed_raw(path: Path) -> bool:
    return re.fullmatch(r"suj_\d+_pp_raw\.fif", path.name.lower()) is not None


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


def canonical_group(group: object, sid: object = "") -> str:
    g = re.sub(r"[^A-Z0-9]+", "", str(group).strip().upper())
    if g in {"", "NAN", "NONE", "UNKNOWN"} and sid:
        return infer_group_and_study_from_sid(str(sid))[0]
    if g in {"ANX", "ANXIETY", "ANXIOUS"}:
        return "ANX"
    if g in {"NANX", "NONANX", "NONANXIOUS", "NONANXIETY", "NONANXIOUSCONTROL"}:
        return "NANX"
    if g in {"HTN", "HYPERTENSION", "HYPERTENSIVE"}:
        return "HTN"
    if g in {"NHTN", "NONHTN", "NONHYPERTENSION", "NONHYPERTENSIVE"}:
        return "NHTN"
    return str(group).strip() or "unknown"


def study_from_group(group: object) -> str:
    g = canonical_group(group)
    if g in {"ANX", "NANX"}:
        return "ANX"
    if g in {"HTN", "NHTN"}:
        return "HTN"
    return "unknown"


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def grid_cols(n_items: int) -> int:
    if int(n_items) == 5:
        return 3
    return min(4, max(1, int(n_items)))


def legend_in_extra_axis(axes: np.ndarray, used: int, handles, labels) -> bool:
    flat = axes.ravel()
    if not handles or used >= len(flat):
        return False
    ax = flat[used]
    ax.axis("off")
    ax.legend(handles, labels, loc="center", frameon=False)
    return True


def add_fdr(df: pd.DataFrame, p_col: str, out_col: str = "p_fdr_bh") -> pd.DataFrame:
    out = df.copy()
    if p_col in out.columns and out[p_col].notna().any():
        mask = out[p_col].notna()
        out.loc[mask, out_col] = multipletests(out.loc[mask, p_col].astype(float), method="fdr_bh")[1]
    return out


def add_groupwise_fdr(df: pd.DataFrame, p_col: str, group_cols: Sequence[str], out_col: str) -> pd.DataFrame:
    out = df.copy()
    if p_col not in out.columns or out.empty:
        return out
    out[out_col] = np.nan
    for _, idx in out.groupby(list(group_cols), dropna=False, observed=False).groups.items():
        mask = out.index.isin(idx) & out[p_col].notna()
        if mask.any():
            out.loc[mask, out_col] = multipletests(out.loc[mask, p_col].astype(float), method="fdr_bh")[1]
    return out


def is_fif_path(path: object) -> bool:
    try:
        suffix = Path(str(path)).suffix.lower()
    except Exception:
        return False
    return suffix in {".fif", ".fiff"}


def sequence_source_index(seq: pd.DataFrame) -> Optional[np.ndarray]:
    """Return zero-based full-record sample indices for a condition sequence.

    The updated MATLAB wrapper writes this column after extracting condition
    segments from heppy/raw_fif/suj_N_pp_raw.fif.  It lets the Python models use
    full-record R-peaks/taps while indexing the condition-specific microstate
    sequence correctly.
    """
    if "source_sample_index" not in seq.columns:
        return None
    idx = pd.to_numeric(seq["source_sample_index"], errors="coerce").to_numpy(float)
    if idx.size != len(seq) or not np.isfinite(idx).any():
        return None
    idx = idx.astype(np.int64)
    # The MATLAB extractor writes intervals in chronological order.  Enforce
    # monotonicity for searchsorted-based indexing; fall back if violated.
    if idx.size > 1 and np.any(np.diff(idx) < 0):
        return None
    return idx


def map_source_samples_to_sequence_positions(samples: np.ndarray, source_idx: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Map full-record sample numbers onto sequence-row positions.

    Returns (sequence_positions, original_sample_positions_masked_index).  Only
    samples present in the extracted condition sequence are retained.
    """
    samples = np.asarray(samples, dtype=np.int64).ravel()
    if source_idx is None:
        return samples.astype(int), np.arange(samples.size, dtype=int)
    if samples.size == 0 or source_idx.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    pos = np.searchsorted(source_idx, samples)
    ok = (pos >= 0) & (pos < source_idx.size) & (source_idx[np.clip(pos, 0, source_idx.size - 1)] == samples)
    return pos[ok].astype(int), np.flatnonzero(ok).astype(int)


def cardiac_phase_for_source_index(source_idx: np.ndarray, rpeaks: np.ndarray) -> np.ndarray:
    """Compute normalised R-to-next-R phase for extracted condition samples."""
    source_idx = np.asarray(source_idx, dtype=np.int64)
    rpeaks = np.asarray(rpeaks, dtype=np.int64)
    phase = np.full(source_idx.size, np.nan, dtype=float)
    if source_idx.size == 0 or rpeaks.size < 2:
        return phase
    prev_i = np.searchsorted(rpeaks, source_idx, side="right") - 1
    next_i = prev_i + 1
    ok = (prev_i >= 0) & (next_i < rpeaks.size)
    if not np.any(ok):
        return phase
    prev_r = rpeaks[prev_i[ok]]
    next_r = rpeaks[next_i[ok]]
    denom = next_r - prev_r
    ok2 = denom > 0
    idx_ok = np.flatnonzero(ok)[ok2]
    phase[idx_ok] = 2.0 * np.pi * (source_idx[idx_ok] - prev_r[ok2]) / denom[ok2]
    return phase


def source_window_slice(source_idx: Optional[np.ndarray], centre: int, fs: float, lo: float, hi: float, n_fallback: int) -> slice | np.ndarray:
    start = int(round(centre + lo * fs))
    stop = int(round(centre + hi * fs))
    if source_idx is None:
        return slice(max(0, start), min(n_fallback, stop))
    left = np.searchsorted(source_idx, start, side="left")
    right = np.searchsorted(source_idx, stop, side="left")
    return np.arange(left, right, dtype=int)


# =============================================================================
# Manifest and sequence discovery
# =============================================================================


def discover_manifest(cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    candidates = [
        cfg.matlab_root / "manifest" / "normalised_heppy_manifest.csv",
        cfg.matlab_root / "normalised_heppy_manifest.csv",
        cfg.matlab_root / "normalised_input_manifest.csv",
    ]
    manifest = pd.DataFrame()
    for path in candidates:
        if path.exists():
            manifest = pd.read_csv(path)
            logger.info(f"Loaded manifest: {path}")
            break

    if manifest.empty:
        heppy_root = resolve_heppy_root(cfg.heppy_root)
        raw_root = heppy_root / "raw_fif"
        prepared_root = cfg.matlab_root / "condition_fifs"
        rows = []

        # Preferred fallback: condition-specific FIFs prepared by the MATLAB
        # wrapper/helper. Their sample coordinates match the global backfit
        # sequences, so they are valid sources for R-peaks, taps, and ECG.
        if prepared_root.exists():
            for fif in sorted(prepared_root.glob("suj_*_*_pp_raw.fif")):
                sid_match = re.match(r"^(suj_\d+)_(intero_fb|feedback|intero|extero|full)_pp_raw\.fif$", fif.name.lower())
                if not sid_match:
                    continue
                sid = sid_match.group(1)
                cond = sid_match.group(2)
                if cond not in cfg.conditions:
                    continue
                group, study = infer_group_and_study_from_sid(sid)
                raw_fif = raw_root / f"{sid}_pp_raw.fif"
                keepcfa_raw = raw_root / f"{sid}_pp_raw_keepcfa.fif"
                ica_fif = raw_root / f"{sid}_ica.fif"
                keepcfa_ica = raw_root / f"{sid}_keepcfa_ica.fif"
                rows.append({
                    "participant": sid,
                    "condition": cond,
                    "group": group,
                    "study": study,
                    "file_path": str(fif),
                    "raw_fif": str(raw_fif) if raw_fif.exists() else "",
                    "source_full_fif": str(raw_fif) if raw_fif.exists() else "",
                    "keepcfa_raw_fif": str(keepcfa_raw) if keepcfa_raw.exists() else "",
                    "ica_fif": str(ica_fif) if ica_fif.exists() else "",
                    "keepcfa_ica_fif": str(keepcfa_ica) if keepcfa_ica.exists() else "",
                })

        # Last-resort fallback: exact full CFA-removed raw files.  These rows
        # are placeholders for sequence-derived participant-condition metadata.
        # They allow the Python side to resolve the full source FIF even if only
        # sequence CSVs are present and the MATLAB manifest was not copied.
        if not rows and raw_root.exists():
            for fif in sorted(raw_root.glob("suj_*_pp_raw.fif")):
                if not is_exact_cfa_removed_raw(fif):
                    continue
                sid_match = re.match(r"^(suj_\d+)_pp_raw\.fif$", fif.name.lower())
                if not sid_match:
                    continue
                sid = sid_match.group(1)
                group, study = infer_group_and_study_from_sid(sid)
                keepcfa_raw = raw_root / f"{sid}_pp_raw_keepcfa.fif"
                ica_fif = raw_root / f"{sid}_ica.fif"
                keepcfa_ica = raw_root / f"{sid}_keepcfa_ica.fif"
                for cond in cfg.conditions:
                    rows.append({
                        "participant": sid,
                        "condition": cond,
                        "group": group,
                        "study": study,
                        "file_path": "",
                        "raw_fif": str(fif),
                        "source_full_fif": str(fif),
                        "keepcfa_raw_fif": str(keepcfa_raw) if keepcfa_raw.exists() else "",
                        "ica_fif": str(ica_fif) if ica_fif.exists() else "",
                        "keepcfa_ica_fif": str(keepcfa_ica) if keepcfa_ica.exists() else "",
                    })
        manifest = pd.DataFrame(rows)
        logger.info(f"Built fallback manifest from condition_fifs/raw_fif placeholders: {len(manifest)} records")

    if manifest.empty:
        raise RuntimeError("No manifest rows available. Run the MATLAB HEPPy script first or check --heppy-root/raw_fif.")

    manifest = manifest.copy()
    rename = {}
    for col in manifest.columns:
        key = re.sub(r"[^a-z0-9]", "", col.lower())
        if key in {"subject", "subjectid", "id"}:
            rename[col] = "participant"
        elif key in {"participantgroup", "diagnosis"}:
            rename[col] = "group"
        elif key in {"filepath", "file"}:
            rename[col] = "file_path"
        elif key in {"sourcefullfif", "sourcefif", "cfafreefif"}:
            rename[col] = "source_full_fif"
        elif key in {"keepcfarawfif", "rawkeepcfafif"}:
            rename[col] = "keepcfa_raw_fif"
    if rename:
        manifest = manifest.rename(columns=rename)

    if "participant" not in manifest.columns or "condition" not in manifest.columns:
        raise ValueError("Manifest must contain participant and condition columns.")
    if "file_path" not in manifest.columns:
        if "sequence_csv" in manifest.columns:
            manifest["file_path"] = ""
        else:
            raise ValueError("Manifest must contain a file_path column, or sequence-derived metadata must be present.")

    manifest["participant"] = manifest["participant"].map(norm_sid)
    manifest["condition"] = manifest["condition"].map(lambda x: str(x).strip().lower())
    if "group" not in manifest.columns:
        manifest[["group", "study"]] = manifest["participant"].apply(lambda s: pd.Series(infer_group_and_study_from_sid(s)))
    else:
        manifest["group"] = [
            canonical_group(group, sid)
            for group, sid in zip(manifest["group"], manifest["participant"])
        ]
    if "study" not in manifest.columns:
        manifest["study"] = manifest["group"].map(study_from_group)
    else:
        manifest["study"] = manifest["study"].astype(str).str.strip()
    manifest["study"] = manifest["study"].replace("unknown", np.nan)
    manifest.loc[manifest["study"].isna(), "study"] = manifest.loc[manifest["study"].isna(), "group"].map(study_from_group)
    manifest["study"] = manifest["study"].fillna("unknown")

    for col in ["raw_fif", "source_full_fif", "keepcfa_raw_fif", "ica_fif", "keepcfa_ica_fif"]:
        if col not in manifest.columns:
            manifest[col] = ""
    # In the MATLAB-generated manifest, file_path may be a staged condition .mat;
    # source_full_fif/raw_fif points back to the original CFA-free FIF.
    manifest.loc[manifest["source_full_fif"].astype(str).isin(["", "nan", "None"]), "source_full_fif"] = manifest.loc[
        manifest["source_full_fif"].astype(str).isin(["", "nan", "None"]), "raw_fif"
    ]
    manifest = manifest[manifest["condition"].isin(set(cfg.conditions))].copy()
    manifest = manifest.drop_duplicates(subset=["participant", "condition", "file_path"], keep="first")
    write_csv(manifest, cfg.tables_dir / "analysis_manifest.csv")
    return manifest


def find_sequence_files(cfg: Config) -> List[Path]:
    roots = [
        cfg.matlab_root / "backfit" / "global",
        cfg.matlab_root / "subjects",
        cfg.matlab_root,
    ]
    seqs: List[Path] = []
    patterns = ["**/*_global_sequence.csv", "**/*_microstate_sequence.csv"]
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            seqs.extend(sorted(root.glob(pat)))
    # Prefer MATLAB global backfits over older Python sequence files.
    seen = set()
    uniq = []
    for p in seqs:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def parse_sid_condition_from_sequence_path(path: Path) -> Tuple[str, str]:
    stem = path.name.lower()
    m = re.match(r"^(suj_\d+|sub_\d+|s\d+|\d+)_(intero_fb|feedback|intero|extero|full).*", stem)
    if m:
        return norm_sid(m.group(1)), condition_from_name(m.group(2))
    parts = [x.lower() for x in path.parts]
    sid = "unknown"
    condition = "unknown"
    for part in reversed(parts):
        if re.match(r"^(suj_\d+|sub_\d+|s\d+|\d+)$", part):
            sid = norm_sid(part)
            break
    for part in reversed(parts):
        c = condition_from_name(part)
        if c != "unknown":
            condition = c
            break
    return sid, condition


def load_sequence(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "microstate_name" not in df.columns:
        if "template_label" in df.columns:
            df["microstate_name"] = df["template_label"]
        elif "microstate_label" in df.columns:
            df["microstate_name"] = df["microstate_label"].map(flatten_state_name)
        elif "state_index_1based" in df.columns:
            df["microstate_name"] = df["state_index_1based"].astype(int).map(lambda k: chr(ord("A") + k - 1))
        else:
            raise ValueError(f"{path} lacks microstate_name/microstate_label/state_index_1based")
    df["microstate_name"] = df["microstate_name"].map(flatten_state_name)
    if "sample_index" not in df.columns:
        df.insert(0, "sample_index", np.arange(len(df)))
    if "gfp" not in df.columns:
        df["gfp"] = np.nan

    sid, cond = parse_sid_condition_from_sequence_path(path)
    if "participant" in df.columns and df["participant"].notna().any():
        sid = norm_sid(df["participant"].dropna().iloc[0])
    if "condition" in df.columns and df["condition"].notna().any():
        cond = condition_from_name(df["condition"].dropna().iloc[0])

    row = manifest[(manifest["participant"] == sid) & (manifest["condition"] == cond)]
    if row.empty:
        group, study = infer_group_and_study_from_sid(sid)
        file_path = ""
        raw_fif = ""
        source_full_fif = ""
        keepcfa_raw_fif = ""
    else:
        rec = row.iloc[0]
        group = canonical_group(rec.get("group", "unknown"), sid)
        study = str(rec.get("study", study_from_group(group)))
        if study.lower() in {"", "nan", "none", "unknown"}:
            study = study_from_group(group)
        file_path = str(rec.get("file_path", ""))
        raw_fif = str(rec.get("raw_fif", ""))
        source_full_fif = str(rec.get("source_full_fif", ""))
        keepcfa_raw_fif = str(rec.get("keepcfa_raw_fif", ""))

    # MATLAB sequence CSVs may carry the original full-FIF path and the original
    # source sample index for every concatenated condition sample.  Preserve both.
    if "source_full_fif" in df.columns and df["source_full_fif"].dropna().astype(str).str.len().gt(0).any():
        source_full_fif = str(df["source_full_fif"].dropna().astype(str).iloc[0])
    if not source_full_fif or source_full_fif.lower() in {"nan", "none"}:
        # If the manifest file_path is a condition-specific FIF prepared by the
        # MATLAB helper, it is the correct sample space for sequence/R-peak/tap
        # alignment.  Only fall back to the full raw when file_path is a staged
        # MAT file or missing.
        if file_path and _is_fif_path(file_path):
            source_full_fif = file_path
        elif raw_fif:
            source_full_fif = raw_fif
        else:
            source_full_fif = file_path

    df["participant"] = sid
    df["condition"] = cond
    df["group"] = group
    df["study"] = study
    df["sequence_csv"] = str(path)
    df["file_path"] = file_path
    # Use a condition-specific FIF as the primary source when available; its
    # sample coordinates match the backfit sequence.  If file_path is a staged
    # MAT file, use the linked full raw instead.
    df["source_fif"] = file_path if (file_path and _is_fif_path(file_path)) else source_full_fif
    df["source_full_fif"] = source_full_fif
    df["raw_fif"] = raw_fif
    df["keepcfa_raw_fif"] = keepcfa_raw_fif
    return df


def load_all_sequences(cfg: Config, manifest: pd.DataFrame, logger: logging.Logger) -> Dict[Tuple[str, str], pd.DataFrame]:
    seq_files = find_sequence_files(cfg)
    if not seq_files:
        raise RuntimeError(f"No sequence CSVs found under {cfg.matlab_root}.")
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for path in seq_files:
        try:
            seq = load_sequence(path, manifest)
            sid = str(seq["participant"].iloc[0])
            cond = str(seq["condition"].iloc[0])
            if cond not in cfg.conditions:
                continue
            key = (sid, cond)
            # Prefer explicitly global MATLAB backfit files when duplicates exist.
            if key in out and "global_sequence" not in path.name.lower():
                continue
            out[key] = seq
        except Exception as exc:
            logger.warning(f"Could not load sequence {path}: {exc}")
            logger.debug(traceback.format_exc())
    logger.info(f"Loaded {len(out)} participant-condition sequence files")
    if not out:
        raise RuntimeError("No usable sequence CSVs were loaded.")
    return out


# =============================================================================
# Microstate summary metrics
# =============================================================================


def run_lengths(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty(0, dtype=int)
    d = np.diff(np.r_[False, mask, False].astype(int))
    starts = np.flatnonzero(d == 1)
    stops = np.flatnonzero(d == -1)
    return stops - starts


def sequence_sfreq(seq: pd.DataFrame, default: float = 256.0) -> float:
    if "sfreq" in seq.columns and pd.to_numeric(seq["sfreq"], errors="coerce").notna().any():
        return float(pd.to_numeric(seq["sfreq"], errors="coerce").dropna().iloc[0])
    if "time_s" in seq.columns:
        t = pd.to_numeric(seq["time_s"], errors="coerce").to_numpy(float)
        dt = np.diff(t[np.isfinite(t)])
        dt = dt[dt > 0]
        if dt.size:
            return float(1.0 / np.median(dt))
    for col in ("source_fif", "file_path", "source_full_fif", "raw_fif"):
        source = str(seq.get(col, pd.Series([""])).iloc[0])
        if source and is_fif_path(source) and Path(source).exists() and mne is not None:
            try:
                raw = mne.io.read_raw_fif(source, preload=False, verbose="ERROR")
                return float(raw.info["sfreq"])
            except Exception:
                pass
    return float(default)


def compute_state_metrics(sequences: Dict[Tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    all_states = sorted({s for seq in sequences.values() for s in seq["microstate_name"].dropna().map(flatten_state_name).unique()})
    for (sid, cond), seq in sequences.items():
        labels = seq["microstate_name"].map(flatten_state_name).to_numpy(str)
        gfp = pd.to_numeric(seq.get("gfp", pd.Series(np.nan, index=seq.index)), errors="coerce").to_numpy(float)
        fs = sequence_sfreq(seq)
        n = len(seq)
        duration_s = n / fs if fs > 0 else np.nan
        for state in all_states:
            mask = labels == state
            lengths = run_lengths(mask)
            rows.append({
                "study": str(seq["study"].iloc[0]),
                "group": str(seq["group"].iloc[0]),
                "participant": sid,
                "condition": cond,
                "microstate": state,
                "n_samples": int(n),
                "sfreq": float(fs),
                "duration_s": float(duration_s),
                "coverage": float(np.mean(mask)) if n else np.nan,
                "gfp": float(np.nanmean(gfp[mask])) if np.any(mask) else np.nan,
                "occurrence_count": int(len(lengths)),
                "occurrence_rate_hz": float(len(lengths) / duration_s) if duration_s and duration_s > 0 else np.nan,
                "mean_duration_ms": float(np.mean(lengths) * 1000.0 / fs) if lengths.size and fs > 0 else np.nan,
                "median_duration_ms": float(np.median(lengths) * 1000.0 / fs) if lengths.size and fs > 0 else np.nan,
                "sequence_csv": str(seq["sequence_csv"].iloc[0]),
                "source_fif": str(seq["source_fif"].iloc[0]),
            })
    return pd.DataFrame(rows)


def zscore_series(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    sd = vals.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.zeros(len(vals)), index=x.index, dtype=float)
    return (vals - vals.mean()) / sd


def wald_terms_from_result(result, model_kind: str) -> pd.DataFrame:
    try:
        tab = result.wald_test_terms(skip_single=False).table.reset_index()
        tab.columns = ["term"] + list(tab.columns[1:])
        return tab
    except Exception:
        rows = []
        for term in result.params.index:
            p = float(result.pvalues.get(term, np.nan)) if hasattr(result, "pvalues") else np.nan
            rows.append({"term": term, "p_value": p})
        return pd.DataFrame(rows)


def normalise_p_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for candidate in ("pvalue", "P>chi2", "Pr > chi2", "Pr > F", "P>|z|", "P>|t|", "p_value"):
        if candidate in out.columns:
            out = out.rename(columns={candidate: "p_value"})
            break
    if "p_value" in out.columns:
        out = add_fdr(out, "p_value")
    return out


def fit_mixed_or_clustered_ols(formula: str, data: pd.DataFrame, group_col: str = "participant"):
    d = data.copy()
    d[group_col] = d[group_col].astype(str)
    try:
        fit = smf.mixedlm(formula, data=d, groups=d[group_col], re_formula="1").fit(
            reml=False, method="lbfgs", maxiter=1000, disp=False
        )
        return fit, "mixedlm"
    except Exception:
        fit = smf.ols(formula, data=d).fit(
            cov_type="cluster",
            cov_kwds={"groups": d[group_col].astype(str).to_numpy()},
        )
        return fit, "ols_clustered"


def coefficients_from_result(result, model_kind: str, meta: Dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({
        "term": result.params.index,
        "estimate": result.params.values,
        "std_error": result.bse.values,
        "test_stat": result.tvalues.values if hasattr(result, "tvalues") else np.nan,
        "p_value": result.pvalues.values if hasattr(result, "pvalues") else np.nan,
        "model_kind": model_kind,
        **meta,
    })


def analyse_group_condition_state_effects(metrics: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    metrics = metrics.copy()
    metrics = metrics[metrics["condition"].isin(cfg.conditions)].copy()
    write_csv(metrics, cfg.tables_dir / "microstate_state_metrics.csv")

    terms_all = []
    coefs_all = []
    means_all = []
    metric_specs = [
        ("coverage", "Microstate coverage"),
        ("gfp", "Mean GFP while active"),
        ("occurrence_rate_hz", "Occurrence rate (Hz)"),
        ("mean_duration_ms", "Mean duration (ms)"),
    ]

    for study, s_df in metrics.groupby("study", observed=False):
        if s_df["participant"].nunique() < cfg.min_participants:
            logger.warning(f"{study}: too few participants for group/condition models; descriptive tables only.")
        for metric, label in metric_specs:
            d = s_df[["participant", "group", "condition", "microstate", metric]].dropna().copy()
            if d.empty:
                continue
            d[metric] = pd.to_numeric(d[metric], errors="coerce")
            means = d.groupby(["group", "condition", "microstate"], observed=False, as_index=False).agg(
                mean=(metric, "mean"),
                sem=(metric, "sem"),
                n_participants=("participant", "nunique"),
            )
            means["metric"] = metric
            means["metric_label"] = label
            means["study"] = study
            means["ci95"] = 1.96 * means["sem"].fillna(0.0)
            means_all.append(means)

            if d["participant"].nunique() < cfg.min_participants or d["group"].nunique() < 2 or d["condition"].nunique() < 2:
                continue
            formula = f"{metric} ~ C(group) * C(condition) * C(microstate)"
            try:
                fit, kind = fit_mixed_or_clustered_ols(formula, d)
                term_df = wald_terms_from_result(fit, kind)
                term_df = normalise_p_column(term_df)
                term_df["study"] = study
                term_df["metric"] = metric
                term_df["metric_label"] = label
                term_df["model_kind"] = kind
                term_df["n_participants"] = d["participant"].nunique()
                term_df["n_rows"] = len(d)
                terms_all.append(term_df)
                coefs_all.append(coefficients_from_result(fit, kind, {"study": study, "metric": metric, "metric_label": label, "n_participants": d["participant"].nunique(), "n_rows": len(d)}))
            except Exception as exc:
                logger.warning(f"{study} {metric}: group × condition × state model failed: {exc}")

            # Per-state group × condition models are usually easier to interpret.
            for state, sd in d.groupby("microstate", observed=False):
                if sd["participant"].nunique() < cfg.min_participants or sd["group"].nunique() < 2 or sd["condition"].nunique() < 2:
                    continue
                try:
                    f2 = f"{metric} ~ C(group) * C(condition)"
                    fit, kind = fit_mixed_or_clustered_ols(f2, sd)
                    term_df = wald_terms_from_result(fit, kind)
                    term_df = normalise_p_column(term_df)
                    term_df["study"] = study
                    term_df["metric"] = metric
                    term_df["metric_label"] = label
                    term_df["microstate"] = state
                    term_df["model_kind"] = kind
                    term_df["n_participants"] = sd["participant"].nunique()
                    term_df["n_rows"] = len(sd)
                    terms_all.append(term_df)
                    coefs_all.append(coefficients_from_result(fit, kind, {"study": study, "metric": metric, "metric_label": label, "microstate": state, "n_participants": sd["participant"].nunique(), "n_rows": len(sd)}))
                except Exception as exc:
                    logger.warning(f"{study} {metric} state {state}: model failed: {exc}")

    if means_all:
        means_df = pd.concat(means_all, ignore_index=True)
        write_csv(means_df, cfg.tables_dir / "microstate_group_condition_state_means.csv")
        plot_group_condition_metric_means(means_df, cfg)
    if terms_all:
        terms_df = pd.concat(terms_all, ignore_index=True)
        if "p_value" in terms_df.columns:
            terms_df = add_fdr(terms_df, "p_value", "p_fdr_bh_across_all_terms")
        write_csv(terms_df, cfg.tables_dir / "microstate_group_condition_state_model_terms.csv")
    if coefs_all:
        coef_df = pd.concat(coefs_all, ignore_index=True)
        coef_df = add_fdr(coef_df, "p_value", "p_fdr_bh_across_all_coefficients")
        write_csv(coef_df, cfg.tables_dir / "microstate_group_condition_state_model_coefficients.csv")


def plot_group_condition_metric_means(means_df: pd.DataFrame, cfg: Config) -> None:
    for metric in sorted(means_df["metric"].unique()):
        d = means_df[means_df["metric"] == metric].copy()
        for study in sorted(d["study"].dropna().unique()):
            ds = d[d["study"] == study]
            states = sorted(ds["microstate"].dropna().unique())
            if not states:
                continue
            ncols = grid_cols(len(states))
            nrows = int(math.ceil(len(states) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.8 * nrows), squeeze=False)
            for ax, state in zip(axes.ravel(), states):
                dd = ds[ds["microstate"] == state]
                x_levels = list(cfg.conditions)
                groups = sorted(dd["group"].dropna().unique())
                for group in groups:
                    dg = dd[dd["group"] == group].set_index("condition").reindex(x_levels)
                    x = np.arange(len(x_levels))
                    y = dg["mean"].to_numpy(float)
                    yerr = dg["ci95"].fillna(0).to_numpy(float)
                    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=group)
                ax.set_title(f"State {state}")
                ax.set_xticks(np.arange(len(x_levels)))
                ax.set_xticklabels(x_levels, rotation=30, ha="right")
                ax.set_ylabel(metric)
                ax.grid(True, axis="y", alpha=0.25)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            used_extra = legend_in_extra_axis(axes, len(states), handles, labels)
            for ax in axes.ravel()[len(states) + int(used_extra):]:
                ax.axis("off")
            if handles and not used_extra:
                fig.legend(handles, labels, loc="upper right")
            fig.suptitle(f"{study}: {metric} by group, condition, and microstate")
            fig.tight_layout(rect=(0, 0, 0.98, 0.95))
            fig.savefig(cfg.figures_dir / f"{study}_{metric}_group_condition_state_means.png", dpi=220, bbox_inches="tight")
            plt.close(fig)


# =============================================================================
# Circular cardiac-cycle modulation
# =============================================================================


def circular_block_features(sequences: Dict[Tuple[str, str], pd.DataFrame], cfg: Config, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    bin_rows: List[Dict[str, object]] = []
    all_states = sorted({
        s
        for seq in sequences.values()
        for s in seq["microstate_name"].dropna().map(flatten_state_name).unique()
    })
    edges = np.linspace(0.0, 2.0 * np.pi, int(cfg.n_phase_bins) + 1)

    for (sid, cond), seq in sequences.items():
        source = pick_source_fif(seq)
        if source is None or mne is None:
            logger.warning(f"{sid} {cond}: no full source FIF available for circular cardiac phase; skipping.")
            continue
        try:
            raw = mne.io.read_raw_fif(source, preload=False, verbose="ERROR")
            rpeaks = find_rpeaks_from_raw(raw, logger)
        except Exception as exc:
            logger.warning(f"{sid} {cond}: R-peak loading failed ({exc})")
            continue

        source_samples, labels, gfp = valid_sequence_arrays(seq, raw.n_times)
        if source_samples.size < 10:
            logger.warning(f"{sid} {cond}: too few sequence samples after source-sample alignment for phase analysis; skipping.")
            continue
        rpeaks = np.asarray(rpeaks, dtype=int)
        rpeaks = rpeaks[(rpeaks >= 0) & (rpeaks < raw.n_times)]
        if rpeaks.size < 5:
            logger.warning(f"{sid} {cond}: too few R-peaks in full source FIF; skipping circular phase.")
            continue
        phase = cardiac_phase_for_source_samples(source_samples, rpeaks)
        valid_phase = np.isfinite(phase)
        if int(valid_phase.sum()) < max(10, cfg.n_phase_bins):
            logger.warning(f"{sid} {cond}: too few samples with valid R-to-R phase; skipping circular phase.")
            continue
        phase_bin = np.digitize(phase, edges, right=False) - 1
        phase_bin[phase_bin == cfg.n_phase_bins] = cfg.n_phase_bins - 1

        lo, hi = int(np.min(source_samples)), int(np.max(source_samples))
        n_rpeaks_here = int(np.sum((rpeaks >= lo) & (rpeaks <= hi)))

        for state in all_states:
            occ = ((labels == state) & valid_phase).astype(float)
            theta = phase[valid_phase]
            occ_v = occ[valid_phase]
            gfp_v = gfp[valid_phase]
            if theta.size == 0:
                continue
            cos1 = float(np.mean(occ_v * np.cos(theta)))
            sin1 = float(np.mean(occ_v * np.sin(theta)))
            cos2 = float(np.mean(occ_v * np.cos(2.0 * theta)))
            sin2 = float(np.mean(occ_v * np.sin(2.0 * theta)))
            rows.append({
                "study": str(seq["study"].iloc[0]),
                "group": str(seq["group"].iloc[0]),
                "participant": sid,
                "condition": cond,
                "microstate": state,
                "n_phase_samples": int(theta.size),
                "n_rpeaks": n_rpeaks_here,
                "coverage": float(np.mean(occ_v)),
                "cos1": cos1,
                "sin1": sin1,
                "cos2": cos2,
                "sin2": sin2,
                "resultant1": float(np.hypot(cos1, sin1)),
                "phase1_rad": float(np.mod(np.arctan2(sin1, cos1), 2.0 * np.pi)),
                "gfp_cos1": float(np.nanmean(np.where(occ_v > 0, gfp_v, np.nan) * np.cos(theta))) if np.any(occ_v > 0) else np.nan,
                "gfp_sin1": float(np.nanmean(np.where(occ_v > 0, gfp_v, np.nan) * np.sin(theta))) if np.any(occ_v > 0) else np.nan,
                "source_fif": str(source),
                "uses_source_sample_index": bool("source_sample_index" in seq.columns),
            })

            for b in range(int(cfg.n_phase_bins)):
                bm = valid_phase & (phase_bin == b)
                if not np.any(bm):
                    continue
                hit = bm & (labels == state)
                bin_rows.append({
                    "study": str(seq["study"].iloc[0]),
                    "group": str(seq["group"].iloc[0]),
                    "participant": sid,
                    "condition": cond,
                    "microstate": state,
                    "phase_bin": int(b),
                    "phase_mid_rad": float((edges[b] + edges[b + 1]) / 2.0),
                    "phase_mid_fraction": float((b + 0.5) / cfg.n_phase_bins),
                    "coverage": float(np.mean(labels[bm] == state)),
                    "gfp": float(np.nanmean(gfp[hit])) if np.any(hit) else np.nan,
                    "n_samples": int(np.sum(bm)),
                })
    return pd.DataFrame(rows), pd.DataFrame(bin_rows)


def hotelling_1sample(X: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return {"n": 0, "p_dim": 0, "T2": np.nan, "F": np.nan, "p_value": np.nan}
    X = X[np.all(np.isfinite(X), axis=1)]
    if X.size == 0:
        return {"n": 0, "p_dim": 0, "T2": np.nan, "F": np.nan, "p_value": np.nan}
    n, p_dim = X.shape
    mu = X.mean(axis=0)
    if n <= p_dim + 1:
        angle = float(np.mod(np.arctan2(mu[1], mu[0]), 2.0 * np.pi)) if p_dim >= 2 else np.nan
        return {
            "n": int(n), "p_dim": int(p_dim),
            "mean_x": float(mu[0]) if p_dim else np.nan,
            "mean_y": float(mu[1]) if p_dim > 1 else np.nan,
            "T2": np.nan, "F": np.nan, "p_value": np.nan,
            "resultant_length": float(np.hypot(mu[0], mu[1])) if p_dim >= 2 else np.nan,
            "mean_angle_rad": angle,
            "mean_phase_fraction": float(angle / (2.0 * np.pi)) if np.isfinite(angle) else np.nan,
        }
    S = np.cov(X, rowvar=False, ddof=1)
    invS = np.linalg.pinv(S)
    T2 = float(n * mu.T @ invS @ mu)
    F = float((n - p_dim) / (p_dim * (n - 1)) * T2)
    p_value = float(1.0 - stats.f.cdf(F, p_dim, n - p_dim))
    angle = float(np.mod(np.arctan2(mu[1], mu[0]), 2.0 * np.pi)) if p_dim >= 2 else np.nan
    return {
        "n": int(n), "p_dim": int(p_dim),
        "mean_x": float(mu[0]) if p_dim else np.nan,
        "mean_y": float(mu[1]) if p_dim > 1 else np.nan,
        "T2": T2, "F": F, "p_value": p_value,
        "resultant_length": float(np.hypot(mu[0], mu[1])) if p_dim >= 2 else np.nan,
        "mean_angle_rad": angle,
        "mean_phase_fraction": float(angle / (2.0 * np.pi)) if np.isfinite(angle) else np.nan,
    }


def analyse_circular_modulation(circ_df: pd.DataFrame, bin_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    write_csv(circ_df, cfg.tables_dir / "cardiac_phase_participant_condition_vectors.csv")
    write_csv(bin_df, cfg.tables_dir / "cardiac_phase_bin_coverage.csv")
    if circ_df.empty:
        logger.warning("No circular cardiac-cycle rows available; skipping circular statistics.")
        return

    hot_rows = []
    group_cols = ["study", "group", "condition", "microstate"]
    for keys, d in circ_df.groupby(group_cols, observed=False):
        rec_base = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        hot_rows.append({**rec_base, "harmonic": "first", **hotelling_1sample(d[["cos1", "sin1"]].to_numpy(float))})
        hot_rows.append({**rec_base, "harmonic": "second", **hotelling_1sample(d[["cos2", "sin2"]].to_numpy(float))})
    hot = pd.DataFrame(hot_rows)
    if not hot.empty:
        hot = add_fdr(hot, "p_value")
    write_csv(hot, cfg.tables_dir / "cardiac_phase_hotelling_tests.csv")

    term_rows = []
    for state, sd in circ_df.groupby("microstate", observed=False):
        for feature in ["coverage", "cos1", "sin1", "cos2", "sin2", "resultant1"]:
            d = sd[["participant", "group", "condition", feature]].dropna().copy()
            if d.empty or d["participant"].nunique() < cfg.min_participants:
                continue
            try:
                result, kind = fit_mixed_or_clustered_ols(d, f"{feature} ~ C(group) * C(condition)", "participant", logger)
                terms = normalise_p_column(wald_terms_from_result(result, kind))
                terms["microstate"] = state
                terms["feature"] = feature
                terms["model_kind"] = kind
                terms["n_participants"] = int(d["participant"].nunique())
                terms["n_rows"] = int(len(d))
                term_rows.append(terms)
            except Exception as exc:
                logger.warning(f"Circular group/condition model failed for state={state}, feature={feature}: {exc}")
    models = pd.concat(term_rows, ignore_index=True) if term_rows else pd.DataFrame()
    if not models.empty and "p_value" in models.columns:
        models = add_fdr(models, "p_value")
    write_csv(models, cfg.tables_dir / "cardiac_phase_group_condition_models.csv")

    plot_cardiac_phase_bins(bin_df, cfg)
    plot_polar_hotelling(hot, cfg)


def plot_cardiac_phase_bins(bin_df: pd.DataFrame, cfg: Config) -> None:
    if bin_df.empty:
        return
    d = (
        bin_df.groupby(["study", "microstate", "phase_bin", "phase_mid_fraction"], observed=False, as_index=False)
        .agg(mean_coverage=("coverage", "mean"), sem=("coverage", "sem"), n=("participant", "nunique"))
    )
    for study, sd in d.groupby("study", observed=False):
        states = sorted(sd["microstate"].dropna().unique())
        if not states:
            continue
        ncols = grid_cols(len(states))
        nrows = int(math.ceil(len(states) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False)
        for ax, state in zip(axes.ravel(), states):
            ss = sd[sd["microstate"] == state].sort_values("phase_bin")
            x = ss["phase_mid_fraction"].to_numpy(float)
            y = ss["mean_coverage"].to_numpy(float)
            sem = ss["sem"].fillna(0).to_numpy(float)
            ax.plot(x, y, marker="o")
            ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.15)
            ax.set_title(f"State {state}")
            ax.set_xlabel("RR phase")
            ax.set_ylabel("Coverage")
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.25)
        for ax in axes.ravel()[len(states):]:
            ax.axis("off")
        fig.suptitle(f"{study}: microstate coverage across normalised cardiac phase")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(cfg.figures_dir / f"{study}_circular_phase_bin_coverage_by_state.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    e = bin_df[bin_df["microstate"].astype(str).str.upper() == cfg.primary_state.upper()].copy()
    if e.empty:
        return
    e_sum = (
        e.groupby(["condition", "phase_bin", "phase_mid_fraction"], observed=False, as_index=False)
        .agg(mean_coverage=("coverage", "mean"), sem=("coverage", "sem"), n=("participant", "nunique"))
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for cond, cd in e_sum.groupby("condition", observed=False):
        cd = cd.sort_values("phase_bin")
        x = cd["phase_mid_fraction"].to_numpy(float)
        y = cd["mean_coverage"].to_numpy(float)
        sem = cd["sem"].fillna(0).to_numpy(float)
        ax.plot(x, y, marker="o", label=str(cond))
        ax.fill_between(x, y - 1.96 * sem, y + 1.96 * sem, alpha=0.12)
    ax.set_xlabel("Normalised RR phase")
    ax.set_ylabel(f"Microstate {cfg.primary_state} coverage")
    ax.set_title(f"Microstate {cfg.primary_state} coverage across the cardiac cycle")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / f"cardiac_phase_microstate_{cfg.primary_state}_coverage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_polar_hotelling(hot: pd.DataFrame, cfg: Config) -> None:
    if hot.empty:
        return
    d = hot[hot["harmonic"] == "first"].dropna(subset=["mean_x", "mean_y"]).copy()
    if d.empty:
        return
    for study, sd in d.groupby("study", observed=False):
        states = sorted(sd["microstate"].dropna().unique())
        if not states:
            continue
        ncols = grid_cols(len(states))
        nrows = int(math.ceil(len(states) / ncols))
        fig, axes = plt.subplots(nrows, ncols, subplot_kw={"projection": "polar"}, figsize=(4.0 * ncols, 3.8 * nrows), squeeze=False)
        for ax, state in zip(axes.ravel(), states):
            st = sd[sd["microstate"] == state]
            for _, row in st.iterrows():
                angle = math.atan2(float(row["mean_y"]), float(row["mean_x"]))
                radius = float(row.get("resultant_length", np.nan))
                if np.isfinite(radius):
                    ax.plot([angle, angle], [0, radius], marker="o", label=f"{row['group']} {row['condition']}")
            ax.set_title(f"State {state}")
            rmax = np.nanmax(st.get("resultant_length", pd.Series([0])).to_numpy(float))
            ax.set_ylim(0, max(1e-6, rmax * 1.2))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        used_extra = legend_in_extra_axis(axes, len(states), handles, labels)
        for ax in axes.ravel()[len(states) + int(used_extra):]:
            ax.axis("off")
        fig.suptitle(f"{study}: first-harmonic mean vectors")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(cfg.figures_dir / f"{study}_circular_first_harmonic_polar_vectors.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# R-peaks and taps
# =============================================================================


def collapse_binary_peak_train(signal: np.ndarray, merge_gap: int = 2) -> np.ndarray:
    peak_times = np.flatnonzero(np.asarray(signal) > 0.5)
    if peak_times.size == 0:
        return np.empty(0, dtype=int)
    peaks = [int(peak_times[0])]
    for idx in peak_times[1:]:
        idx = int(idx)
        if idx - peaks[-1] > merge_gap:
            peaks.append(idx)
    return np.asarray(peaks, dtype=int)


def plausible_rpeaks(peaks: np.ndarray, fs: float) -> bool:
    peaks = np.asarray(peaks, dtype=int)
    if peaks.size < 5:
        return False
    rr = np.diff(peaks) / float(fs)
    rr = rr[np.isfinite(rr)]
    if rr.size < 4:
        return False
    med = float(np.median(rr))
    return 0.25 <= med <= 2.50


def find_ecg_channel(raw) -> Optional[str]:
    preferred = ["ECG", "EKG", "ECG1", "EKG1", "EXG4", "EXG3"]
    upper = {ch.upper().replace(" ", ""): ch for ch in raw.ch_names}
    for p in preferred:
        key = p.upper().replace(" ", "")
        if key in upper:
            return upper[key]
    for ch in raw.ch_names:
        u = ch.upper()
        if "ECG" in u or "EKG" in u or u.startswith("EXG"):
            return ch
    return None


def find_rpeaks_from_raw(raw, logger: Optional[logging.Logger] = None) -> np.ndarray:
    fs = float(raw.info["sfreq"])
    candidates = []
    for ch in raw.ch_names:
        u = ch.upper().replace(" ", "")
        if "RPEAK" in u or "R_PEAK" in u or u.startswith("STI") or "STI014" in u:
            candidates.append(ch)
    candidates = sorted(set(candidates), key=lambda x: 0 if x.upper().replace(" ", "") == "STI014" else 1)

    for ch in candidates:
        try:
            sig = raw.get_data(picks=[ch])[0]
            peaks = collapse_binary_peak_train(sig)
            if plausible_rpeaks(peaks, fs):
                return peaks
        except Exception as exc:
            if logger:
                logger.debug(f"Sparse R-peak extraction failed for {ch}: {exc}")

    for ch in candidates:
        try:
            events = mne.find_events(
                raw,
                stim_channel=ch,
                output="onset",
                consecutive="increasing",
                min_duration=0,
                shortest_event=1,
                initial_event=False,
                verbose="ERROR",
            )
            peaks = np.asarray(events[:, 0], dtype=int) - int(raw.first_samp)
            if plausible_rpeaks(peaks, fs):
                return peaks
        except Exception as exc:
            if logger:
                logger.debug(f"mne.find_events failed for {ch}: {exc}")

    ecg_ch = find_ecg_channel(raw)
    if ecg_ch is not None:
        try:
            import neurokit2 as nk
            ecg = raw.get_data(picks=[ecg_ch])[0]
            ecg_clean = nk.ecg_clean(-ecg, sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(ecg_clean, sampling_rate=fs, method="promac", correct_artifacts=True)
            peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
            if plausible_rpeaks(peaks, fs):
                return peaks
        except Exception as exc:
            if logger:
                logger.debug(f"ECG fallback R-peak detection failed: {exc}")
    return np.empty(0, dtype=int)


def resolve_tap_markers(raw, condition: str) -> List[str]:
    descriptions = {str(x) for x in raw.annotations.description}
    preferred = [m for m in TAP_MARKERS_BY_CONDITION.get(condition, ()) if any(m in d for d in descriptions)]
    if preferred:
        return preferred
    fallback = [m for m in FALLBACK_TAP_MARKERS_BY_CONDITION.get(condition, ()) if any(m in d for d in descriptions)]
    if fallback:
        return fallback
    return list(TAP_MARKERS_BY_CONDITION.get(condition, ()) or FALLBACK_TAP_MARKERS_BY_CONDITION.get(condition, ()))


def extract_tap_samples(raw, condition: str, logger: Optional[logging.Logger] = None) -> np.ndarray:
    markers = resolve_tap_markers(raw, condition)
    if not markers:
        return np.empty(0, dtype=int)

    def mapper(description: str):
        desc = str(description)
        return 1 if any(m in desc for m in markers) else None

    try:
        events, _ = mne.events_from_annotations(raw, event_id=mapper, verbose="ERROR")
        if len(events) == 0:
            return np.empty(0, dtype=int)
        samples = events[:, 0].astype(int) - int(raw.first_samp)
        return np.sort(samples[(samples >= 0) & (samples < raw.n_times)].astype(int))
    except Exception as exc:
        if logger:
            logger.debug(f"Tap extraction failed for condition={condition}: {exc}")
        return np.empty(0, dtype=int)


def nearest_r_for_taps(rpeaks: np.ndarray, taps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rpeaks = np.asarray(rpeaks, dtype=int)
    taps = np.asarray(taps, dtype=int)
    if rpeaks.size == 0 or taps.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    idx_right = np.searchsorted(rpeaks, taps, side="left")
    idx_left = idx_right - 1
    dist_left = np.full(taps.size, np.inf)
    dist_right = np.full(taps.size, np.inf)
    left_ok = idx_left >= 0
    right_ok = idx_right < rpeaks.size
    dist_left[left_ok] = np.abs(taps[left_ok] - rpeaks[idx_left[left_ok]])
    dist_right[right_ok] = np.abs(taps[right_ok] - rpeaks[idx_right[right_ok]])
    assigned = np.where(dist_right < dist_left, idx_right, idx_left)
    valid = (assigned >= 0) & (assigned < rpeaks.size)
    return assigned[valid].astype(int), np.where(valid)[0].astype(int)


def assign_valid_taps_to_beats(
    rpeaks: np.ndarray,
    taps: np.ndarray,
    fs: float,
    valid_window_s: Tuple[float, float],
) -> Dict[int, int]:
    assigned_r_idx, tap_positions = nearest_r_for_taps(rpeaks, taps)
    if assigned_r_idx.size == 0:
        return {}
    taps_valid = taps[tap_positions]
    delays_s = (taps_valid - rpeaks[assigned_r_idx]) / float(fs)
    in_window = (delays_s >= valid_window_s[0]) & (delays_s <= valid_window_s[1])
    assigned_r_idx = assigned_r_idx[in_window]
    taps_valid = taps_valid[in_window]
    delays_s = delays_s[in_window]
    best: Dict[int, Tuple[int, float]] = {}
    for r_i, tap, delay in zip(assigned_r_idx, taps_valid, delays_s):
        r_i = int(r_i)
        score = abs(float(delay))
        if r_i not in best or score < best[r_i][1]:
            best[r_i] = (int(tap), score)
    return {k: v[0] for k, v in best.items()}


def _is_fif_path(value: object) -> bool:
    try:
        p = Path(str(value))
    except Exception:
        return False
    return p.suffix.lower() in {".fif", ".fiff"}


def pick_source_fif(seq: pd.DataFrame) -> Optional[Path]:
    # Prefer the prepared condition-specific FIF because its sample coordinates
    # match the global backfit sequence. The full raw and keep-CFA files are
    # provenance/fallback sources only.
    for col in ["source_fif", "file_path", "source_full_fif", "raw_fif", "keepcfa_raw_fif"]:
        if col in seq.columns:
            vals = [str(x) for x in seq[col].dropna().unique() if str(x) and str(x).lower() not in {"nan", "none", "null"}]
            for v in vals:
                p = Path(v)
                if p.exists() and _is_fif_path(p):
                    return p
    return None


def sequence_source_samples(seq: pd.DataFrame, raw_n_times: Optional[int] = None) -> np.ndarray:
    """Return full-raw, zero-based source samples for a sequence row set."""
    candidates = ["source_sample_index", "source_sample", "source_sample_zero_based", "source_samples"]
    for col in candidates:
        if col in seq.columns:
            src = pd.to_numeric(seq[col], errors="coerce").to_numpy(float)
            if np.isfinite(src).sum() >= max(1, int(0.9 * len(seq))):
                src = src.astype(float)
                # MATLAB writes zero-based source_sample_index.  Only convert
                # older one-based exports if the maximum sample is outside the
                # zero-based Raw range.  Do not subtract merely because a condition
                # segment starts after sample zero.
                finite = src[np.isfinite(src)]
                if finite.size and raw_n_times is not None and np.nanmax(finite) >= raw_n_times:
                    src = src - 1.0
                out = np.asarray(np.rint(src), dtype=int)
                return out
    return np.arange(len(seq), dtype=int)


def valid_sequence_arrays(seq: pd.DataFrame, raw_n_times: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = sequence_source_samples(seq, raw_n_times=raw_n_times)
    labels = seq["microstate_name"].map(flatten_state_name).to_numpy(str)
    gfp = pd.to_numeric(seq.get("gfp", pd.Series(np.nan, index=seq.index)), errors="coerce").to_numpy(float)
    n = min(len(src), len(labels), len(gfp))
    src = src[:n]
    labels = labels[:n]
    gfp = gfp[:n]
    ok = (src >= 0) & (src < int(raw_n_times))
    return src[ok], labels[ok], gfp[ok]


def cardiac_phase_for_source_samples(source_samples: np.ndarray, rpeaks: np.ndarray) -> np.ndarray:
    src = np.asarray(source_samples, dtype=int)
    rpeaks = np.asarray(rpeaks, dtype=int)
    rpeaks = rpeaks[np.isfinite(rpeaks)] if np.issubdtype(rpeaks.dtype, np.floating) else rpeaks
    rpeaks = np.sort(rpeaks.astype(int))
    phase = np.full(src.size, np.nan, dtype=float)
    if src.size == 0 or rpeaks.size < 2:
        return phase
    idx = np.searchsorted(rpeaks, src, side="right") - 1
    valid = (idx >= 0) & (idx + 1 < len(rpeaks))
    rr = np.full(src.size, np.nan, dtype=float)
    rr[valid] = rpeaks[idx[valid] + 1] - rpeaks[idx[valid]]
    valid &= rr > 0
    phase[valid] = 2.0 * np.pi * (src[valid] - rpeaks[idx[valid]]) / rr[valid]
    return phase


def mask_source_window(source_samples: np.ndarray, centre_source_sample: int, fs: float, lo: float, hi: float) -> np.ndarray:
    start = int(round(centre_source_sample + lo * fs))
    stop = int(round(centre_source_sample + hi * fs))
    return (source_samples >= start) & (source_samples < stop)


def fixed_window_fraction(labels: np.ndarray, state: str, centre: int, fs: float, lo: float, hi: float) -> float:
    start = max(0, int(round(centre + lo * fs)))
    stop = min(len(labels), int(round(centre + hi * fs)))
    if stop <= start:
        return np.nan
    return float(np.mean(labels[start:stop] == state))


def fixed_window_fraction_source(source_samples: np.ndarray, labels: np.ndarray, state: str, centre_source_sample: int, fs: float, lo: float, hi: float) -> float:
    wm = mask_source_window(source_samples, centre_source_sample, fs, lo, hi)
    if not np.any(wm):
        return np.nan
    return float(np.mean(labels[wm] == state))


def phase_bin_col(b: int) -> str:
    return f"E_phase_{int(b):02d}_coverage"


def phase_bin_cols(cfg: Config) -> List[str]:
    return [phase_bin_col(b) for b in range(int(cfg.n_phase_bins))]


def prev_phase_bin_col(b: int) -> str:
    return f"E_prev_phase_{int(b):02d}_coverage"


def prev_phase_bin_cols(cfg: Config) -> List[str]:
    return [prev_phase_bin_col(b) for b in range(int(cfg.n_phase_bins))]


def cardiac_period_col(phase_lag: str, period: str) -> str:
    prefix = "previous" if str(phase_lag) == PREVIOUS_PHASE_LAG else "current"
    return f"E_{prefix}_{period}_coverage"


def cardiac_period_cols(phase_lag: str) -> List[str]:
    return [cardiac_period_col(phase_lag, p) for p in ("systole", "diastole")]


def cardiac_period_from_col(col: str) -> str:
    if "systole" in str(col):
        return "systole"
    if "diastole" in str(col):
        return "diastole"
    return str(col)


def phase_lag_label(phase_lag: object) -> str:
    if str(phase_lag) == PREVIOUS_PHASE_LAG:
        return "Previous beat (previous R to R)"
    return "Current beat (R to next R)"


def phase_bin_fraction_source(source_samples: np.ndarray, labels: np.ndarray, state: str, r_sample: int, next_r_sample: int, phase_lo: float, phase_hi: float) -> float:
    rr = int(next_r_sample) - int(r_sample)
    if rr <= 0:
        return np.nan
    start = int(round(int(r_sample) + phase_lo * rr))
    stop = int(round(int(r_sample) + phase_hi * rr))
    wm = (source_samples >= start) & (source_samples < stop)
    if not np.any(wm):
        return np.nan
    return float(np.mean(labels[wm] == state))


def cardiac_period_fraction_source(source_samples: np.ndarray, labels: np.ndarray, state: str, r_sample: int, next_r_sample: int, fs: float, period: str) -> float:
    start_rr = int(r_sample)
    stop_rr = int(next_r_sample)
    if stop_rr <= start_rr:
        return np.nan
    sys_start = int(round(start_rr + SYSTOLE_WINDOW_S[0] * fs))
    sys_stop = int(round(start_rr + SYSTOLE_WINDOW_S[1] * fs))
    base = (source_samples >= start_rr) & (source_samples < stop_rr)
    systole = base & (source_samples >= sys_start) & (source_samples < min(sys_stop, stop_rr))
    wm = systole if period == "systole" else (base & ~systole)
    if not np.any(wm):
        return np.nan
    return float(np.mean(labels[wm] == state))


def build_beat_level_table(sequences: Dict[Tuple[str, str], pd.DataFrame], cfg: Config, logger: logging.Logger) -> pd.DataFrame:
    cache_path = cfg.cache_dir / "beat_level_tapping_microstates.csv"
    if cache_path.exists() and not cfg.force_rebuild:
        cached = pd.read_csv(cache_path)
        cache_cols = ["valid_tap_window_lo_s", "valid_tap_window_hi_s", "systole_window_lo_s", "systole_window_hi_s"]
        required_period_cols = cardiac_period_cols(CURRENT_PHASE_LAG) + cardiac_period_cols(PREVIOUS_PHASE_LAG)
        if all(c in cached.columns for c in required_period_cols) and all(c in cached.columns for c in cache_cols):
            lo = pd.to_numeric(cached["valid_tap_window_lo_s"], errors="coerce").dropna()
            hi = pd.to_numeric(cached["valid_tap_window_hi_s"], errors="coerce").dropna()
            sys_lo = pd.to_numeric(cached["systole_window_lo_s"], errors="coerce").dropna()
            sys_hi = pd.to_numeric(cached["systole_window_hi_s"], errors="coerce").dropna()
            if (
                (lo.empty or np.isclose(float(lo.iloc[0]), float(cfg.valid_tap_window_s[0])))
                and (hi.empty or np.isclose(float(hi.iloc[0]), float(cfg.valid_tap_window_s[1])))
                and (sys_lo.empty or np.isclose(float(sys_lo.iloc[0]), float(SYSTOLE_WINDOW_S[0])))
                and (sys_hi.empty or np.isclose(float(sys_hi.iloc[0]), float(SYSTOLE_WINDOW_S[1])))
            ):
                return cached
    rng = np.random.default_rng(cfg.random_seed)
    rows = []
    for (sid, cond), seq in sequences.items():
        if cond not in cfg.accuracy_conditions:
            continue
        source = pick_source_fif(seq)
        if source is None or mne is None:
            logger.warning(f"{sid} {cond}: no source FIF for tapping beat table; skipping.")
            continue
        try:
            raw = mne.io.read_raw_fif(source, preload=False, verbose="ERROR")
            fs = float(raw.info["sfreq"])
            rpeaks_all = find_rpeaks_from_raw(raw, logger)
            taps_all = extract_tap_samples(raw, cond, logger)
        except Exception as exc:
            logger.warning(f"{sid} {cond}: could not load R-peaks/taps ({exc})")
            continue

        source_samples, labels, gfp = valid_sequence_arrays(seq, raw.n_times)
        if source_samples.size < 10:
            logger.warning(f"{sid} {cond}: too few sequence samples after source-sample alignment; skipping.")
            continue
        source_set = set(source_samples.astype(int).tolist())
        rpeaks_all = np.asarray(rpeaks_all, dtype=int)
        rpeaks_all = rpeaks_all[(rpeaks_all >= 0) & (rpeaks_all < raw.n_times)]
        taps_all = np.asarray(taps_all, dtype=int)
        taps_cond = np.asarray([t for t in taps_all if int(t) in source_set], dtype=int)

        # Retain R peaks that are inside the condition sequence, or have at least
        # some sequence samples in the widest requested R-locked window.
        rpeaks_cond = []
        for rp in rpeaks_all:
            if int(rp) in source_set:
                rpeaks_cond.append(int(rp))
                continue
            has_window_samples = any(np.any(mask_source_window(source_samples, int(rp), fs, lo, hi)) for _, lo, hi, _ in WINDOWS_E)
            if has_window_samples:
                rpeaks_cond.append(int(rp))
        rpeaks_cond = np.asarray(sorted(set(rpeaks_cond)), dtype=int)
        if rpeaks_cond.size < 5:
            logger.warning(f"{sid} {cond}: too few condition R-peaks for beat table; skipping.")
            continue

        hit_taps = assign_valid_taps_to_beats(rpeaks_cond, taps_cond, fs, cfg.valid_tap_window_s)
        beat_idx = np.arange(len(rpeaks_cond))
        beat_idx = beat_idx[(beat_idx > 0) & (beat_idx < len(rpeaks_cond) - 1)]
        if beat_idx.size > cfg.max_beats_per_participant_condition:
            beat_idx = np.sort(rng.choice(beat_idx, size=cfg.max_beats_per_participant_condition, replace=False))

        full_index_lookup = {int(rp): i for i, rp in enumerate(rpeaks_all)}
        for bi in beat_idx:
            rp = int(rpeaks_cond[bi])
            full_i = full_index_lookup.get(rp)
            if full_i is None or full_i <= 0 or full_i >= len(rpeaks_all) - 1:
                continue
            prev_r = int(rpeaks_all[full_i - 1])
            next_r = int(rpeaks_all[full_i + 1])
            rr_prev_s = (rp - prev_r) / fs
            rr_next_s = (next_r - rp) / fs
            if not (0.25 <= rr_prev_s <= 2.5 and 0.25 <= rr_next_s <= 2.5):
                continue
            tap = hit_taps.get(int(bi))
            rec = {
                "study": str(seq["study"].iloc[0]),
                "group": str(seq["group"].iloc[0]),
                "participant": sid,
                "condition": cond,
                "block": f"{sid}|{cond}",
                "beat_index": int(bi),
                "r_sample": rp,
                "rr_prev_s": float(rr_prev_s),
                "rr_next_s": float(rr_next_s),
                "heart_rate_bpm_next": float(60.0 / rr_next_s),
                "tap_sample": int(tap) if tap is not None else np.nan,
                "hit": int(tap is not None),
                "miss": int(tap is None),
                "tap_latency_s": float((tap - rp) / fs) if tap is not None else np.nan,
                "n_rpeaks_condition": int(len(rpeaks_cond)),
                "n_taps_condition": int(len(taps_cond)),
                "n_phase_bins": int(cfg.n_phase_bins),
                "valid_tap_window_lo_s": float(cfg.valid_tap_window_s[0]),
                "valid_tap_window_hi_s": float(cfg.valid_tap_window_s[1]),
                "systole_window_lo_s": float(SYSTOLE_WINDOW_S[0]),
                "systole_window_hi_s": float(SYSTOLE_WINDOW_S[1]),
                "source_fif": str(source),
                "uses_source_sample_index": bool("source_sample_index" in seq.columns or "source_sample" in seq.columns),
            }
            for period in ("systole", "diastole"):
                rec[cardiac_period_col(PREVIOUS_PHASE_LAG, period)] = cardiac_period_fraction_source(
                    source_samples, labels, cfg.primary_state, prev_r, rp, fs, period
                )
                rec[cardiac_period_col(CURRENT_PHASE_LAG, period)] = cardiac_period_fraction_source(
                    source_samples, labels, cfg.primary_state, rp, next_r, fs, period
                )
            for col, lo, hi, _ in WINDOWS_E:
                frac = fixed_window_fraction_source(source_samples, labels, cfg.primary_state, rp, fs, lo, hi)
                rec[f"{col}_coverage"] = frac
                rec[f"{col}_any"] = int(frac > 0) if np.isfinite(frac) else np.nan
                wm = mask_source_window(source_samples, rp, fs, lo, hi)
                if np.isfinite(frac) and frac > 0 and np.any(wm & (labels == cfg.primary_state)):
                    rec[f"{col}_gfp"] = float(np.nanmean(gfp[wm & (labels == cfg.primary_state)]))
                else:
                    rec[f"{col}_gfp"] = np.nan
            rows.append(rec)
        logger.info(f"Beat table: {sid} {cond} | beats={len(beat_idx)} taps={len(taps_cond)}")
    beat_df = pd.DataFrame(rows)
    write_csv(beat_df, cache_path)
    return beat_df


def summarise_hit_miss(beat_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if beat_df.empty:
        return pd.DataFrame()
    summary = beat_df.groupby(["study", "group", "participant", "condition"], observed=False, as_index=False).agg(
        n_beats=("hit", "size"),
        n_hits=("hit", "sum"),
        n_misses=("miss", "sum"),
        mean_tap_latency_s=("tap_latency_s", "mean"),
        sd_tap_latency_s=("tap_latency_s", "std"),
        n_taps_condition=("n_taps_condition", "first"),
        n_rpeaks_condition=("n_rpeaks_condition", "first"),
    )
    summary["hit_rate"] = summary["n_hits"] / summary["n_beats"]
    summary["miss_rate"] = summary["n_misses"] / summary["n_beats"]
    summary["false_or_extra_taps"] = summary["n_taps_condition"] - summary["n_hits"]
    summary["false_or_extra_taps_per_beat"] = summary["false_or_extra_taps"] / summary["n_beats"]
    write_csv(summary, cfg.tables_dir / "heartbeat_tapping_hit_miss_by_condition.csv")
    write_hit_miss_descriptives(summary, cfg)
    write_tap_latency_descriptives(beat_df, cfg)
    return summary


def summary_stats(x: pd.Series) -> Dict[str, float]:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(float)
    if vals.size == 0:
        return {"mean": np.nan, "sd": np.nan, "sem": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
    sem = float(sd / math.sqrt(vals.size)) if vals.size > 1 else np.nan
    half = 1.96 * sem if np.isfinite(sem) else np.nan
    return {"mean": mean, "sd": sd, "sem": sem, "ci95_low": mean - half if np.isfinite(half) else np.nan, "ci95_high": mean + half if np.isfinite(half) else np.nan}


def write_hit_miss_descriptives(summary: pd.DataFrame, cfg: Config) -> None:
    desc_rows = []
    for keys, d in summary.groupby(["study", "group", "condition"], observed=False):
        study, group, condition = keys
        hit_stats = summary_stats(d["hit_rate"])
        miss_stats = summary_stats(d["miss_rate"])
        n_beats = float(pd.to_numeric(d["n_beats"], errors="coerce").sum())
        n_hits = float(pd.to_numeric(d["n_hits"], errors="coerce").sum())
        n_misses = float(pd.to_numeric(d["n_misses"], errors="coerce").sum())
        desc_rows.append({
            "study": study,
            "group": group,
            "condition": condition,
            "n_participants": int(d["participant"].nunique()),
            "n_beats": int(n_beats),
            "n_hits": int(n_hits),
            "n_misses": int(n_misses),
            "hit_rate_beat_weighted": n_hits / n_beats if n_beats > 0 else np.nan,
            "miss_rate_beat_weighted": n_misses / n_beats if n_beats > 0 else np.nan,
            "hit_rate_mean_participant": hit_stats["mean"],
            "hit_rate_sd_participant": hit_stats["sd"],
            "hit_rate_sem_participant": hit_stats["sem"],
            "hit_rate_ci95_low": hit_stats["ci95_low"],
            "hit_rate_ci95_high": hit_stats["ci95_high"],
            "miss_rate_mean_participant": miss_stats["mean"],
            "miss_rate_sd_participant": miss_stats["sd"],
            "miss_rate_sem_participant": miss_stats["sem"],
            "miss_rate_ci95_low": miss_stats["ci95_low"],
            "miss_rate_ci95_high": miss_stats["ci95_high"],
        })
    desc = pd.DataFrame(desc_rows)
    write_csv(desc, cfg.tables_dir / "heartbeat_tapping_hit_miss_group_condition_descriptives.csv")

    group_diff_rows = []
    for control_group, case_group, condition_effect in CASE_CONTROL_CONTRASTS:
        for condition, d in summary[summary["group"].isin({control_group, case_group})].groupby("condition", observed=False):
            control = d.loc[d["group"] == control_group, "hit_rate"].dropna()
            case = d.loc[d["group"] == case_group, "hit_rate"].dropna()
            if control.empty or case.empty:
                continue
            test = stats.ttest_ind(case, control, equal_var=False, nan_policy="omit") if len(control) > 1 and len(case) > 1 else None
            group_diff_rows.append({
                "condition_effect": condition_effect,
                "condition": condition,
                "control_group": control_group,
                "case_group": case_group,
                "estimate_direction": f"{case_group}_minus_{control_group}",
                "mean_hit_rate_difference": float(case.mean() - control.mean()),
                "case_mean_hit_rate": float(case.mean()),
                "control_mean_hit_rate": float(control.mean()),
                "n_case": int(len(case)),
                "n_control": int(len(control)),
                "t_value": float(test.statistic) if test is not None else np.nan,
                "p_value": float(test.pvalue) if test is not None else np.nan,
            })
    group_diffs = pd.DataFrame(group_diff_rows)
    if not group_diffs.empty:
        group_diffs = add_fdr(group_diffs, "p_value", "p_fdr_bh_group_hit_rate_differences")
    write_csv(group_diffs, cfg.tables_dir / "heartbeat_tapping_hit_rate_group_differences_by_condition.csv")

    condition_diff_rows = []
    ordered_conditions = [c for c in cfg.accuracy_conditions if c in set(summary["condition"])]
    for (study, group), d in summary.groupby(["study", "group"], observed=False):
        wide = d.pivot_table(index="participant", columns="condition", values="hit_rate", aggfunc="mean")
        for i, cond_a in enumerate(ordered_conditions):
            for cond_b in ordered_conditions[i + 1:]:
                if cond_a not in wide.columns or cond_b not in wide.columns:
                    continue
                diff = (wide[cond_b] - wide[cond_a]).dropna()
                if diff.empty:
                    continue
                test = stats.ttest_1samp(diff, 0.0, nan_policy="omit") if len(diff) > 1 else None
                st = summary_stats(diff)
                condition_diff_rows.append({
                    "study": study,
                    "group": group,
                    "condition_a": cond_a,
                    "condition_b": cond_b,
                    "estimate_direction": f"{cond_b}_minus_{cond_a}",
                    "mean_hit_rate_difference": st["mean"],
                    "sd_difference": st["sd"],
                    "sem_difference": st["sem"],
                    "ci95_low": st["ci95_low"],
                    "ci95_high": st["ci95_high"],
                    "n_participants": int(len(diff)),
                    "t_value": float(test.statistic) if test is not None else np.nan,
                    "p_value": float(test.pvalue) if test is not None else np.nan,
                })
    condition_diffs = pd.DataFrame(condition_diff_rows)
    if not condition_diffs.empty:
        condition_diffs = add_fdr(condition_diffs, "p_value", "p_fdr_bh_condition_hit_rate_differences")
    write_csv(condition_diffs, cfg.tables_dir / "heartbeat_tapping_hit_rate_condition_differences_within_group.csv")
    plot_hit_miss_descriptives(summary, desc, cfg)


def plot_hit_miss_descriptives(summary: pd.DataFrame, desc: pd.DataFrame, cfg: Config) -> None:
    if summary.empty or desc.empty:
        return
    ordered_conditions = [c for c in cfg.accuracy_conditions if c in set(summary["condition"])]
    studies = sorted(desc["study"].dropna().unique())
    fig, axes = plt.subplots(len(studies), 1, figsize=(10, max(3.4, 3.2 * len(studies))), squeeze=False)
    for ax, study in zip(axes.ravel(), studies):
        d = desc[desc["study"] == study].copy()
        seen_groups = [str(g) for g in d["group"].dropna().unique()]
        groups = [g for g in ("ANX", "NANX", "HTN", "NHTN") if g in seen_groups] + sorted(set(seen_groups) - {"ANX", "NANX", "HTN", "NHTN"})
        d["condition"] = pd.Categorical(d["condition"], categories=ordered_conditions, ordered=True)
        d["group"] = pd.Categorical(d["group"], categories=groups, ordered=True)
        d = d.sort_values(["group", "condition"])
        x = np.arange(len(d))
        hit = d["hit_rate_beat_weighted"].to_numpy(float)
        miss = d["miss_rate_beat_weighted"].to_numpy(float)
        ax.bar(x, hit, color="#2563eb", label="Hit")
        ax.bar(x, miss, bottom=hit, color="#d1d5db", label="Miss")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r.group}\n{r.condition}" for r in d.itertuples()], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Beat fraction")
        ax.set_title(f"{study}: beat-level hits vs misses")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "heartbeat_tapping_hits_misses_by_group_condition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    plot_df = (
        summary.groupby(["study", "group", "condition"], observed=False, as_index=False)
        .agg(mean=("hit_rate", "mean"), sem=("hit_rate", "sem"), n=("participant", "nunique"))
    )
    fig, axes = plt.subplots(len(studies), 1, figsize=(8.5, max(3.2, 3.0 * len(studies))), sharex=True, squeeze=False)
    x_base = np.arange(len(ordered_conditions))
    for ax, study in zip(axes.ravel(), studies):
        d = plot_df[plot_df["study"] == study]
        seen_groups = [str(g) for g in d["group"].dropna().unique()]
        groups = [g for g in ("ANX", "NANX", "HTN", "NHTN") if g in seen_groups] + sorted(set(seen_groups) - {"ANX", "NANX", "HTN", "NHTN"})
        offsets = np.linspace(-0.18, 0.18, max(1, len(groups)))
        for offset, group in zip(offsets, groups):
            gd = d[d["group"] == group].set_index("condition").reindex(ordered_conditions)
            y = gd["mean"].to_numpy(float)
            sem = gd["sem"].fillna(0).to_numpy(float)
            ax.errorbar(x_base + offset, y, yerr=1.96 * sem, marker="o", capsize=3, label=group)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Hit rate")
        ax.set_title(f"{study}: participant hit rate by group and condition")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, ncols=max(1, len(groups)))
    axes.ravel()[-1].set_xticks(x_base)
    axes.ravel()[-1].set_xticklabels(ordered_conditions)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "heartbeat_tapping_hit_rate_group_condition.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def tap_latency_summary_row(scope: str, scope_label: str, group: str, data: pd.DataFrame) -> Dict[str, object]:
    hits = data.dropna(subset=["tap_latency_s"]).copy()
    part = hits.groupby("participant", observed=False)["tap_latency_s"].mean()
    st = summary_stats(part)
    raw = pd.to_numeric(hits["tap_latency_s"], errors="coerce").dropna()
    return {
        "analysis_scope": scope,
        "scope_label": scope_label,
        "group": group,
        "n_participants": int(part.dropna().size),
        "n_hit_taps": int(raw.size),
        "mean_tap_to_previous_r_s": st["mean"],
        "mean_tap_to_previous_r_ms": st["mean"] * 1000.0 if np.isfinite(st["mean"]) else np.nan,
        "sd_participant_mean_s": st["sd"],
        "sem_participant_mean_s": st["sem"],
        "ci95_low_s": st["ci95_low"],
        "ci95_high_s": st["ci95_high"],
        "ci95_low_ms": st["ci95_low"] * 1000.0 if np.isfinite(st["ci95_low"]) else np.nan,
        "ci95_high_ms": st["ci95_high"] * 1000.0 if np.isfinite(st["ci95_high"]) else np.nan,
        "raw_mean_s": float(raw.mean()) if raw.size else np.nan,
        "raw_median_s": float(raw.median()) if raw.size else np.nan,
    }


def write_tap_latency_descriptives(beat_df: pd.DataFrame, cfg: Config) -> None:
    if beat_df.empty or "tap_latency_s" not in beat_df.columns:
        return
    rows = [tap_latency_summary_row("global", "global population", "all", beat_df)]
    for group, gd in beat_df.groupby("group", observed=False):
        rows.append(tap_latency_summary_row("group", f"group {group}", str(group), gd))
    out = pd.DataFrame(rows)
    write_csv(out, cfg.tables_dir / "heartbeat_tapping_tap_latency_to_previous_r_descriptives.csv")
    plot_tap_latency_descriptives(out, cfg)


def build_latency_consistency_table(beat_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    cols = phase_bin_cols(cfg)
    if beat_df.empty or "tap_latency_s" not in beat_df.columns or not all(c in beat_df.columns for c in cols):
        return pd.DataFrame()
    hits = beat_df.dropna(subset=["tap_latency_s"]).copy()
    if hits.empty:
        return pd.DataFrame()
    rows = []
    for keys, d in hits.groupby(["study", "group", "participant", "condition"], observed=False):
        study, group, participant, condition = keys
        lat = pd.to_numeric(d["tap_latency_s"], errors="coerce").dropna()
        if lat.size < 3:
            continue
        var = float(lat.var(ddof=1))
        if not np.isfinite(var) or var <= 0:
            continue
        rec = {
            "study": study,
            "group": group,
            "participant": participant,
            "condition": condition,
            "n_hit_taps": int(lat.size),
            "mean_tap_latency_s": float(lat.mean()),
            "latency_variance_s2": var,
            "latency_inverse_variance": float(1.0 / var),
            "log_latency_inverse_variance": float(np.log(1.0 / var)),
        }
        for col in cols:
            rec[col] = float(pd.to_numeric(d[col], errors="coerce").mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def run_latency_consistency_phase_models(
    coef_rows: List[Dict[str, object]],
    data: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    scope: str,
    scope_label: str,
    study: str,
    group: str,
) -> None:
    cols = phase_bin_cols(cfg)
    if data.empty or data["participant"].nunique() < cfg.min_participants or not all(c in data.columns for c in cols):
        return
    base_cols = ["log_latency_inverse_variance", "participant", "study", "group", "condition", "n_hit_taps"]
    n_bins = float(cfg.n_phase_bins)
    for b, col in enumerate(cols):
        d = data[base_cols + [col]].dropna().copy()
        if d.empty or d["log_latency_inverse_variance"].nunique() < 2 or d[col].nunique() < 2:
            continue
        d[f"{col}_z"] = zscore_series(d[col])
        d["log_n_hit_taps"] = np.log(pd.to_numeric(d["n_hit_taps"], errors="coerce").clip(lower=1))
        rhs = [f"{col}_z"]
        if d["group"].nunique() > 1 and d["condition"].nunique() > 1:
            rhs.append("C(group) * C(condition)")
        elif d["group"].nunique() > 1:
            rhs.append("C(group)")
        elif d["condition"].nunique() > 1:
            rhs.append("C(condition)")
        rhs.append("log_n_hit_taps")
        formula = "log_latency_inverse_variance ~ " + " + ".join(rhs)
        try:
            fit = fit_gee_gaussian(formula, d)
            for term in fit.params.index:
                est = float(fit.params[term])
                se = float(fit.bse[term])
                coef_rows.append({
                    "analysis_scope": scope,
                    "scope_label": scope_label,
                    "study": study,
                    "group": group,
                    "phase_bin": int(b),
                    "phase_lo_fraction": float(b / n_bins),
                    "phase_mid_fraction": float((b + 0.5) / n_bins),
                    "phase_hi_fraction": float((b + 1.0) / n_bins),
                    "window": col,
                    "window_label": f"RR phase {b / n_bins:.3f}-{(b + 1.0) / n_bins:.3f}",
                    "model": "cardiac_phase_latency_consistency_gee",
                    "inference": "gee_gaussian",
                    "outcome": "log_latency_inverse_variance",
                    "term": term,
                    "effect_type": beat_effect_type(str(term)),
                    "estimate": est,
                    "std_error": se,
                    "z_value": float(fit.tvalues[term]),
                    "p_value": float(fit.pvalues[term]),
                    "interval_low": est - 1.96 * se,
                    "interval_high": est + 1.96 * se,
                    "interval_kind": "wald_95_ci",
                    "formula": formula,
                    "n_participants": int(d["participant"].nunique()),
                    "n_rows": int(len(d)),
                })
        except Exception as exc:
            logger.warning(f"Latency consistency cardiac phase GEE failed: {scope_label} bin {b}: {exc}")


def run_latency_consistency_case_control_models(
    coef_rows: List[Dict[str, object]],
    data: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    control_group: str,
    case_group: str,
    condition_effect: str,
) -> None:
    cols = phase_bin_cols(cfg)
    d0 = data[data["group"].isin({control_group, case_group})].copy()
    if d0.empty or d0["group"].nunique() < 2 or d0["participant"].nunique() < cfg.min_participants or not all(c in d0.columns for c in cols):
        return
    if d0.groupby("group", observed=False)["participant"].nunique().min() < 2:
        return
    base_cols = ["log_latency_inverse_variance", "participant", "study", "group", "condition", "n_hit_taps"]
    scope_label = f"{case_group} vs {control_group}"
    study = str(d0["study"].dropna().iloc[0]) if d0["study"].dropna().nunique() == 1 else "mixed"
    n_bins = float(cfg.n_phase_bins)
    for b, col in enumerate(cols):
        d = d0[base_cols + [col]].dropna().copy()
        if d.empty or d["log_latency_inverse_variance"].nunique() < 2 or d[col].nunique() < 2:
            continue
        d[f"{col}_z"] = zscore_series(d[col])
        d["log_n_hit_taps"] = np.log(pd.to_numeric(d["n_hit_taps"], errors="coerce").clip(lower=1))
        group_term = f"C(group, Treatment(reference={control_group!r}))"
        rhs = [f"{group_term} * ({col}_z)"]
        if d["condition"].nunique() > 1:
            rhs.append("C(condition)")
        rhs.append("log_n_hit_taps")
        formula = "log_latency_inverse_variance ~ " + " + ".join(rhs)
        try:
            fit = fit_gee_gaussian(formula, d)
            for term in fit.params.index:
                est = float(fit.params[term])
                se = float(fit.bse[term])
                coef_rows.append({
                    "analysis_scope": "case_control_contrast",
                    "scope_label": scope_label,
                    "study": study,
                    "group": scope_label,
                    "control_group": control_group,
                    "case_group": case_group,
                    "condition_effect": condition_effect,
                    "estimate_direction": f"{case_group}_minus_{control_group}",
                    "phase_bin": int(b),
                    "phase_lo_fraction": float(b / n_bins),
                    "phase_mid_fraction": float((b + 0.5) / n_bins),
                    "phase_hi_fraction": float((b + 1.0) / n_bins),
                    "window": col,
                    "window_label": f"RR phase {b / n_bins:.3f}-{(b + 1.0) / n_bins:.3f}",
                    "model": "cardiac_phase_latency_consistency_gee_case_control",
                    "inference": "gee_gaussian",
                    "outcome": "log_latency_inverse_variance",
                    "term": term,
                    "effect_type": beat_effect_type(str(term)),
                    "estimate": est,
                    "std_error": se,
                    "z_value": float(fit.tvalues[term]),
                    "p_value": float(fit.pvalues[term]),
                    "interval_low": est - 1.96 * se,
                    "interval_high": est + 1.96 * se,
                    "interval_kind": "wald_95_ci",
                    "formula": formula,
                    "n_participants": int(d["participant"].nunique()),
                    "n_rows": int(len(d)),
                })
        except Exception as exc:
            logger.warning(f"Latency consistency case-control GEE failed: {scope_label} bin {b}: {exc}")


def analyse_latency_consistency_cardiac_phase(beat_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    consistency = build_latency_consistency_table(beat_df, cfg)
    write_csv(consistency, cfg.tables_dir / "heartbeat_tapping_latency_consistency_by_condition.csv")
    if consistency.empty:
        return
    coef_rows: List[Dict[str, object]] = []
    run_latency_consistency_phase_models(coef_rows, consistency, cfg, logger, "global", "global population", "all", "all")
    for group, gd in consistency.groupby("group", observed=False):
        studies = sorted(str(x) for x in gd["study"].dropna().unique())
        study = studies[0] if len(studies) == 1 else "mixed"
        run_latency_consistency_phase_models(coef_rows, gd, cfg, logger, "group", f"group {group}", study, str(group))
    for control_group, case_group, condition_effect in CASE_CONTROL_CONTRASTS:
        run_latency_consistency_case_control_models(coef_rows, consistency, cfg, logger, control_group, case_group, condition_effect)
    if not coef_rows:
        return
    coefs = pd.DataFrame(coef_rows)
    coefs = add_fdr(coefs, "p_value", "p_fdr_bh_across_latency_consistency_phase_models")
    coefs = add_groupwise_fdr(
        coefs,
        "p_value",
        ["analysis_scope", "scope_label", "model", "effect_type"],
        "p_fdr_bh_within_latency_consistency_curve",
    )
    coefs = add_zero_deviation_flags(coefs, "p_fdr_bh_within_latency_consistency_curve")
    write_csv(coefs, cfg.tables_dir / "cardiac_phase_latency_consistency_E_effect_coefficients.csv")
    write_csv(
        coefs[(coefs["analysis_scope"] == "case_control_contrast") & (coefs["effect_type"] == "case_control_E_effect_difference")].copy(),
        cfg.tables_dir / "cardiac_phase_latency_consistency_E_effect_case_control_contrasts.csv",
    )
    plot_latency_consistency_phase_effects(coefs, cfg)


def plot_tap_latency_descriptives(desc: pd.DataFrame, cfg: Config) -> None:
    d = desc[desc["n_participants"] > 0].copy()
    if d.empty:
        return
    order = ["global population"] + [f"group {g}" for g in ("ANX", "NANX", "HTN", "NHTN")]
    labels = [label for label in order if label in set(d["scope_label"])]
    labels += sorted(set(d["scope_label"]) - set(labels))
    d = d.set_index("scope_label").reindex(labels).dropna(subset=["mean_tap_to_previous_r_ms"])
    if d.empty:
        return
    y = np.arange(len(d))
    mean = d["mean_tap_to_previous_r_ms"].to_numpy(float)
    lo = d["ci95_low_ms"].to_numpy(float)
    hi = d["ci95_high_ms"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(d) + 1.5)))
    ax.errorbar(mean, y, xerr=[mean - lo, hi - mean], fmt="o", capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d.index)
    ax.set_xlabel("Tap latency from previous R peak (ms)")
    ax.set_title("Tap timing relative to previous R peak")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "heartbeat_tapping_tap_latency_to_previous_r_global_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_binomial_count_model(df: pd.DataFrame, rhs_formula: str, success_col: str, failure_col: str, cluster_col: str) -> Tuple[Optional[object], str, Optional[pd.DataFrame]]:
    if patsy is None:
        return None, "patsy_unavailable", None
    d = df.dropna(subset=[success_col, failure_col, cluster_col]).copy()
    if d.empty:
        return None, "empty", None
    y = d[[success_col, failure_col]].to_numpy(float)
    X = patsy.dmatrix(rhs_formula, d, return_type="dataframe")
    try:
        fit = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster",
            cov_kwds={"groups": d[cluster_col].astype(str).to_numpy()},
        )
        return fit, "glm_binomial_clustered", X
    except Exception:
        return None, "failed", X


def analyse_tapping_accuracy_linkage(hit_summary: pd.DataFrame, state_metrics: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    if hit_summary.empty or state_metrics.empty:
        return
    ms_wide = state_metrics.pivot_table(
        index=["study", "group", "participant", "condition"],
        columns="microstate",
        values=["coverage", "gfp", "occurrence_rate_hz", "mean_duration_ms"],
        aggfunc="mean",
    )
    ms_wide.columns = [f"{metric}_{state}" for metric, state in ms_wide.columns]
    ms_wide = ms_wide.reset_index()
    merged = hit_summary.merge(ms_wide, on=["study", "group", "participant", "condition"], how="inner")
    write_csv(merged, cfg.tables_dir / "heartbeat_tapping_with_microstate_variables.csv")
    if merged.empty:
        return

    coef_rows = []
    # Primary association: all states' coverage and GFP separately, not all in one overfit model.
    candidate_cols = [c for c in merged.columns if re.match(r"^(coverage|gfp|occurrence_rate_hz|mean_duration_ms)_[A-Z0-9]+$", c)]
    for study, sd in merged.groupby("study", observed=False):
        if sd["participant"].nunique() < cfg.min_participants:
            continue
        for col in candidate_cols:
            d = sd[["participant", "group", "condition", "n_hits", "n_misses", col]].dropna().copy()
            if d.empty or d[col].nunique() < 2:
                continue
            d[f"{col}_z"] = zscore_series(d[col])
            rhs = f"1 + C(condition) + {col}_z"
            if d["group"].nunique() > 1:
                rhs = f"1 + C(group) * C(condition) + {col}_z"
            fit, kind, _ = fit_binomial_count_model(d, rhs, "n_hits", "n_misses", "participant")
            if fit is None:
                logger.warning(f"Binomial tapping model failed: {study} {col} ({kind})")
                continue
            for term in fit.params.index:
                coef_rows.append({
                    "study": study,
                    "predictor": col,
                    "term": term,
                    "estimate_log_odds": float(fit.params[term]),
                    "std_error": float(fit.bse[term]),
                    "z_value": float(fit.tvalues[term]),
                    "p_value": float(fit.pvalues[term]),
                    "model_kind": kind,
                    "n_participants": int(d["participant"].nunique()),
                    "n_rows": int(len(d)),
                })
    if coef_rows:
        coefs = pd.DataFrame(coef_rows)
        coefs = add_fdr(coefs, "p_value", "p_fdr_bh_across_accuracy_linkage")
        write_csv(coefs, cfg.tables_dir / "heartbeat_tapping_microstate_linkage_binomial_coefficients.csv")
        plot_accuracy_linkage_summary(coefs, cfg)


def plot_accuracy_linkage_summary(coefs: pd.DataFrame, cfg: Config) -> None:
    d = coefs[coefs["term"].str.endswith("_z")].copy()
    if d.empty:
        return
    d = d.sort_values("p_value").head(40)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(d))))
    ax.errorbar(d["estimate_log_odds"], y, xerr=1.96 * d["std_error"], fmt="o", capsize=3)
    ax.axvline(0, lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.study}: {r.predictor}" for r in d.itertuples()], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Log-odds estimate for hit rate per 1 SD microstate predictor")
    ax.set_title("Heartbeat-tapping hit rate associations with microstate variables")
    fig.tight_layout()
    fig.savefig(cfg.figures_dir / "heartbeat_tapping_microstate_linkage_coefficients.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fit_gee_binomial(formula: str, data: pd.DataFrame, group_col: str = "participant"):
    d = data.copy()
    d[group_col] = d[group_col].astype(str)
    model = smf.gee(
        formula,
        groups=d[group_col],
        data=d,
        family=sm.families.Binomial(),
        cov_struct=Exchangeable(),
    )
    return model.fit(maxiter=100)


def fit_gee_gaussian(formula: str, data: pd.DataFrame, group_col: str = "participant"):
    d = data.copy()
    d[group_col] = d[group_col].astype(str)
    model = smf.gee(
        formula,
        groups=d[group_col],
        data=d,
        family=sm.families.Gaussian(),
        cov_struct=Independence(),
    )
    return model.fit(maxiter=100)


def fit_bayes_binomial_mixed(formula: str, data: pd.DataFrame, group_col: str = "participant"):
    if BinomialBayesMixedGLM is None:
        raise RuntimeError("BinomialBayesMixedGLM unavailable in this statsmodels install")
    d = data.copy()
    d[group_col] = d[group_col].astype(str)
    model = BinomialBayesMixedGLM.from_formula(
        formula,
        {group_col: f"0 + C({group_col})"},
        d,
        vcp_p=1,
        fe_p=2,
    )
    return model.fit_vb(minim_opts={"maxiter": 500}, scale_fe=False)


def beat_model_formula(pred_terms: Sequence[str], data: pd.DataFrame) -> str:
    rhs = list(pred_terms)
    if data["group"].nunique() > 1 and data["condition"].nunique() > 1:
        rhs.append("C(group) * C(condition)")
    elif data["group"].nunique() > 1:
        rhs.append("C(group)")
    elif data["condition"].nunique() > 1:
        rhs.append("C(condition)")
    rhs.append("rr_prev_s")
    return "hit ~ " + " + ".join(rhs)


def case_control_beat_formula(pred_terms: Sequence[str], data: pd.DataFrame, control_group: str) -> str:
    group_term = f"C(group, Treatment(reference={control_group!r}))"
    pred_rhs = " + ".join(pred_terms)
    rhs = [f"{group_term} * ({pred_rhs})"]
    if data["condition"].nunique() > 1:
        rhs.append("C(condition)")
    rhs.append("rr_prev_s")
    return "hit ~ " + " + ".join(rhs)


def beat_effect_type(term: str) -> str:
    has_e = (
        any(f"{col}_coverage_z" in term for col, _, _, _ in WINDOWS_E)
        or bool(re.search(r"E_(?:prev_)?phase_\d+_coverage_z", str(term)))
        or bool(re.search(r"E_(?:current|previous)_(?:systole|diastole)_coverage_z", str(term)))
    )
    has_group = "C(group" in term
    if has_e and has_group and ":" in term:
        return "case_control_E_effect_difference"
    if has_group:
        return "clinical_group_main_effect"
    if has_e:
        return "E_effect"
    return "covariate"


def append_gee_rows(coef_rows: List[Dict[str, object]], fit, meta: Dict[str, object]) -> None:
    for term in fit.params.index:
        est = float(fit.params[term])
        se = float(fit.bse[term])
        coef_rows.append({
            **meta,
            "term": term,
            "effect_type": beat_effect_type(str(term)),
            "estimate_log_odds": est,
            "std_error": se,
            "z_value": float(fit.tvalues[term]),
            "p_value": float(fit.pvalues[term]),
            "interval_low": est - 1.96 * se,
            "interval_high": est + 1.96 * se,
            "interval_kind": "wald_95_ci",
        })


def append_bayes_rows(coef_rows: List[Dict[str, object]], fit, meta: Dict[str, object]) -> None:
    for term, est, se in zip(fit.model.exog_names, fit.fe_mean, fit.fe_sd):
        est = float(est)
        se = float(se)
        z = est / se if np.isfinite(se) and se > 0 else np.nan
        coef_rows.append({
            **meta,
            "term": term,
            "effect_type": beat_effect_type(str(term)),
            "estimate_log_odds": est,
            "std_error": se,
            "z_value": float(z) if np.isfinite(z) else np.nan,
            "p_value": float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
            "interval_low": est - 1.96 * se,
            "interval_high": est + 1.96 * se,
            "interval_kind": "posterior_normal_95_interval",
        })


def analyse_beat_by_beat_E_windows(beat_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    write_csv(beat_df, cfg.tables_dir / "beat_level_tapping_microstate_E_windows.csv")
    for path in (
        cfg.tables_dir / "beat_by_beat_E_window_gee_coefficients.csv",
        cfg.tables_dir / "beat_by_beat_E_window_case_control_effects.csv",
        cfg.figures_dir / "beat_by_beat_E_window_gee_coefficients.png",
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(f"Could not remove legacy fixed-window beat output {path}: {exc}")
    logger.info("Skipping legacy fixed-window joint/single beat models; using systole/diastole empirical-Bayes analysis instead.")


def run_cardiac_phase_beat_models(
    coef_rows: List[Dict[str, object]],
    data: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    scope: str,
    scope_label: str,
    study: str,
    group: str,
    method: str,
    cols: Optional[Sequence[str]] = None,
    phase_lag: str = CURRENT_PHASE_LAG,
) -> None:
    cols = list(cols) if cols is not None else phase_bin_cols(cfg)
    if data["participant"].nunique() < cfg.min_participants or not all(c in data.columns for c in cols):
        return
    base_cols = ["hit", "participant", "study", "group", "condition", "rr_prev_s", "heart_rate_bpm_next"]
    fit_fn = fit_gee_binomial if method == "gee" else fit_bayes_binomial_mixed
    n_bins = float(len(cols))

    for b, col in enumerate(cols):
        period = cardiac_period_from_col(col)
        d = data[base_cols + [col]].dropna().copy()
        if d.empty or d["hit"].nunique() < 2 or d[col].nunique() < 2:
            continue
        d[f"{col}_z"] = zscore_series(d[col])
        formula = beat_model_formula([f"{col}_z"], d)
        try:
            fit = fit_fn(formula, d)
            append = append_gee_rows if method == "gee" else append_bayes_rows
            append(coef_rows, fit, {
                "analysis_scope": scope,
                "scope_label": scope_label,
                "study": study,
                "group": group,
                "phase_lag": phase_lag,
                "phase_lag_label": phase_lag_label(phase_lag),
                "phase_bin": int(b),
                "phase_lo_fraction": float(b / n_bins),
                "phase_mid_fraction": float((b + 0.5) / n_bins),
                "phase_hi_fraction": float((b + 1.0) / n_bins),
                "cardiac_period": period,
                "window": col,
                "window_label": period.capitalize(),
                "model": f"cardiac_period_{method}" if method == "gee" else "cardiac_period_bayes_mixed_logit",
                "inference": method,
                "formula": formula,
                "n_participants": int(d["participant"].nunique()),
                "n_beats": int(len(d)),
            })
        except Exception as exc:
            logger.warning(f"Cardiac phase beat {method} failed: {scope_label} {phase_lag} bin {b}: {exc}")


def run_cardiac_phase_case_control_gee(
    coef_rows: List[Dict[str, object]],
    data: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    control_group: str,
    case_group: str,
    condition_effect: str,
    cols: Optional[Sequence[str]] = None,
    phase_lag: str = CURRENT_PHASE_LAG,
) -> None:
    cols = list(cols) if cols is not None else phase_bin_cols(cfg)
    d0 = data[data["group"].isin({control_group, case_group})].copy()
    if d0["group"].nunique() < 2 or d0["participant"].nunique() < cfg.min_participants or not all(c in d0.columns for c in cols):
        return
    if d0.groupby("group", observed=False)["participant"].nunique().min() < 2:
        return
    base_cols = ["hit", "participant", "study", "group", "condition", "rr_prev_s", "heart_rate_bpm_next"]
    scope_label = f"{case_group} vs {control_group}"
    study = str(d0["study"].dropna().iloc[0]) if d0["study"].dropna().nunique() == 1 else "mixed"
    n_bins = float(len(cols))

    for b, col in enumerate(cols):
        period = cardiac_period_from_col(col)
        d = d0[base_cols + [col]].dropna().copy()
        if d.empty or d["hit"].nunique() < 2 or d[col].nunique() < 2:
            continue
        d[f"{col}_z"] = zscore_series(d[col])
        formula = case_control_beat_formula([f"{col}_z"], d, control_group)
        try:
            fit = fit_gee_binomial(formula, d)
            append_gee_rows(coef_rows, fit, {
                "analysis_scope": "case_control_contrast",
                "scope_label": scope_label,
                "study": study,
                "group": scope_label,
                "control_group": control_group,
                "case_group": case_group,
                "condition_effect": condition_effect,
                "estimate_direction": f"{case_group}_minus_{control_group}",
                "phase_lag": phase_lag,
                "phase_lag_label": phase_lag_label(phase_lag),
                "phase_bin": int(b),
                "phase_lo_fraction": float(b / n_bins),
                "phase_mid_fraction": float((b + 0.5) / n_bins),
                "phase_hi_fraction": float((b + 1.0) / n_bins),
                "cardiac_period": period,
                "window": col,
                "window_label": period.capitalize(),
                "model": "cardiac_period_gee_case_control",
                "inference": "gee",
                "formula": formula,
                "n_participants": int(d["participant"].nunique()),
                "n_beats": int(len(d)),
            })
        except Exception as exc:
            logger.warning(f"Cardiac phase case-control GEE failed: {scope_label} {phase_lag} bin {b}: {exc}")


def run_cardiac_phase_case_control_bayes(
    coef_rows: List[Dict[str, object]],
    data: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    control_group: str,
    case_group: str,
    condition_effect: str,
    cols: Optional[Sequence[str]] = None,
    phase_lag: str = CURRENT_PHASE_LAG,
) -> None:
    cols = list(cols) if cols is not None else phase_bin_cols(cfg)
    d0 = data[data["group"].isin({control_group, case_group})].copy()
    if d0["group"].nunique() < 2 or d0["participant"].nunique() < cfg.min_participants or not all(c in d0.columns for c in cols):
        return
    if d0.groupby("group", observed=False)["participant"].nunique().min() < 2:
        return
    base_cols = ["hit", "participant", "study", "group", "condition", "rr_prev_s", "heart_rate_bpm_next"]
    scope_label = f"{case_group} vs {control_group}"
    study = str(d0["study"].dropna().iloc[0]) if d0["study"].dropna().nunique() == 1 else "mixed"
    n_bins = float(len(cols))

    for b, col in enumerate(cols):
        period = cardiac_period_from_col(col)
        d = d0[base_cols + [col]].dropna().copy()
        if d.empty or d["hit"].nunique() < 2 or d[col].nunique() < 2:
            continue
        d[f"{col}_z"] = zscore_series(d[col])
        formula = case_control_beat_formula([f"{col}_z"], d, control_group)
        try:
            fit = fit_bayes_binomial_mixed(formula, d)
            append_bayes_rows(coef_rows, fit, {
                "analysis_scope": "case_control_contrast",
                "scope_label": scope_label,
                "study": study,
                "group": scope_label,
                "control_group": control_group,
                "case_group": case_group,
                "condition_effect": condition_effect,
                "estimate_direction": f"{case_group}_minus_{control_group}",
                "phase_lag": phase_lag,
                "phase_lag_label": phase_lag_label(phase_lag),
                "phase_bin": int(b),
                "phase_lo_fraction": float(b / n_bins),
                "phase_mid_fraction": float((b + 0.5) / n_bins),
                "phase_hi_fraction": float((b + 1.0) / n_bins),
                "cardiac_period": period,
                "window": col,
                "window_label": period.capitalize(),
                "model": "cardiac_period_bayes_case_control",
                "inference": "bayes",
                "formula": formula,
                "n_participants": int(d["participant"].nunique()),
                "n_beats": int(len(d)),
            })
        except Exception as exc:
            logger.warning(f"Cardiac phase case-control Bayes failed: {scope_label} {phase_lag} bin {b}: {exc}")


def run_cardiac_phase_omnibus_gee(
    rows: List[Dict[str, object]],
    data: pd.DataFrame,
    cfg: Config,
    logger: logging.Logger,
    scope: str,
    scope_label: str,
    study: str,
    group: str,
    cols: Optional[Sequence[str]] = None,
    phase_lag: str = CURRENT_PHASE_LAG,
) -> None:
    cols = list(cols) if cols is not None else phase_bin_cols(cfg)
    if data["participant"].nunique() < cfg.min_participants or not all(c in data.columns for c in cols):
        return
    base_cols = ["hit", "participant", "study", "group", "condition", "rr_prev_s", "heart_rate_bpm_next"]
    d = data[base_cols + cols].dropna().copy()
    if d.empty or d["hit"].nunique() < 2:
        return
    pred_terms = []
    for col in cols:
        if d[col].nunique() < 2:
            continue
        zcol = f"{col}_z"
        d[zcol] = zscore_series(d[col])
        pred_terms.append(zcol)
    if not pred_terms:
        return
    formula = beat_model_formula(pred_terms, d)
    try:
        fit = fit_gee_binomial(formula, d)
        terms = [t for t in pred_terms if t in fit.params.index]
        if not terms:
            return
        R = np.zeros((len(terms), len(fit.params)), dtype=float)
        for i, term in enumerate(terms):
            R[i, list(fit.params.index).index(term)] = 1.0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="covariance of constraints does not have full rank")
            test = fit.wald_test(R, scalar=True)
        rows.append({
            "analysis_scope": scope,
            "scope_label": scope_label,
            "study": study,
            "group": group,
            "phase_lag": phase_lag,
            "phase_lag_label": phase_lag_label(phase_lag),
            "model": "cardiac_period_gee_omnibus",
            "test": "systole_diastole_E_terms_zero",
            "n_phase_terms": int(len(terms)),
            "chi2": float(np.asarray(test.statistic).squeeze()),
            "df": int(getattr(test, "df_constraints", len(terms))),
            "p_value": float(test.pvalue),
            "n_participants": int(d["participant"].nunique()),
            "n_beats": int(len(d)),
            "formula": formula,
        })
    except Exception as exc:
        logger.warning(f"Cardiac phase omnibus GEE failed: {scope_label} {phase_lag}: {exc}")


def append_cardiac_phase_empirical_bayes_rows(coef_rows: List[Dict[str, object]], logger: logging.Logger) -> None:
    gee = pd.DataFrame(coef_rows)
    if gee.empty:
        return
    for col in ("estimate_direction", "control_group", "case_group", "condition_effect"):
        if col not in gee.columns:
            gee[col] = np.nan
    d = gee[
        (gee["inference"] == "gee")
        & (gee["effect_type"].isin(["E_effect", "case_control_E_effect_difference"]))
    ].copy()
    if d.empty:
        return
    curve_cols = [
        "analysis_scope",
        "scope_label",
        "study",
        "group",
        "phase_lag",
        "effect_type",
        "estimate_direction",
        "control_group",
        "case_group",
        "condition_effect",
    ]
    n_added = 0
    for _, gd in d.groupby(curve_cols, dropna=False, observed=False):
        gd = gd.sort_values("phase_bin").copy()
        y = pd.to_numeric(gd["estimate_log_odds"], errors="coerce").to_numpy(float)
        se = pd.to_numeric(gd["std_error"], errors="coerce").to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(se) & (se > 0)
        if int(ok.sum()) < 2:
            continue
        gd = gd.loc[ok].copy()
        y = y[ok]
        se = se[ok]
        var = np.maximum(se ** 2, 1e-8)
        w = 1.0 / var
        mu = float(np.sum(w * y) / np.sum(w))
        q = float(np.sum(w * (y - mu) ** 2))
        c = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
        tau2 = max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0
        mu_var = float(1.0 / np.sum(w))

        if tau2 <= 1e-10:
            post_mean = np.full_like(y, mu, dtype=float)
            post_var = np.full_like(y, mu_var, dtype=float)
        else:
            shrink_var = 1.0 / (1.0 / var + 1.0 / tau2)
            post_mean = shrink_var * (y / var + mu / tau2)
            post_var = shrink_var + mu_var

        post_sd = np.sqrt(np.maximum(post_var, 1e-8))
        for row, est, sd, gee_est, gee_se in zip(gd.to_dict("records"), post_mean, post_sd, y, se):
            z = float(est / sd) if np.isfinite(sd) and sd > 0 else np.nan
            model = "cardiac_period_bayes_gee_case_control" if row.get("analysis_scope") == "case_control_contrast" else "cardiac_period_bayes_gee"
            row.update({
                "model": model,
                "inference": "bayes",
                "formula": "empirical_bayes_systole_diastole_from_gee",
                "gee_estimate_log_odds": float(gee_est),
                "gee_std_error": float(gee_se),
                "estimate_log_odds": float(est),
                "std_error": float(sd),
                "z_value": z,
                "p_value": float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
                "interval_low": float(est - 1.96 * sd),
                "interval_high": float(est + 1.96 * sd),
                "interval_kind": "empirical_bayes_curve_95_interval",
                "bayes_curve_mean_log_odds": mu,
                "bayes_curve_tau2": tau2,
                "n_phase_bins_in_curve": int(len(y)),
                "n_cardiac_periods_in_curve": int(len(y)),
            })
            coef_rows.append(row)
            n_added += 1
    logger.info(f"Added {n_added} empirical-Bayes cardiac phase rows from GEE curves.")


def add_zero_deviation_flags(coefs: pd.DataFrame, fdr_col: str) -> pd.DataFrame:
    out = coefs.copy()
    if {"interval_low", "interval_high"}.issubset(out.columns):
        lo = pd.to_numeric(out["interval_low"], errors="coerce")
        hi = pd.to_numeric(out["interval_high"], errors="coerce")
        out["significant_deviation_from_zero"] = ((lo > 0) | (hi < 0)).fillna(False)
    else:
        out["significant_deviation_from_zero"] = False
    if fdr_col in out.columns:
        out["significant_fdr_05"] = pd.to_numeric(out[fdr_col], errors="coerce") < 0.05
        out["significant_ci_and_fdr_05"] = out["significant_deviation_from_zero"] & out["significant_fdr_05"].fillna(False)
    return out


def cardiac_phase_significant_clusters(coefs: pd.DataFrame) -> pd.DataFrame:
    required = {"phase_bin", "phase_lo_fraction", "phase_hi_fraction", "estimate_log_odds", "p_value", "significant_ci_and_fdr_05"}
    if coefs.empty or not required.issubset(coefs.columns):
        return pd.DataFrame()
    d = coefs[
        (coefs["inference"].isin(["gee", "bayes"]))
        & (coefs["effect_type"].isin(["E_effect", "case_control_E_effect_difference"]))
        & (coefs["significant_ci_and_fdr_05"].fillna(False))
    ].copy()
    if d.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["analysis_scope", "scope_label", "study", "group", "model", "inference", "effect_type", "estimate_direction", "phase_lag"]
    for col in group_cols:
        if col not in d.columns:
            d[col] = np.nan
    for keys, gd in d.groupby(group_cols, dropna=False, observed=False):
        gd = gd.sort_values("phase_bin")
        current = []
        prev_bin = None
        clusters = []
        for row in gd.itertuples():
            b = int(row.phase_bin)
            if prev_bin is None or b == prev_bin + 1:
                current.append(row)
            else:
                clusters.append(current)
                current = [row]
            prev_bin = b
        if current:
            clusters.append(current)
        for i, cluster in enumerate(clusters, start=1):
            est = np.asarray([float(r.estimate_log_odds) for r in cluster], dtype=float)
            abs_peak_idx = int(np.nanargmax(np.abs(est)))
            rows.append({
                **dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,))),
                "cluster_index": i,
                "n_bins": int(len(cluster)),
                "phase_lo_fraction": float(cluster[0].phase_lo_fraction),
                "phase_hi_fraction": float(cluster[-1].phase_hi_fraction),
                "mean_estimate_log_odds": float(np.nanmean(est)),
                "peak_estimate_log_odds": float(est[abs_peak_idx]),
                "peak_phase_mid_fraction": float(cluster[abs_peak_idx].phase_mid_fraction),
                "direction": "positive" if float(np.nanmean(est)) > 0 else "negative",
                "min_p_value": float(np.nanmin([float(r.p_value) for r in cluster])),
                "min_p_fdr_bh_within_cardiac_phase_curve": float(np.nanmin([float(r.p_fdr_bh_within_cardiac_phase_curve) for r in cluster])),
            })
    return pd.DataFrame(rows)


def analyse_cardiac_phase_beat_effects(beat_df: pd.DataFrame, cfg: Config, logger: logging.Logger) -> None:
    cols = cardiac_period_cols(CURRENT_PHASE_LAG)
    if beat_df.empty or not all(c in beat_df.columns for c in cols):
        logger.warning("No beat-level systole/diastole E columns available.")
        return
    phase_specs: List[Tuple[str, List[str]]] = [(CURRENT_PHASE_LAG, cols)]
    prev_cols = cardiac_period_cols(PREVIOUS_PHASE_LAG)
    if all(c in beat_df.columns for c in prev_cols):
        phase_specs.append((PREVIOUS_PHASE_LAG, prev_cols))
    else:
        logger.warning("Previous-beat systole/diastole E columns unavailable; previous-beat analysis skipped.")
    coef_rows: List[Dict[str, object]] = []
    omnibus_rows: List[Dict[str, object]] = []

    for phase_lag, lag_cols in phase_specs:
        run_cardiac_phase_beat_models(coef_rows, beat_df, cfg, logger, "global", "global population", "all", "all", "gee", lag_cols, phase_lag)
        run_cardiac_phase_omnibus_gee(omnibus_rows, beat_df, cfg, logger, "global", "global population", "all", "all", lag_cols, phase_lag)
        for group, gd in beat_df.groupby("group", observed=False):
            studies = sorted(str(x) for x in gd["study"].dropna().unique())
            study = studies[0] if len(studies) == 1 else "mixed"
            run_cardiac_phase_beat_models(coef_rows, gd, cfg, logger, "group", f"group {group}", study, str(group), "gee", lag_cols, phase_lag)
            run_cardiac_phase_omnibus_gee(omnibus_rows, gd, cfg, logger, "group", f"group {group}", study, str(group), lag_cols, phase_lag)

    for control_group, case_group, condition_effect in CASE_CONTROL_CONTRASTS:
        for phase_lag, lag_cols in phase_specs:
            run_cardiac_phase_case_control_gee(coef_rows, beat_df, cfg, logger, control_group, case_group, condition_effect, lag_cols, phase_lag)

    if not coef_rows:
        return
    append_cardiac_phase_empirical_bayes_rows(coef_rows, logger)
    coefs = pd.DataFrame(coef_rows)
    coefs = add_fdr(coefs, "p_value", "p_fdr_bh_across_cardiac_phase_E_effects")
    coefs = add_groupwise_fdr(
        coefs,
        "p_value",
        ["analysis_scope", "scope_label", "model", "effect_type", "estimate_direction", "phase_lag"],
        "p_fdr_bh_within_cardiac_phase_curve",
    )
    coefs = add_zero_deviation_flags(coefs, "p_fdr_bh_within_cardiac_phase_curve")
    write_csv(coefs, cfg.tables_dir / "cardiac_phase_beat_E_effect_coefficients.csv")

    omnibus = pd.DataFrame(omnibus_rows)
    if not omnibus.empty:
        omnibus = add_fdr(omnibus, "p_value", "p_fdr_bh_cardiac_phase_omnibus")
    write_csv(omnibus, cfg.tables_dir / "cardiac_phase_beat_E_effect_omnibus_gee_tests.csv")
    write_csv(cardiac_phase_significant_clusters(coefs), cfg.tables_dir / "cardiac_phase_beat_E_effect_significant_clusters.csv")

    contrasts = coefs[
        (coefs["analysis_scope"] == "case_control_contrast")
        & (coefs["effect_type"] == "case_control_E_effect_difference")
    ].copy()
    write_csv(contrasts, cfg.tables_dir / "cardiac_phase_beat_E_effect_case_control_contrasts.csv")
    plot_cardiac_phase_bayes_effects(coefs, cfg)
    plot_cardiac_phase_case_control_bayes_effects(contrasts, cfg)
    plot_cardiac_phase_global_gee_effect(coefs, cfg)
    plot_cardiac_phase_beat_gee_effects(coefs, cfg)
    plot_cardiac_phase_case_control_gee_effects(contrasts, cfg)


def shade_significant_phase_bins(ax, d: pd.DataFrame) -> None:
    sig = d[d["significant_deviation_from_zero"].fillna(False)]
    for row in sig.itertuples():
        ax.axvspan(float(row.phase_lo_fraction), float(row.phase_hi_fraction), color="#f59e0b", alpha=0.18, lw=0)


def plot_phase_effect_curve(ax, d: pd.DataFrame, title: str, ylabel: str) -> None:
    d = d.sort_values("phase_bin")
    x = d["phase_mid_fraction"].to_numpy(float)
    estimate_col = "estimate_log_odds" if "estimate_log_odds" in d.columns else "estimate"
    y = d[estimate_col].to_numpy(float)
    lo = d["interval_low"].to_numpy(float)
    hi = d["interval_high"].to_numpy(float)
    shade_significant_phase_bins(ax, d)
    ax.fill_between(x, lo, hi, alpha=0.18)
    ax.plot(x, y, marker="o", lw=1.5)
    sig = d[d["significant_deviation_from_zero"].fillna(False)]
    if not sig.empty:
        ax.scatter(sig["phase_mid_fraction"], sig[estimate_col], color="black", s=22, zorder=3)
    ax.axhline(0, lw=1, ls="--", color="black", alpha=0.65)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def phase_lag_order(d: pd.DataFrame) -> List[str]:
    if "phase_lag" not in d.columns:
        return [CURRENT_PHASE_LAG]
    present = [str(x) for x in d["phase_lag"].dropna().unique()]
    ordered = [x for x in (CURRENT_PHASE_LAG, PREVIOUS_PHASE_LAG) if x in present]
    ordered.extend(sorted(x for x in present if x not in set(ordered)))
    return ordered or [CURRENT_PHASE_LAG]


def add_ecg_reference_strip(ax) -> None:
    ecg = ax.inset_axes([0.08, -0.34, 0.84, 0.18], transform=ax.transAxes)
    x = np.linspace(-0.03, 2.03, 1200)
    y = np.zeros_like(x)
    for r in (0.0, 1.0, 2.0):
        y += 1.15 * np.exp(-0.5 * ((x - r) / 0.009) ** 2)
        y -= 0.28 * np.exp(-0.5 * ((x - (r - 0.018)) / 0.006) ** 2)
        y -= 0.35 * np.exp(-0.5 * ((x - (r + 0.024)) / 0.011) ** 2)
        if r + 0.43 <= 2.03:
            y += 0.34 * np.exp(-0.5 * ((x - (r + 0.43)) / 0.060) ** 2)
        if r + 0.86 <= 2.03:
            y += 0.15 * np.exp(-0.5 * ((x - (r + 0.86)) / 0.040) ** 2)
    for offset in (0.0, 1.0):
        ecg.axvspan(offset, offset + SYSTOLE_WINDOW_S[0], color="#e0f2fe", alpha=0.20, lw=0)
        ecg.axvspan(offset + SYSTOLE_WINDOW_S[0], offset + SYSTOLE_WINDOW_S[1], color="#f59e0b", alpha=0.22, lw=0)
        ecg.axvspan(offset + SYSTOLE_WINDOW_S[1], offset + 1.0, color="#e0f2fe", alpha=0.20, lw=0)
    ecg.plot(x, y, color="#111827", lw=1.1)
    ecg.axhline(0, color="#94a3b8", lw=0.6)
    ecg.axvline(1.0, color="#64748b", lw=0.8, ls=":")
    ecg.text(0.5, 1.08, "previous cardiac cycle", ha="center", va="bottom", fontsize=7, color="#334155")
    ecg.text(1.5, 1.08, "current cardiac cycle", ha="center", va="bottom", fontsize=7, color="#334155")
    ecg.text(SYSTOLE_WINDOW_S[0] + 0.01, 0.78, "systole", fontsize=7, color="#92400e", va="top")
    ecg.text(1.0 + SYSTOLE_WINDOW_S[0] + 0.01, 0.78, "systole", fontsize=7, color="#92400e", va="top")
    ecg.set_xlim(-0.03, 2.03)
    ecg.set_ylim(-0.55, 1.25)
    ecg.set_yticks([])
    ecg.set_xticks([0, SYSTOLE_WINDOW_S[0], SYSTOLE_WINDOW_S[1], 1.0, 1.0 + SYSTOLE_WINDOW_S[0], 1.0 + SYSTOLE_WINDOW_S[1], 2.0])
    ecg.set_xticklabels(["prev R", "0.35", "0.60", "R", "0.35", "0.60", "next R"], fontsize=7)
    ecg.set_xlabel("ECG reference", fontsize=7, labelpad=1)
    for spine in ("left", "right", "top"):
        ecg.spines[spine].set_visible(False)
    ecg.spines["bottom"].set_color("#cbd5e1")
    ecg.tick_params(axis="x", length=2, colors="#334155", pad=1)


def plot_phase_effect_lag_overlay(ax, d: pd.DataFrame, title: str, ylabel: str, show_ecg: bool = False) -> None:
    if d.empty:
        return
    d = d.copy()
    if "phase_lag" not in d.columns:
        d["phase_lag"] = CURRENT_PHASE_LAG
    estimate_col = "estimate_log_odds" if "estimate_log_odds" in d.columns else "estimate"
    colors = {CURRENT_PHASE_LAG: "#2563eb", PREVIOUS_PHASE_LAG: "#dc2626"}
    markers = {CURRENT_PHASE_LAG: "o", PREVIOUS_PHASE_LAG: "^"}
    lags = phase_lag_order(d)
    if "cardiac_period" in d.columns and d["cardiac_period"].notna().any():
        section_specs = [
            (PREVIOUS_PHASE_LAG, "systole", 0.475, "Systole"),
            (PREVIOUS_PHASE_LAG, "diastole", 0.800, "Diastole"),
            (CURRENT_PHASE_LAG, "systole", 1.475, "Systole"),
            (CURRENT_PHASE_LAG, "diastole", 1.800, "Diastole"),
        ]
        for offset in (0.0, 1.0):
            ax.axvspan(offset, offset + SYSTOLE_WINDOW_S[0], color="#e0f2fe", alpha=0.12, lw=0, zorder=0)
            ax.axvspan(offset + SYSTOLE_WINDOW_S[0], offset + SYSTOLE_WINDOW_S[1], color="#f59e0b", alpha=0.16, lw=0, zorder=0)
            ax.axvspan(offset + SYSTOLE_WINDOW_S[1], offset + 1.0, color="#e0f2fe", alpha=0.12, lw=0, zorder=0)
        ax.axvline(1.0, color="#64748b", lw=0.8, ls=":", zorder=1)
        rows = []
        for lag, period, x_pos, label in section_specs:
            sd = d[(d["phase_lag"].astype(str) == lag) & (d["cardiac_period"].astype(str) == period)].copy()
            if sd.empty:
                continue
            row = sd.iloc[0]
            rows.append((lag, period, x_pos, label, row))
        if rows:
            x = np.asarray([r[2] for r in rows], dtype=float)
            y = np.asarray([float(r[4][estimate_col]) for r in rows], dtype=float)
            ax.plot(x, y, color="#111827", lw=1.35, marker="o", ms=3.5, zorder=3)
            for lag, period, x_pos, _, row in rows:
                est = float(row[estimate_col])
                lo = float(row["interval_low"])
                hi = float(row["interval_high"])
                se = float(row["std_error"]) if np.isfinite(float(row.get("std_error", np.nan))) else (hi - lo) / 3.92
                box_lo = max(lo, est - se)
                box_hi = min(hi, est + se)
                if box_hi <= box_lo:
                    box_lo, box_hi = est - 0.03, est + 0.03
                face = "#fff7ed" if period == "systole" else "#eff6ff"
                edge = "#111827" if bool(row.get("significant_deviation_from_zero", False)) else "#475569"
                lw = 1.4 if bool(row.get("significant_deviation_from_zero", False)) else 0.9
                ax.vlines(x_pos, lo, hi, color="#475569", lw=1.0, zorder=4)
                ax.hlines([lo, hi], x_pos - 0.045, x_pos + 0.045, color="#475569", lw=1.0, zorder=4)
                ax.add_patch(plt.Rectangle((x_pos - 0.075, box_lo), 0.15, box_hi - box_lo, facecolor=face, edgecolor=edge, lw=lw, zorder=5))
                ax.hlines(est, x_pos - 0.075, x_pos + 0.075, color="#111827", lw=1.5, zorder=6)
        ax.set_xticks([r[2] for r in section_specs])
        ax.set_xticklabels([r[3] for r in section_specs])
        ax.set_xlim(-0.05, 2.05)
    else:
        for lag in lags:
            sd = d[d["phase_lag"].astype(str) == lag].sort_values("phase_bin")
            if sd.empty:
                continue
            x = sd["phase_mid_fraction"].to_numpy(float)
            y = sd[estimate_col].to_numpy(float)
            lo = sd["interval_low"].to_numpy(float)
            hi = sd["interval_high"].to_numpy(float)
            color = colors.get(lag, "#334155")
            ax.fill_between(x, lo, hi, color=color, alpha=0.13)
            ax.plot(x, y, marker=markers.get(lag, "o"), lw=1.5, color=color, label=phase_lag_label(lag))
            sig = sd[sd["significant_deviation_from_zero"].fillna(False)]
            if not sig.empty:
                ax.scatter(
                    sig["phase_mid_fraction"],
                    sig[estimate_col],
                    marker="s",
                    s=34,
                    color=color,
                    edgecolor="black",
                    linewidth=0.55,
                    zorder=4,
                )
        ax.set_xlim(0, 1)
    ax.axhline(0, lw=1, ls="--", color="black", alpha=0.65)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, legend_labels, loc="best", frameon=False, fontsize=8)
    ax.text(
        0.995,
        0.02,
        "black marker/box: interval excludes zero",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#334155",
    )
    if show_ecg and "cardiac_period" in d.columns and d["cardiac_period"].notna().any():
        add_ecg_reference_strip(ax)


def plot_latency_consistency_phase_effects(coefs: pd.DataFrame, cfg: Config) -> None:
    d = coefs[
        (coefs["inference"] == "gee_gaussian")
        & (coefs["analysis_scope"].isin(["global", "group"]))
        & (coefs["effect_type"] == "E_effect")
    ].copy()
    if d.empty:
        return
    labels = ["global population"] + [f"group {g}" for g in ("ANX", "NANX", "HTN", "NHTN")]
    labels = [label for label in labels if label in set(d["scope_label"])]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, max(3.2, 2.4 * len(labels))), sharex=True, squeeze=False)
    for ax, label in zip(axes.ravel(), labels):
        plot_phase_effect_curve(
            ax,
            d[d["scope_label"] == label],
            label,
            "Effect on log inverse latency variance",
        )
    axes.ravel()[-1].set_xlabel("Normalised R-to-next-R phase")
    fig.suptitle(f"Microstate {cfg.primary_state} and tap-latency consistency across the cardiac cycle")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(cfg.figures_dir / "cardiac_phase_latency_consistency_E_effect_global_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cardiac_phase_beat_gee_effects(coefs: pd.DataFrame, cfg: Config) -> None:
    d = coefs[
        (coefs["inference"] == "gee")
        & (coefs["analysis_scope"].isin(["global", "group"]))
        & (coefs["effect_type"] == "E_effect")
    ].copy()
    if d.empty:
        return
    labels = ["global population"] + [f"group {g}" for g in ("ANX", "NANX", "HTN", "NHTN")]
    labels = [label for label in labels if label in set(d["scope_label"])]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, max(3.2, 2.4 * len(labels))), sharex=True, squeeze=False)
    for i, (ax, label) in enumerate(zip(axes.ravel(), labels)):
        plot_phase_effect_lag_overlay(
            ax,
            d[d["scope_label"] == label],
            label,
            "GEE log-odds",
            show_ecg=(i == len(labels) - 1),
        )
    axes.ravel()[-1].set_xlabel("Cardiac period")
    fig.suptitle(f"Beat-level effect of microstate {cfg.primary_state}: systole vs diastole")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(cfg.figures_dir / "cardiac_phase_beat_E_effect_gee_global_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cardiac_phase_global_gee_effect(coefs: pd.DataFrame, cfg: Config) -> None:
    d = coefs[
        (coefs["inference"] == "gee")
        & (coefs["analysis_scope"] == "global")
        & (coefs["effect_type"] == "E_effect")
    ].copy()
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    plot_phase_effect_lag_overlay(
        ax,
        d,
        f"Global beat-level effect of microstate {cfg.primary_state}: systole vs diastole",
        "GEE log-odds",
        show_ecg=True,
    )
    ax.set_xlabel("Cardiac period")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(cfg.figures_dir / "cardiac_phase_beat_E_effect_gee_global_only.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_cardiac_phase_bayes_effects(coefs: pd.DataFrame, cfg: Config) -> None:
    d = coefs[
        (coefs["inference"] == "bayes")
        & (coefs["analysis_scope"].isin(["global", "group"]))
        & (coefs["effect_type"] == "E_effect")
    ].copy()
    if d.empty:
        return
    labels = ["global population"] + [f"group {g}" for g in ("ANX", "NANX", "HTN", "NHTN")]
    labels = [label for label in labels if label in set(d["scope_label"])]
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, max(3.2, 2.4 * len(labels))), sharex=True, squeeze=False)
    for i, (ax, label) in enumerate(zip(axes.ravel(), labels)):
        plot_phase_effect_lag_overlay(
            ax,
            d[d["scope_label"] == label],
            label,
            "Posterior log-odds",
            show_ecg=(i == len(labels) - 1),
        )
    axes.ravel()[-1].set_xlabel("Cardiac period")
    fig.suptitle(f"Empirical-Bayes beat-level effect of microstate {cfg.primary_state}: systole vs diastole")
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    fig.savefig(cfg.figures_dir / "cardiac_phase_beat_E_effect_bayes_global_group.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    gd = d[d["analysis_scope"] == "global"].copy()
    if not gd.empty:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        plot_phase_effect_lag_overlay(
            ax,
            gd,
            f"Global empirical-Bayes beat-level effect of microstate {cfg.primary_state}: systole vs diastole",
            "Posterior log-odds",
            show_ecg=True,
        )
        ax.set_xlabel("Cardiac period")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(cfg.figures_dir / "cardiac_phase_beat_E_effect_bayes_global_only.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_cardiac_phase_case_control_bayes_effects(contrasts: pd.DataFrame, cfg: Config) -> None:
    d = contrasts[contrasts["inference"] == "bayes"].copy()
    if d.empty:
        return
    for label, sd in d.groupby("scope_label", observed=False):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        direction = str(sd["estimate_direction"].dropna().iloc[0]) if sd["estimate_direction"].notna().any() else str(label).replace(" ", "_")
        plot_phase_effect_lag_overlay(
            ax,
            sd,
            f"{label}: empirical-Bayes difference in microstate {cfg.primary_state} effect by cardiac period",
            "Posterior log-odds difference",
            show_ecg=True,
        )
        ax.set_xlabel("Cardiac period")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        safe = re.sub(r"[^A-Za-z0-9]+", "_", direction).strip("_")
        fig.savefig(cfg.figures_dir / f"cardiac_phase_beat_E_effect_{safe}_bayes.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_cardiac_phase_case_control_gee_effects(contrasts: pd.DataFrame, cfg: Config) -> None:
    d = contrasts[contrasts["inference"] == "gee"].copy()
    if d.empty:
        return
    for label, sd in d.groupby("scope_label", observed=False):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        direction = str(sd["estimate_direction"].dropna().iloc[0]) if sd["estimate_direction"].notna().any() else str(label).replace(" ", "_")
        plot_phase_effect_lag_overlay(
            ax,
            sd,
            f"{label}: difference in microstate {cfg.primary_state} effect by cardiac period",
            "GEE log-odds difference",
            show_ecg=True,
        )
        ax.set_xlabel("Cardiac period")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        safe = re.sub(r"[^A-Za-z0-9]+", "_", direction).strip("_")
        fig.savefig(cfg.figures_dir / f"cardiac_phase_beat_E_effect_{safe}_gee.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


# =============================================================================
# Main orchestration
# =============================================================================


def write_run_metadata(cfg: Config) -> None:
    payload = asdict(cfg)
    payload["script_version"] = SCRIPT_VERSION
    payload = {k: str(v) if isinstance(v, Path) else v for k, v in payload.items()}
    (cfg.output_dir / "run_metadata.json").write_text(json.dumps(payload, indent=2))


def run_pipeline(cfg: Config) -> None:
    ensure_dirs(cfg)
    logger = setup_logger(cfg.output_dir)
    write_run_metadata(cfg)
    logger.info("Starting HEPPy microstate publication statistics.")
    logger.info(json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2))

    manifest = discover_manifest(cfg, logger)
    sequences = load_all_sequences(cfg, manifest, logger)

    state_metrics = compute_state_metrics(sequences)
    analyse_group_condition_state_effects(state_metrics, cfg, logger)

    circ_df, bin_df = circular_block_features(sequences, cfg, logger)
    analyse_circular_modulation(circ_df, bin_df, cfg, logger)

    beat_df = build_beat_level_table(sequences, cfg, logger)
    hit_summary = summarise_hit_miss(beat_df, cfg)
    analyse_tapping_accuracy_linkage(hit_summary, state_metrics, cfg, logger)
    analyse_beat_by_beat_E_windows(beat_df, cfg, logger)
    analyse_cardiac_phase_beat_effects(beat_df, cfg, logger)
    analyse_latency_consistency_cardiac_phase(beat_df, cfg, logger)

    logger.info("Finished HEPPy microstate publication statistics.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream HEPPy cardiac-cycle microstate statistics.")
    p.add_argument("--matlab-root", type=Path, default='/Users/rohan/EEG/Microstates_and_Interoception/output/heppy_microstates_matlab', help="Output directory from fit_microstates_heppy_global_group_condition.m")
    p.add_argument("--heppy-root", type=Path, default='/Users/rohan/EEG/Microstates_and_Interoception/heppy', help="HEPPy directory containing raw_fif")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--conditions", nargs="+", default=list(INFERENTIAL_CONDITIONS))
    p.add_argument("--accuracy-conditions", nargs="+", default=list(ACCURACY_CONDITIONS))
    p.add_argument("--primary-state", default="E")
    p.add_argument("--n-phase-bins", type=int, default=24)
    p.add_argument("--valid-tap-window-s", nargs=2, type=float, default=list(DEFAULT_TAP_HIT_WINDOW_S), metavar=("LO", "HI"))
    p.add_argument("--min-participants", type=int, default=6)
    p.add_argument("--min-beats-per-cell", type=int, default=8)
    p.add_argument("--max-beats-per-participant-condition", type=int, default=2000)
    p.add_argument("--n-circular-permutations", type=int, default=0)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if float(args.valid_tap_window_s[0]) >= float(args.valid_tap_window_s[1]):
        raise ValueError("--valid-tap-window-s requires LO < HI")
    out_dir = args.output_dir or (args.matlab_root / "python_publication_stats")
    cfg = Config(
        matlab_root=args.matlab_root,
        heppy_root=args.heppy_root,
        output_dir=out_dir,
        conditions=tuple(args.conditions),
        accuracy_conditions=tuple(args.accuracy_conditions),
        primary_state=flatten_state_name(args.primary_state),
        n_phase_bins=int(args.n_phase_bins),
        valid_tap_window_s=(float(args.valid_tap_window_s[0]), float(args.valid_tap_window_s[1])),
        min_participants=int(args.min_participants),
        min_beats_per_cell=int(args.min_beats_per_cell),
        max_beats_per_participant_condition=int(args.max_beats_per_participant_condition),
        n_circular_permutations=int(args.n_circular_permutations),
        random_seed=int(args.random_seed),
        quick=bool(args.quick),
        force_rebuild=bool(args.force_rebuild),
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
