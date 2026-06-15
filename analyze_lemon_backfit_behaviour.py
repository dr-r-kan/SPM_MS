#!/usr/bin/env python3
"""Relate hierarchical backfit outputs to LEMON psychometric PCs and demographics.

This script is intentionally implemented without pandas because the current
environment has binary-compatibility issues in the pandas/pyarrow stack.

It expects the hierarchical pipeline CSV exports written by
`metamicrostate_dataset_pipeline.m`:

    - participant_condition_state_backfit_metrics.csv
    - participant_condition_record_backfit_summary.csv

It then:

1. Builds participant-level backfit feature matrices for one or more backfit
   methods (`gaussian_mixture`, `hard`).
2. Auto-discovers psychometric CSV files under the LEMON behavioural folder,
   keeps only numeric variables from files that overlap the analysed EEG cohort,
   and performs PCA on the merged psychometric matrix.
3. Relates the retained psychometric PCs to each backfit feature using Pearson
   correlation.
4. Performs simple demographic effect tests when a demographic table overlaps
   the analysed cohort:
      - numeric/ordinal variables: Spearman correlation
      - binary variables: Welch t-test + Cohen's d
      - categorical variables with >2 levels: Kruskal-Wallis + epsilon-squared

Outputs are written as CSV/TXT files under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


DEFAULT_STATE_METRICS = Path(
    "outputs/hierarchical_microstates/participant_condition_state_backfit_metrics.csv"
)
DEFAULT_RECORD_SUMMARY = Path(
    "outputs/hierarchical_microstates/participant_condition_record_backfit_summary.csv"
)
DEFAULT_BEHAVIOUR_DIR = Path(
    "/home/rohan/EEG_Data/LEMON/Behavioural_Data_MPILMBB_LEMON"
)
DEFAULT_NAME_MATCH = Path("/home/rohan/EEG_Data/LEMON/name_match.csv")
DEFAULT_DEMOGRAPHICS = DEFAULT_BEHAVIOUR_DIR / (
    "META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs/hierarchical_microstates/behavioural_backfit_analysis")

PID_RE = re.compile(r"^sub-\d+$", re.IGNORECASE)
AGE_BIN_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

STATE_FEATURE_COLUMNS = (
    "occupancy",
    "percentage_record_present",
    "mean_gfp",
    "occurrence_rate_hz",
    "template_match_abs_correlation",
)
RECORD_FEATURE_COLUMNS = (
    "record_differential_entropy_bits",
    "record_shannon_entropy_bits",
)
DEFAULT_PSYCHOMETRIC_DIRS = (
    "Emotion_and_Personality_Test_Battery_LEMON",
    "Cognitive_Test_Battery_LEMON",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate hierarchical microstate backfit outputs to LEMON psychometric "
            "principal components and demographic variables."
        )
    )
    parser.add_argument(
        "--state-metrics",
        type=Path,
        default=DEFAULT_STATE_METRICS,
        help="CSV exported by metamicrostate_dataset_pipeline.m with state metrics.",
    )
    parser.add_argument(
        "--record-summary",
        type=Path,
        default=DEFAULT_RECORD_SUMMARY,
        help="CSV exported by metamicrostate_dataset_pipeline.m with record summaries.",
    )
    parser.add_argument(
        "--behaviour-dir",
        type=Path,
        default=DEFAULT_BEHAVIOUR_DIR,
        help="LEMON behavioural data directory.",
    )
    parser.add_argument(
        "--name-match-csv",
        type=Path,
        default=DEFAULT_NAME_MATCH,
        help="CSV mapping LEMON behavioural IDs onto EEG/backfit participant IDs.",
    )
    parser.add_argument(
        "--demographics-csv",
        type=Path,
        default=DEFAULT_DEMOGRAPHICS,
        help="Demographic CSV used for simple effect analyses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output CSV/TXT files.",
    )
    parser.add_argument(
        "--backfit-methods",
        nargs="*",
        default=[],
        help=(
            "Backfit methods to analyse. If omitted, all methods present in the "
            "state-metrics CSV are used."
        ),
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["eyes_closed", "eyes_open"],
        help="Conditions to include when building participant-level backfit features.",
    )
    parser.add_argument(
        "--disable-condition-deltas",
        action="store_true",
        help="Do not create eyes-open minus eyes-closed delta backfit features.",
    )
    parser.add_argument(
        "--min-psychometric-overlap",
        type=int,
        default=25,
        help="Minimum participant overlap required for a psychometric CSV to be used.",
    )
    parser.add_argument(
        "--min-feature-coverage",
        type=float,
        default=0.25,
        help="Minimum non-missing fraction required for a psychometric feature.",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=10,
        help="Upper bound on retained psychometric principal components.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.8,
        help="Retain the smallest number of PCs reaching this cumulative variance.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows[0], rows[1:]


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_dict_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_output_value(row.get(key)) for key in fieldnames})


def write_matrix_csv(
    path: Path,
    row_ids: list[str],
    col_names: list[str],
    matrix: np.ndarray,
    row_label: str = "participant",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([row_label, *col_names])
        for idx, row_id in enumerate(row_ids):
            writer.writerow([row_id, *[format_output_value(v) for v in matrix[idx, :]]])


def format_output_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.10g}"
    return value


def sanitize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def parse_float(value: str | None) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().strip('"')
    if not text or text.lower() in {"nan", "na", "n/a", "none", "null"}:
        return np.nan
    text = text.replace(",", "")
    if text.startswith((">", "<")):
        text = text[1:].strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_boolish(value: str | None) -> bool:
    if value is None:
        return False
    text = str(value).strip().strip('"').lower()
    return text in {"1", "true", "t", "yes", "y"}


def clean_participant_id(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"')


def load_name_match_map(name_match_csv: Path) -> dict[str, str]:
    if not name_match_csv.is_file():
        return {}

    rows = read_csv_dicts(name_match_csv)
    if not rows:
        return {}

    field_lookup = {
        sanitize_name(field): field
        for field in rows[0].keys()
        if field is not None and str(field).strip()
    }
    initial_key = field_lookup.get("initial_id")
    indi_key = field_lookup.get("indi_id")
    if initial_key is None or indi_key is None:
        raise ValueError(
            f"Expected Initial_ID and INDI_ID columns in name-match CSV: {name_match_csv}"
        )

    alias_to_canonical: dict[str, str] = {}
    for row in rows:
        initial_id = clean_participant_id(row.get(initial_key))
        indi_id = clean_participant_id(row.get(indi_key))
        canonical_id = initial_id or indi_id
        if not canonical_id:
            continue
        for alias in (initial_id, indi_id):
            if not alias:
                continue
            existing = alias_to_canonical.get(alias)
            if existing is not None and existing != canonical_id:
                raise ValueError(
                    f"Conflicting participant mapping for {alias}: {existing} vs {canonical_id}"
                )
            alias_to_canonical[alias] = canonical_id
    return alias_to_canonical


def resolve_participant_id(value: str | None, alias_to_canonical: dict[str, str]) -> str:
    participant = clean_participant_id(value)
    if not participant:
        return ""
    return alias_to_canonical.get(participant, participant)


def detect_participant_column(header: list[str], rows: list[list[str]]) -> int | None:
    best_idx = None
    best_score = -1
    max_cols = max((len(row) for row in rows), default=len(header))
    for col_idx in range(max_cols):
        sample = []
        for row in rows[:100]:
            if col_idx < len(row):
                value = row[col_idx].strip().strip('"')
                if value:
                    sample.append(value)
        if not sample:
            continue
        score = sum(bool(PID_RE.match(value)) for value in sample)
        if score > best_score:
            best_score = score
            best_idx = col_idx
    return best_idx if best_score > 0 else None


def detect_numeric_column(values: list[str]) -> tuple[bool, int, int]:
    nonempty = 0
    numeric = 0
    for value in values:
        text = value.strip().strip('"')
        if not text:
            continue
        nonempty += 1
        if math.isfinite(parse_float(text)):
            numeric += 1
    return (nonempty > 0 and numeric / nonempty >= 0.8), nonempty, numeric


def age_bin_midpoint(value: str | None) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().strip('"')
    if not text:
        return np.nan
    match = AGE_BIN_RE.match(text)
    if match:
        low = float(match.group(1))
        high = float(match.group(2))
        return 0.5 * (low + high)
    return parse_float(text)


def fdr_bh(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    finite_mask = np.isfinite(p)
    if not np.any(finite_mask):
        return q.tolist()
    p_finite = p[finite_mask]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    n = ranked.size
    adjusted = np.empty(n, dtype=float)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        candidate = ranked[i] * n / rank
        cumulative = min(cumulative, candidate)
        adjusted[i] = cumulative
    restored = np.empty(n, dtype=float)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite_mask] = restored
    return q.tolist()


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def build_backfit_feature_matrix(
    state_rows: list[dict[str, str]],
    record_rows: list[dict[str, str]],
    method: str,
    conditions: set[str],
    include_deltas: bool,
    participant_aliases: dict[str, str],
) -> tuple[list[str], list[str], np.ndarray, dict[str, object]]:
    feature_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    feature_names: set[str] = set()
    participants: set[str] = set()
    state_counts = 0
    record_counts = 0

    for row in state_rows:
        if row.get("backfit_method", "") != method:
            continue
        if not parse_boolish(row.get("backfit_available")):
            continue
        participant = resolve_participant_id(row.get("participant"), participant_aliases)
        condition = row.get("condition", "").strip()
        label = row.get("template_label", "").strip()
        if not participant or not condition or not label or condition not in conditions:
            continue
        participants.add(participant)
        for metric in STATE_FEATURE_COLUMNS:
            value = parse_float(row.get(metric))
            if math.isfinite(value):
                name = f"state__{condition}__{label}__{metric}"
                feature_values[participant][name].append(value)
                feature_names.add(name)
        state_counts += 1

    for row in record_rows:
        if row.get("backfit_method", "") != method:
            continue
        if not parse_boolish(row.get("backfit_available")):
            continue
        participant = resolve_participant_id(row.get("participant"), participant_aliases)
        condition = row.get("condition", "").strip()
        if not participant or condition not in conditions:
            continue
        participants.add(participant)
        for metric in RECORD_FEATURE_COLUMNS:
            value = parse_float(row.get(metric))
            if math.isfinite(value):
                name = f"record__{condition}__{metric}"
                feature_values[participant][name].append(value)
                feature_names.add(name)
        record_counts += 1

    if include_deltas and {"eyes_closed", "eyes_open"}.issubset(conditions):
        participant_ids = list(participants)
        delta_targets = list(feature_names)
        for participant in participant_ids:
            row_dict = feature_values[participant]
            for feature_name in delta_targets:
                if "__eyes_closed__" in feature_name:
                    open_name = feature_name.replace("__eyes_closed__", "__eyes_open__")
                    if open_name in row_dict and feature_name in row_dict:
                        closed_vals = row_dict[feature_name]
                        open_vals = row_dict[open_name]
                        if closed_vals and open_vals:
                            delta_name = (
                                feature_name.replace(
                                    "state__eyes_closed__", "delta__eyes_open_minus_closed__"
                                ).replace(
                                    "record__eyes_closed__", "delta_record__eyes_open_minus_closed__"
                                )
                            )
                            delta_value = float(np.mean(open_vals) - np.mean(closed_vals))
                            row_dict[delta_name].append(delta_value)
                            feature_names.add(delta_name)

    participant_list = sorted(participants)
    feature_list = sorted(feature_names)
    matrix = np.full((len(participant_list), len(feature_list)), np.nan, dtype=float)
    for i, participant in enumerate(participant_list):
        row_dict = feature_values[participant]
        for j, feature_name in enumerate(feature_list):
            values = row_dict.get(feature_name)
            if values:
                matrix[i, j] = float(np.mean(values))

    summary = {
        "method": method,
        "participants": len(participant_list),
        "features": len(feature_list),
        "state_rows_used": state_counts,
        "record_rows_used": record_counts,
    }
    return participant_list, feature_list, matrix, summary


def discover_psychometric_csvs(behaviour_dir: Path) -> list[Path]:
    csvs: list[Path] = []
    for rel_dir in DEFAULT_PSYCHOMETRIC_DIRS:
        root = behaviour_dir / rel_dir
        if root.is_dir():
            csvs.extend(sorted(root.rglob("*.csv")))
    return csvs


def load_psychometric_matrix(
    behaviour_dir: Path,
    participant_pool: set[str],
    min_overlap: int,
    min_feature_coverage: float,
    participant_aliases: dict[str, str],
) -> tuple[list[str], list[str], np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    per_participant: dict[str, dict[str, float]] = defaultdict(dict)
    inventory_rows: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []

    for csv_path in discover_psychometric_csvs(behaviour_dir):
        header, rows = read_csv_rows(csv_path)
        pid_col = detect_participant_column(header, rows)
        if pid_col is None:
            continue

        participant_ids = set()
        for row in rows:
            if pid_col < len(row):
                participant = resolve_participant_id(row[pid_col], participant_aliases)
                if PID_RE.match(participant):
                    participant_ids.add(participant)
        overlap_ids = participant_ids & participant_pool
        if len(overlap_ids) < min_overlap:
            file_rows.append(
                {
                    "file": str(csv_path.relative_to(behaviour_dir)),
                    "participant_column_index": pid_col,
                    "n_participants_in_file": len(participant_ids),
                    "n_overlap_with_backfit": len(overlap_ids),
                    "n_numeric_features_used": 0,
                    "status": "skipped_low_overlap",
                }
            )
            continue

        prefix = sanitize_name(str(csv_path.relative_to(behaviour_dir).with_suffix("")))
        max_cols = max((len(row) for row in rows), default=len(header))
        used_columns = 0
        for col_idx in range(max_cols):
            if col_idx == pid_col:
                continue
            col_values = []
            for row in rows:
                col_values.append(row[col_idx] if col_idx < len(row) else "")
            is_numeric, nonempty_count, numeric_count = detect_numeric_column(col_values)
            col_name = header[col_idx] if col_idx < len(header) else f"col_{col_idx}"
            feature_name = f"{prefix}__{sanitize_name(col_name or f'col_{col_idx}')}"
            if not is_numeric:
                inventory_rows.append(
                    {
                        "file": str(csv_path.relative_to(behaviour_dir)),
                        "column_index": col_idx,
                        "source_column": col_name,
                        "feature_name": feature_name,
                        "n_overlap_with_backfit": len(overlap_ids),
                        "n_nonempty_values": nonempty_count,
                        "n_numeric_values": numeric_count,
                        "status": "skipped_non_numeric",
                    }
                )
                continue

            overlap_nonmissing = 0
            for row in rows:
                if pid_col >= len(row):
                    continue
                participant = resolve_participant_id(row[pid_col], participant_aliases)
                if participant not in overlap_ids:
                    continue
                value = parse_float(row[col_idx] if col_idx < len(row) else "")
                if math.isfinite(value):
                    per_participant[participant][feature_name] = value
                    overlap_nonmissing += 1

            inventory_rows.append(
                {
                    "file": str(csv_path.relative_to(behaviour_dir)),
                    "column_index": col_idx,
                    "source_column": col_name,
                    "feature_name": feature_name,
                    "n_overlap_with_backfit": len(overlap_ids),
                    "n_nonempty_values": nonempty_count,
                    "n_numeric_values": numeric_count,
                    "n_overlap_nonmissing": overlap_nonmissing,
                    "status": "used" if overlap_nonmissing > 0 else "skipped_all_missing_after_overlap",
                }
            )
            if overlap_nonmissing > 0:
                used_columns += 1

        file_rows.append(
            {
                "file": str(csv_path.relative_to(behaviour_dir)),
                "participant_column_index": pid_col,
                "n_participants_in_file": len(participant_ids),
                "n_overlap_with_backfit": len(overlap_ids),
                "n_numeric_features_used": used_columns,
                "status": "used" if used_columns > 0 else "skipped_no_numeric_features",
            }
        )

    participant_list = sorted(per_participant)
    feature_list = sorted(
        unique_preserve_order(
            feature for participant in participant_list for feature in per_participant[participant]
        )
    )
    if not participant_list or not feature_list:
        return [], [], np.empty((0, 0)), inventory_rows, file_rows

    matrix = np.full((len(participant_list), len(feature_list)), np.nan, dtype=float)
    for i, participant in enumerate(participant_list):
        row_dict = per_participant[participant]
        for j, feature_name in enumerate(feature_list):
            if feature_name in row_dict:
                matrix[i, j] = row_dict[feature_name]

    coverage = np.mean(np.isfinite(matrix), axis=0)
    keep = coverage >= float(min_feature_coverage)
    kept_features = [name for name, flag in zip(feature_list, keep) if flag]
    kept_matrix = matrix[:, keep]

    final_keep = np.zeros(len(kept_features), dtype=bool)
    for j in range(len(kept_features)):
        col = kept_matrix[:, j]
        finite = col[np.isfinite(col)]
        if finite.size >= 2 and np.nanstd(finite) > 0:
            final_keep[j] = True

    kept_features = [name for name, flag in zip(kept_features, final_keep) if flag]
    kept_matrix = kept_matrix[:, final_keep]
    return participant_list, kept_features, kept_matrix, inventory_rows, file_rows


def run_psychometric_pca(
    participants: list[str],
    feature_names: list[str],
    matrix: np.ndarray,
    max_components: int,
    variance_threshold: float,
) -> dict[str, object]:
    if matrix.size == 0 or not participants or not feature_names:
        return {
            "participants": [],
            "feature_names": [],
            "raw_matrix": np.empty((0, 0)),
            "imputed_matrix": np.empty((0, 0)),
            "z_matrix": np.empty((0, 0)),
            "pc_scores": np.empty((0, 0)),
            "pc_names": [],
            "retained_components": 0,
            "explained_variance_ratio": np.empty(0),
            "cumulative_variance_ratio": np.empty(0),
            "component_weights": np.empty((0, 0)),
            "loadings": np.empty((0, 0)),
        }

    x_raw = np.asarray(matrix, dtype=float)
    x_imputed = x_raw.copy()
    medians = np.nanmedian(x_imputed, axis=0)
    for j in range(x_imputed.shape[1]):
        col = x_imputed[:, j]
        missing = ~np.isfinite(col)
        if np.any(missing):
            col[missing] = medians[j]
            x_imputed[:, j] = col

    means = np.mean(x_imputed, axis=0)
    stds = np.std(x_imputed, axis=0, ddof=0)
    stds[stds == 0] = 1.0
    x_z = (x_imputed - means) / stds

    n_components = int(min(max_components, x_z.shape[0], x_z.shape[1]))
    if n_components < 1:
        return {
            "participants": participants,
            "feature_names": feature_names,
            "raw_matrix": x_raw,
            "imputed_matrix": x_imputed,
            "z_matrix": x_z,
            "pc_scores": np.empty((len(participants), 0)),
            "pc_names": [],
            "retained_components": 0,
            "explained_variance_ratio": np.empty(0),
            "cumulative_variance_ratio": np.empty(0),
            "component_weights": np.empty((len(feature_names), 0)),
            "loadings": np.empty((len(feature_names), 0)),
        }

    u, singular_values, vt = np.linalg.svd(x_z, full_matrices=False)
    component_weights_full = vt[:n_components, :].T
    scores_full = x_z @ component_weights_full
    if x_z.shape[0] > 1:
        explained_variance_full = (singular_values[:n_components] ** 2) / (x_z.shape[0] - 1)
    else:
        explained_variance_full = singular_values[:n_components] ** 2
    total_variance = float(np.sum(explained_variance_full))
    if total_variance > 0:
        explained = explained_variance_full / total_variance
    else:
        explained = np.zeros_like(explained_variance_full)
    cumulative = np.cumsum(explained)
    retained = int(np.searchsorted(cumulative, variance_threshold, side="left") + 1)
    retained = max(1, min(retained, n_components))

    eigenvalues = explained_variance_full[:retained]
    component_weights = component_weights_full[:, :retained]
    loadings = component_weights * np.sqrt(eigenvalues)
    pc_names = [f"PC{i + 1}" for i in range(retained)]
    return {
        "participants": participants,
        "feature_names": feature_names,
        "raw_matrix": x_raw,
        "imputed_matrix": x_imputed,
        "z_matrix": x_z,
        "pc_scores": scores_full[:, :retained],
        "pc_names": pc_names,
        "retained_components": retained,
        "explained_variance_ratio": explained[:retained],
        "cumulative_variance_ratio": cumulative[:retained],
        "component_weights": component_weights,
        "loadings": loadings,
    }


def intersect_rows(
    left_ids: list[str], left_matrix: np.ndarray, right_ids: list[str], right_matrix: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray]:
    right_index = {participant: idx for idx, participant in enumerate(right_ids)}
    common = [participant for participant in left_ids if participant in right_index]
    if not common:
        return [], np.empty((0, left_matrix.shape[1])), np.empty((0, right_matrix.shape[1]))
    left_lookup = {participant: idx for idx, participant in enumerate(left_ids)}
    left_out = np.vstack([left_matrix[left_lookup[participant], :] for participant in common])
    right_out = np.vstack([right_matrix[right_index[participant], :] for participant in common])
    return common, left_out, right_out


def correlate_backfit_with_pcs(
    participants: list[str],
    backfit_feature_names: list[str],
    backfit_matrix: np.ndarray,
    pc_names: list[str],
    pc_scores: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_idx, feature_name in enumerate(backfit_feature_names):
        x = backfit_matrix[:, feature_idx]
        if np.sum(np.isfinite(x)) < 10 or np.nanstd(x) == 0:
            continue
        for pc_idx, pc_name in enumerate(pc_names):
            y = pc_scores[:, pc_idx]
            mask = np.isfinite(x) & np.isfinite(y)
            if np.sum(mask) < 10 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
                continue
            r, p_value = stats.pearsonr(x[mask], y[mask])
            rows.append(
                {
                    "feature_name": feature_name,
                    "pc_name": pc_name,
                    "n": int(np.sum(mask)),
                    "pearson_r": float(r),
                    "p_value": float(p_value),
                }
            )
    q_values = fdr_bh([row["p_value"] for row in rows])
    for row, q_value in zip(rows, q_values):
        row["fdr_q_value"] = q_value
    rows.sort(key=lambda row: (row["fdr_q_value"], row["p_value"], row["feature_name"], row["pc_name"]))
    return rows


def load_demographics(
    demographics_csv: Path,
    participant_pool: set[str],
    participant_aliases: dict[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not demographics_csv.is_file():
        return [], []
    header, rows = read_csv_rows(demographics_csv)
    pid_col = detect_participant_column(header, rows)
    if pid_col is None:
        return [], []

    header_map = {sanitize_name(col): idx for idx, col in enumerate(header)}

    def get_value(row: list[str], key_fragment: str) -> str:
        for key, idx in header_map.items():
            if key_fragment in key and idx < len(row):
                return row[idx].strip().strip('"')
        return ""

    demographic_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for row in rows:
        if pid_col >= len(row):
            continue
        participant = resolve_participant_id(row[pid_col], participant_aliases)
        if participant not in participant_pool:
            continue
        demographic_rows.append(
            {
                "participant": participant,
                "gender_code": parse_float(get_value(row, "gender")),
                "age_bin": get_value(row, "age"),
                "age_midpoint": age_bin_midpoint(get_value(row, "age")),
                "handedness": get_value(row, "handedness"),
                "education": get_value(row, "education"),
                "drug_positive": parse_float(get_value(row, "drug_0_negative_1_positive")),
                "smoking": get_value(row, "smoking"),
                "smoking_num": parse_float(get_value(row, "smoking_num")),
                "relationship_status": get_value(row, "relationship_status"),
            }
        )

    counts = Counter()
    for row in demographic_rows:
        for key, value in row.items():
            if key == "participant":
                continue
            if isinstance(value, str):
                if value:
                    counts[key] += 1
            elif isinstance(value, float):
                if math.isfinite(value):
                    counts[key] += 1
    for key, count in sorted(counts.items()):
        inventory_rows.append({"variable": key, "n_nonmissing_overlap": count})
    return demographic_rows, inventory_rows


def build_demographic_maps(
    demographic_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    demo_map: dict[str, dict[str, object]] = {}
    for row in demographic_rows:
        participant = str(row["participant"])
        demo_map[participant] = {key: value for key, value in row.items() if key != "participant"}
    return demo_map


def cohen_d(x0: np.ndarray, x1: np.ndarray) -> float:
    n0 = x0.size
    n1 = x1.size
    if n0 < 2 or n1 < 2:
        return np.nan
    v0 = np.var(x0, ddof=1)
    v1 = np.var(x1, ddof=1)
    pooled = ((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1)
    if pooled <= 0:
        return np.nan
    return float((np.mean(x1) - np.mean(x0)) / math.sqrt(pooled))


def analyze_demographic_effects(
    participants: list[str],
    backfit_feature_names: list[str],
    backfit_matrix: np.ndarray,
    demographic_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not demographic_rows:
        return []

    demo_map = build_demographic_maps(demographic_rows)
    results: list[dict[str, object]] = []
    candidate_variables = (
        "age_midpoint",
        "gender_code",
        "education",
        "handedness",
        "smoking_num",
        "smoking",
        "drug_positive",
        "relationship_status",
    )

    for feature_idx, feature_name in enumerate(backfit_feature_names):
        x = backfit_matrix[:, feature_idx]
        if np.sum(np.isfinite(x)) < 10 or np.nanstd(x) == 0:
            continue
        for variable in candidate_variables:
            paired: list[tuple[float, object]] = []
            for participant, value in zip(participants, x):
                if not math.isfinite(value) or participant not in demo_map:
                    continue
                demo_value = demo_map[participant].get(variable)
                if isinstance(demo_value, str):
                    if demo_value:
                        paired.append((float(value), demo_value))
                elif isinstance(demo_value, float):
                    if math.isfinite(demo_value):
                        paired.append((float(value), float(demo_value)))
            if len(paired) < 10:
                continue

            numeric_values = [value for _, value in paired if isinstance(value, float)]
            if len(numeric_values) == len(paired):
                y = np.array([value for _, value in paired], dtype=float)
                if np.unique(y).size == 2:
                    groups = sorted(np.unique(y))
                    x0 = np.array([xv for xv, yv in paired if yv == groups[0]], dtype=float)
                    x1 = np.array([xv for xv, yv in paired if yv == groups[1]], dtype=float)
                    if x0.size >= 3 and x1.size >= 3:
                        t_stat, p_value = stats.ttest_ind(x0, x1, equal_var=False)
                        results.append(
                            {
                                "feature_name": feature_name,
                                "demographic_variable": variable,
                                "analysis_type": "welch_t_test",
                                "n": int(len(paired)),
                                "group_labels": f"{groups[0]} vs {groups[1]}",
                                "statistic": float(t_stat),
                                "effect_size": cohen_d(x0, x1),
                                "effect_size_name": "cohen_d",
                                "p_value": float(p_value),
                            }
                        )
                elif np.unique(y).size >= 3:
                    rho, p_value = stats.spearmanr(
                        np.array([xv for xv, _ in paired], dtype=float), y
                    )
                    results.append(
                        {
                            "feature_name": feature_name,
                            "demographic_variable": variable,
                            "analysis_type": "spearman",
                            "n": int(len(paired)),
                            "group_labels": "",
                            "statistic": float(rho),
                            "effect_size": float(rho),
                            "effect_size_name": "spearman_rho",
                            "p_value": float(p_value),
                        }
                    )
                continue

            labels = [str(value).strip().lower() for _, value in paired if str(value).strip()]
            unique_labels = sorted(set(labels))
            if len(unique_labels) < 2:
                continue
            grouped = [
                np.array([xv for xv, yv in paired if str(yv).strip().lower() == label], dtype=float)
                for label in unique_labels
            ]
            grouped = [group for group in grouped if group.size >= 3]
            if len(grouped) < 2:
                continue
            h_stat, p_value = stats.kruskal(*grouped)
            n_total = sum(group.size for group in grouped)
            epsilon_sq = (
                float((h_stat - len(grouped) + 1) / (n_total - len(grouped)))
                if n_total > len(grouped)
                else np.nan
            )
            results.append(
                {
                    "feature_name": feature_name,
                    "demographic_variable": variable,
                    "analysis_type": "kruskal_wallis",
                    "n": int(n_total),
                    "group_labels": " | ".join(unique_labels),
                    "statistic": float(h_stat),
                    "effect_size": epsilon_sq,
                    "effect_size_name": "epsilon_squared",
                    "p_value": float(p_value),
                }
            )

    q_values = fdr_bh([row["p_value"] for row in results])
    for row, q_value in zip(results, q_values):
        row["fdr_q_value"] = q_value
    results.sort(key=lambda row: (row["fdr_q_value"], row["p_value"], row["feature_name"]))
    return results


def top_rows_text(rows: list[dict[str, object]], limit: int, columns: list[str]) -> str:
    if not rows:
        return "None\n"
    lines = []
    for row in rows[:limit]:
        parts = [f"{column}={row.get(column, '')}" for column in columns]
        lines.append(" | ".join(parts))
    return "\n".join(lines) + "\n"


def write_summary_report(
    path: Path,
    config: argparse.Namespace,
    pca_result: dict[str, object],
    psychometric_file_rows: list[dict[str, object]],
    method_summaries: list[dict[str, object]],
    demographic_inventory_rows: list[dict[str, object]],
) -> None:
    lines = []
    lines.append("Hierarchical backfit behavioural analysis")
    lines.append("")
    lines.append(f"State metrics CSV: {config.state_metrics.resolve()}")
    lines.append(f"Record summary CSV: {config.record_summary.resolve()}")
    lines.append(f"Behaviour directory: {config.behaviour_dir.resolve()}")
    lines.append(f"Name-match CSV: {config.name_match_csv.resolve()}")
    lines.append(f"Demographics CSV: {config.demographics_csv.resolve()}")
    lines.append("")
    lines.append("Psychometric PCA")
    lines.append(
        f"Participants in psychometric PCA: {len(pca_result['participants'])}"
    )
    lines.append(
        f"Psychometric variables retained: {len(pca_result['feature_names'])}"
    )
    lines.append(
        f"PCs retained: {pca_result['retained_components']}"
    )
    if pca_result["retained_components"]:
        explained = pca_result["explained_variance_ratio"]
        cumulative = pca_result["cumulative_variance_ratio"]
        for idx, (var_ratio, cum_ratio) in enumerate(zip(explained, cumulative), start=1):
            lines.append(
                f"  PC{idx}: explained_variance={float(var_ratio):.4f}, cumulative={float(cum_ratio):.4f}"
            )
    lines.append("")
    lines.append("Psychometric files used")
    used_files = [row for row in psychometric_file_rows if row.get("status") == "used"]
    if used_files:
        for row in used_files:
            lines.append(
                f"  {row['file']}: overlap={row['n_overlap_with_backfit']}, "
                f"numeric_features_used={row['n_numeric_features_used']}"
            )
    else:
        lines.append("  None")
    lines.append("")
    lines.append("Demographic overlap")
    if demographic_inventory_rows:
        for row in demographic_inventory_rows:
            lines.append(f"  {row['variable']}: n_nonmissing_overlap={row['n_nonmissing_overlap']}")
    else:
        lines.append("  No overlapping demographic rows were found.")
    lines.append("")

    for method_summary in method_summaries:
        lines.append(f"Backfit method: {method_summary['method']}")
        lines.append(
            f"  participants={method_summary['participants']}, "
            f"features={method_summary['features']}, "
            f"state_rows_used={method_summary['state_rows_used']}, "
            f"record_rows_used={method_summary['record_rows_used']}"
        )
        lines.append(
            f"  psychometric_pc_tests={method_summary['pc_test_count']}, "
            f"demographic_tests={method_summary['demographic_test_count']}"
        )
        lines.append("  top_pc_hits:")
        lines.append(
            top_rows_text(
                method_summary["pc_rows"],
                5,
                ["feature_name", "pc_name", "pearson_r", "p_value", "fdr_q_value"],
            ).rstrip("\n")
        )
        lines.append("  top_demographic_hits:")
        lines.append(
            top_rows_text(
                method_summary["demographic_rows"],
                5,
                [
                    "feature_name",
                    "demographic_variable",
                    "analysis_type",
                    "effect_size",
                    "p_value",
                    "fdr_q_value",
                ],
            ).rstrip("\n")
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    state_metrics_path = args.state_metrics.expanduser().resolve()
    record_summary_path = args.record_summary.expanduser().resolve()
    behaviour_dir = args.behaviour_dir.expanduser().resolve()
    name_match_csv = args.name_match_csv.expanduser().resolve()
    demographics_csv = args.demographics_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not state_metrics_path.is_file():
        raise SystemExit(f"State metrics CSV not found: {state_metrics_path}")
    if not record_summary_path.is_file():
        raise SystemExit(f"Record summary CSV not found: {record_summary_path}")
    if not behaviour_dir.is_dir():
        raise SystemExit(f"Behaviour directory not found: {behaviour_dir}")
    if not name_match_csv.is_file():
        raise SystemExit(f"Name-match CSV not found: {name_match_csv}")

    participant_aliases = load_name_match_map(name_match_csv)
    state_rows = read_csv_dicts(state_metrics_path)
    record_rows = read_csv_dicts(record_summary_path)
    available_methods = sorted({row.get("backfit_method", "").strip() for row in state_rows if row.get("backfit_method", "").strip()})
    methods = args.backfit_methods or available_methods
    conditions = set(args.conditions)
    include_deltas = not args.disable_condition_deltas

    all_backfit_participants = {
        resolve_participant_id(row.get("participant"), participant_aliases)
        for row in state_rows
        if resolve_participant_id(row.get("participant"), participant_aliases)
    }

    psychometric_participants, psychometric_features, psychometric_matrix, psychometric_inventory_rows, psychometric_file_rows = load_psychometric_matrix(
        behaviour_dir=behaviour_dir,
        participant_pool=all_backfit_participants,
        min_overlap=int(args.min_psychometric_overlap),
        min_feature_coverage=float(args.min_feature_coverage),
        participant_aliases=participant_aliases,
    )

    pca_result = run_psychometric_pca(
        participants=psychometric_participants,
        feature_names=psychometric_features,
        matrix=psychometric_matrix,
        max_components=int(args.max_components),
        variance_threshold=float(args.variance_threshold),
    )

    demographic_rows, demographic_inventory_rows = load_demographics(
        demographics_csv, all_backfit_participants, participant_aliases
    )

    write_dict_rows(
        output_dir / "psychometric_file_inventory.csv",
        psychometric_file_rows,
        [
            "file",
            "participant_column_index",
            "n_participants_in_file",
            "n_overlap_with_backfit",
            "n_numeric_features_used",
            "status",
        ],
    )
    write_dict_rows(
        output_dir / "psychometric_feature_inventory.csv",
        psychometric_inventory_rows,
        [
            "file",
            "column_index",
            "source_column",
            "feature_name",
            "n_overlap_with_backfit",
            "n_nonempty_values",
            "n_numeric_values",
            "n_overlap_nonmissing",
            "status",
        ],
    )
    if demographic_inventory_rows:
        write_dict_rows(
            output_dir / "demographic_overlap_inventory.csv",
            demographic_inventory_rows,
            ["variable", "n_nonmissing_overlap"],
        )

    if pca_result["participants"]:
        write_matrix_csv(
            output_dir / "psychometric_raw_matrix.csv",
            pca_result["participants"],
            pca_result["feature_names"],
            pca_result["raw_matrix"],
        )
        write_matrix_csv(
            output_dir / "psychometric_pc_scores.csv",
            pca_result["participants"],
            pca_result["pc_names"],
            pca_result["pc_scores"],
        )
        pca_summary_rows = []
        for idx, pc_name in enumerate(pca_result["pc_names"]):
            pca_summary_rows.append(
                {
                    "pc_name": pc_name,
                    "explained_variance_ratio": float(pca_result["explained_variance_ratio"][idx]),
                    "cumulative_variance_ratio": float(pca_result["cumulative_variance_ratio"][idx]),
                }
            )
        write_dict_rows(
            output_dir / "psychometric_pca_summary.csv",
            pca_summary_rows,
            ["pc_name", "explained_variance_ratio", "cumulative_variance_ratio"],
        )

        loading_rows = []
        loadings = pca_result["loadings"]
        weights = pca_result["component_weights"]
        for feature_idx, feature_name in enumerate(pca_result["feature_names"]):
            row = {"feature_name": feature_name}
            for pc_idx, pc_name in enumerate(pca_result["pc_names"]):
                row[f"{pc_name}_loading"] = float(loadings[feature_idx, pc_idx])
                row[f"{pc_name}_weight"] = float(weights[feature_idx, pc_idx])
            loading_rows.append(row)
        loading_fields = ["feature_name"] + [
            f"{pc_name}_{suffix}"
            for pc_name in pca_result["pc_names"]
            for suffix in ("loading", "weight")
        ]
        write_dict_rows(output_dir / "psychometric_pc_loadings.csv", loading_rows, loading_fields)

    method_summaries: list[dict[str, object]] = []
    for method in methods:
        method_dir = output_dir / sanitize_name(f"method_{method}")
        participants, feature_names, backfit_matrix, summary = build_backfit_feature_matrix(
            state_rows=state_rows,
            record_rows=record_rows,
            method=method,
            conditions=conditions,
            include_deltas=include_deltas,
            participant_aliases=participant_aliases,
        )
        if participants and feature_names:
            write_matrix_csv(
                method_dir / "backfit_feature_matrix.csv",
                participants,
                feature_names,
                backfit_matrix,
            )

        pc_rows: list[dict[str, object]] = []
        if participants and pca_result["participants"] and pca_result["pc_names"]:
            common, backfit_common, pc_common = intersect_rows(
                participants,
                backfit_matrix,
                pca_result["participants"],
                pca_result["pc_scores"],
            )
            if common:
                pc_rows = correlate_backfit_with_pcs(
                    common,
                    feature_names,
                    backfit_common,
                    pca_result["pc_names"],
                    pc_common,
                )
                if pc_rows:
                    write_dict_rows(
                        method_dir / "backfit_vs_psychometric_pcs.csv",
                        pc_rows,
                        ["feature_name", "pc_name", "n", "pearson_r", "p_value", "fdr_q_value"],
                    )

        demographic_rows_out: list[dict[str, object]] = []
        if participants and feature_names and demographic_rows:
            common_demo, backfit_demo, _ = intersect_rows(
                participants,
                backfit_matrix,
                [row["participant"] for row in demographic_rows],
                np.zeros((len(demographic_rows), 1), dtype=float),
            )
            if common_demo:
                kept_demographics = [
                    row for row in demographic_rows if row["participant"] in set(common_demo)
                ]
                demographic_rows_out = analyze_demographic_effects(
                    common_demo,
                    feature_names,
                    backfit_demo,
                    kept_demographics,
                )
                if demographic_rows_out:
                    write_dict_rows(
                        method_dir / "demographic_effects.csv",
                        demographic_rows_out,
                        [
                            "feature_name",
                            "demographic_variable",
                            "analysis_type",
                            "n",
                            "group_labels",
                            "statistic",
                            "effect_size",
                            "effect_size_name",
                            "p_value",
                            "fdr_q_value",
                        ],
                    )

        method_summaries.append(
            {
                **summary,
                "pc_rows": pc_rows,
                "pc_test_count": len(pc_rows),
                "demographic_rows": demographic_rows_out,
                "demographic_test_count": len(demographic_rows_out),
            }
        )

    write_summary_report(
        output_dir / "analysis_summary.txt",
        args,
        pca_result,
        psychometric_file_rows,
        method_summaries,
        demographic_inventory_rows,
    )

    print(f"Wrote behavioural backfit analysis to: {output_dir}")


if __name__ == "__main__":
    main()
