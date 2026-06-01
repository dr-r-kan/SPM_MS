#!/usr/bin/env python3
"""Plot per-K microstate model diagnostics from JSON exports.

This script expects JSON files produced by ``analyze_single_eeg_file`` via
``save_microstate_json``. It can aggregate per-K silhouette/free-energy
diagnostics across files, and can relate selected-model free energy to:

1. similarity between an individual solution and a dataset-level global
   solution (provided as a CSV of centers), and
2. similarity between an individual solution and the canonical template
   alignment saved in the JSON metadata.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import plotly.graph_objects as go

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - fallback only
    linear_sum_assignment = None


@dataclass
class ModelDiagnostic:
    json_path: str
    subject: str
    condition: str
    k_candidates: np.ndarray
    silhouette_vals: np.ndarray
    free_energy_vals: np.ndarray
    selected_k: float
    selected_free_energy: float
    selected_silhouette: float
    template_similarity: float
    centers: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot microstate silhouette/free-energy diagnostics from JSON exports."
    )
    parser.add_argument(
        "--json-glob",
        action="append",
        default=[],
        help="Glob pattern for input JSON files. May be repeated.",
    )
    parser.add_argument(
        "--json-dir",
        default="",
        help="Directory to search recursively for JSON files if --json-glob is omitted.",
    )
    parser.add_argument(
        "--global-centers-csv",
        default="",
        help="Optional CSV of global/meta microstate centers for solution-vs-global similarity.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plots and summary CSVs will be written.",
    )
    parser.add_argument(
        "--prefix",
        default="microstate_model_diagnostics",
        help="Filename prefix for plot and CSV outputs.",
    )
    return parser.parse_args()


def find_json_files(args: argparse.Namespace) -> List[str]:
    paths: List[str] = []
    patterns = list(args.json_glob)
    if not patterns:
        if args.json_dir:
            patterns = [os.path.join(args.json_dir, "**", "*.json")]
        else:
            patterns = [os.path.join("outputs", "json", "*.json")]
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    paths = sorted(set(os.path.abspath(p) for p in paths if os.path.isfile(p)))
    if not paths:
        raise FileNotFoundError("No JSON files matched the requested inputs.")
    return paths


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_condition(path: str, metadata: dict) -> str:
    if "condition" in metadata and metadata["condition"]:
        return str(metadata["condition"])
    stem = Path(path).stem.lower()
    if "_ec" in stem or "eyes_closed" in stem:
        return "eyes_closed"
    if "_eo" in stem or "eyes_open" in stem:
        return "eyes_open"
    return ""


def state_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.split("_")[-1]), name
    except Exception:
        return (10**9, name)


def extract_centers(data: dict) -> np.ndarray:
    est = data.get("estimated_microstates", {})
    ch_labels = data.get("channel_info", {}).get("labels_sanitized", [])
    if not est or not ch_labels:
        return np.empty((0, 0), dtype=float)
    rows: List[List[float]] = []
    for key in sorted(est.keys(), key=state_sort_key):
        state = est[key]
        rows.append([float(state[label]) for label in ch_labels])
    return np.asarray(rows, dtype=float)


def template_similarity_from_json(data: dict) -> float:
    meta = data.get("metadata", {})
    ta = meta.get("template_alignment", {})
    if isinstance(ta, dict):
        if "mean_correlation" in ta:
            return float(ta["mean_correlation"])
        corr = ta.get("correlations", [])
        if corr:
            vals = np.asarray(corr, dtype=float)
            return float(np.nanmean(vals))

    est = data.get("estimated_microstates", {})
    vals = []
    for state in est.values():
        if isinstance(state, dict) and "template_correlation" in state:
            vals.append(float(state["template_correlation"]))
    if vals:
        return float(np.nanmean(vals))
    return math.nan


def load_diagnostic(path: str) -> ModelDiagnostic:
    data = load_json(path)
    meta = data.get("metadata", {})
    k_candidates = np.asarray(meta.get("K_candidates", []), dtype=float)
    silhouette_vals = np.asarray(meta.get("silhouette_vals", []), dtype=float)
    free_energy_vals = np.asarray(meta.get("free_energy_vals", []), dtype=float)
    n = min(len(k_candidates), len(silhouette_vals), len(free_energy_vals))
    if n == 0:
        raise ValueError(f"{path} does not contain per-K silhouette/free-energy arrays.")
    k_candidates = k_candidates[:n]
    silhouette_vals = silhouette_vals[:n]
    free_energy_vals = free_energy_vals[:n]

    selected_k = float(meta.get("K_model_selected", meta.get("K_estimated", math.nan)))
    if math.isnan(selected_k):
        raise ValueError(f"{path} is missing K_model_selected/K_estimated.")
    selected_idx = first_index_equal(k_candidates, selected_k)
    if selected_idx is None:
        selected_idx = int(meta.get("selected_model_index", 1)) - 1
    selected_free_energy = float(
        meta.get("selected_model_free_energy", free_energy_vals[selected_idx])
    )
    selected_silhouette = float(
        meta.get("selected_model_silhouette", silhouette_vals[selected_idx])
    )

    subject = str(meta.get("subject", Path(path).stem))
    condition = infer_condition(path, meta)
    centers = extract_centers(data)
    return ModelDiagnostic(
        json_path=path,
        subject=subject,
        condition=condition,
        k_candidates=k_candidates,
        silhouette_vals=silhouette_vals,
        free_energy_vals=free_energy_vals,
        selected_k=selected_k,
        selected_free_energy=selected_free_energy,
        selected_silhouette=selected_silhouette,
        template_similarity=template_similarity_from_json(data),
        centers=centers,
    )


def first_index_equal(arr: np.ndarray, value: float) -> int | None:
    hits = np.where(np.isclose(arr, value))[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def read_centers_csv(path: str) -> np.ndarray:
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return np.empty((0, 0), dtype=float)
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    if not rows:
        return np.empty((0, len(header)), dtype=float)
    return np.asarray(rows, dtype=float)


def normalize_topographies(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("Expected a 2D array of topographies.")
    x = x - np.nanmean(x, axis=1, keepdims=True)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms <= np.finfo(float).eps] = 1.0
    return x / norms


def mean_optimal_abs_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return math.nan
    a_n = normalize_topographies(a)
    b_n = normalize_topographies(b)
    sim = np.abs(a_n @ b_n.T)
    n_assign = min(sim.shape)
    if n_assign == 0:
        return math.nan
    if linear_sum_assignment is not None:
        cost = -sim
        row_ind, col_ind = linear_sum_assignment(cost)
        return float(np.mean(sim[row_ind, col_ind]))
    chosen_cols = set()
    vals = []
    for r in range(sim.shape[0]):
        col_order = np.argsort(sim[r])[::-1]
        for c in col_order:
            if c not in chosen_cols:
                chosen_cols.add(int(c))
                vals.append(sim[r, c])
                break
        if len(vals) >= n_assign:
            break
    if not vals:
        return math.nan
    return float(np.mean(vals))


def write_per_k_csv(rows: Sequence[dict], path: str) -> None:
    fieldnames = ["json_path", "subject", "condition", "K", "silhouette", "free_energy"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_solution_csv(rows: Sequence[dict], path: str) -> None:
    fieldnames = [
        "json_path",
        "subject",
        "condition",
        "selected_k",
        "selected_free_energy",
        "selected_silhouette",
        "global_similarity",
        "template_similarity",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def boxplot_metric(per_k_rows: Sequence[dict], metric_key: str, ylabel: str, out_path: str) -> None:
    grouped: dict[int, List[float]] = {}
    for row in per_k_rows:
        k = int(row["K"])
        grouped.setdefault(k, []).append(float(row[metric_key]))
    ks = sorted(grouped)
    fig = go.Figure()
    for k in ks:
        fig.add_trace(
            go.Box(
                y=grouped[k],
                name=str(k),
                boxpoints=False,
                marker_color="#3366cc",
            )
        )
    fig.update_layout(
        title=f"{ylabel} across files by K",
        xaxis_title="K",
        yaxis_title=ylabel,
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def scatter_metric(
    solution_rows: Sequence[dict],
    y_key: str,
    ylabel: str,
    out_path: str,
) -> None:
    rows = [row for row in solution_rows if math.isfinite(float(row[y_key]))]
    if not rows:
        return
    x = np.asarray([float(row["selected_free_energy"]) for row in rows], dtype=float)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=float)
    hover = [
        f"{row['subject']} | {row['condition']} | K={row['selected_k']}"
        for row in rows
    ]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(size=9, color="#c44e52", opacity=0.8),
                text=hover,
                hovertemplate="%{text}<br>free energy=%{x:.3f}<br>similarity=%{y:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=f"{ylabel} vs selected-model free energy",
        xaxis_title="Selected-model free energy",
        yaxis_title=ylabel,
        template="plotly_white",
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = [load_diagnostic(path) for path in find_json_files(args)]
    global_centers = None
    if args.global_centers_csv:
        global_centers = read_centers_csv(args.global_centers_csv)

    per_k_rows: List[dict] = []
    solution_rows: List[dict] = []
    for diag in diagnostics:
        for k, sil, fe in zip(diag.k_candidates, diag.silhouette_vals, diag.free_energy_vals):
            if math.isfinite(float(sil)) and math.isfinite(float(fe)):
                per_k_rows.append(
                    {
                        "json_path": diag.json_path,
                        "subject": diag.subject,
                        "condition": diag.condition,
                        "K": int(k),
                        "silhouette": float(sil),
                        "free_energy": float(fe),
                    }
                )
        global_similarity = math.nan
        if global_centers is not None and global_centers.size and diag.centers.size:
            global_similarity = mean_optimal_abs_corr(diag.centers, global_centers)
        solution_rows.append(
            {
                "json_path": diag.json_path,
                "subject": diag.subject,
                "condition": diag.condition,
                "selected_k": int(diag.selected_k),
                "selected_free_energy": float(diag.selected_free_energy),
                "selected_silhouette": float(diag.selected_silhouette),
                "global_similarity": float(global_similarity),
                "template_similarity": float(diag.template_similarity),
            }
        )

    if not per_k_rows:
        raise RuntimeError("No valid per-K rows were extracted from the JSON inputs.")

    per_k_csv = output_dir / f"{args.prefix}_per_k_summary.csv"
    solution_csv = output_dir / f"{args.prefix}_solution_summary.csv"
    write_per_k_csv(per_k_rows, str(per_k_csv))
    write_solution_csv(solution_rows, str(solution_csv))

    boxplot_metric(
        per_k_rows,
        metric_key="silhouette",
        ylabel="Silhouette score",
        out_path=str(output_dir / f"{args.prefix}_silhouette_boxplot.html"),
    )
    boxplot_metric(
        per_k_rows,
        metric_key="free_energy",
        ylabel="Free energy",
        out_path=str(output_dir / f"{args.prefix}_free_energy_boxplot.html"),
    )
    scatter_metric(
        solution_rows,
        y_key="global_similarity",
        ylabel="Similarity to global solution",
        out_path=str(output_dir / f"{args.prefix}_global_similarity_vs_free_energy.html"),
    )
    scatter_metric(
        solution_rows,
        y_key="template_similarity",
        ylabel="Similarity to template",
        out_path=str(output_dir / f"{args.prefix}_template_similarity_vs_free_energy.html"),
    )

    print(f"Wrote per-K summary: {per_k_csv}")
    print(f"Wrote solution summary: {solution_csv}")
    print(f"Wrote plots under: {output_dir}")


if __name__ == "__main__":
    main()
