#!/usr/bin/env python3
"""
Compile HEPPy microstate analysis outputs into a readable HTML report.

Run after heppy_microstate_publication_stats.py, for example:

    python compile_heppy_microstate_report.py \
        --stats-root /path/to/heppy_microstates_matlab/python_publication_stats

The report is deliberately conservative: it displays inclusion counts,
analysis metadata, key model tables sorted by p value, and all generated PNG
figures.  It does not reinterpret the results beyond the table contents.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


P_VALUE_COLUMNS = [
    "p_fdr_bh_across_cardiac_phase_E_effects",
    "p_fdr_bh_across_accuracy_linkage",
    "p_fdr_bh_group_hit_rate_differences",
    "p_fdr_bh_condition_hit_rate_differences",
    "p_fdr_bh_cardiac_phase_omnibus",
    "p_fdr_bh_within_cardiac_phase_curve",
    "p_fdr_bh_across_latency_consistency_phase_models",
    "p_fdr_bh_within_latency_consistency_curve",
    "p_fdr_bh_across_circular_terms",
    "p_fdr_bh_across_all_terms",
    "p_fdr_bh_across_all_coefficients",
    "p_fdr_bh",
    "p_value",
]

KEY_TABLES = [
    ("analysis_manifest.csv", "Input manifest"),
    ("microstate_state_metrics.csv", "Participant-condition microstate metrics"),
    ("microstate_group_condition_state_means.csv", "Group × condition × state means"),
    ("microstate_group_condition_state_model_terms.csv", "Group/condition/state model terms"),
    ("microstate_group_condition_state_model_coefficients.csv", "Group/condition/state coefficients"),
    ("circular_block_features.csv", "Participant-condition circular Fourier features"),
    ("circular_hotelling_by_state_group_condition.csv", "Circular Hotelling tests"),
    ("circular_fourier_component_model_terms.csv", "Circular Fourier component model terms"),
    ("heartbeat_tapping_hit_miss_by_condition.csv", "Heartbeat-tapping hit/miss rates"),
    ("heartbeat_tapping_hit_miss_group_condition_descriptives.csv", "Hit/miss descriptive rates by group and condition"),
    ("heartbeat_tapping_hit_rate_group_differences_by_condition.csv", "Descriptive group differences in hit rate"),
    ("heartbeat_tapping_hit_rate_condition_differences_within_group.csv", "Descriptive condition differences in hit rate"),
    ("heartbeat_tapping_tap_latency_to_previous_r_descriptives.csv", "Tap latency to previous R peak"),
    ("heartbeat_tapping_latency_consistency_by_condition.csv", "Tap-latency consistency by participant and condition"),
    ("heartbeat_tapping_with_microstate_variables.csv", "Tapping metrics merged with microstate variables"),
    ("heartbeat_tapping_microstate_linkage_binomial_coefficients.csv", "Hit-rate associations with microstate variables"),
    ("beat_level_tapping_microstate_E_windows.csv", "Beat-level tapping and microstate E windows"),
    ("cardiac_phase_beat_E_effect_coefficients.csv", "Systole/diastole beat-level E-effect coefficients"),
    ("cardiac_phase_beat_E_effect_omnibus_gee_tests.csv", "Systole/diastole omnibus GEE tests"),
    ("cardiac_phase_beat_E_effect_significant_clusters.csv", "Systole/diastole significant period clusters"),
    ("cardiac_phase_beat_E_effect_case_control_contrasts.csv", "Systole/diastole case-control E-effect contrasts"),
]

EXCLUDED_EXTRA_TABLES = {
    "beat_by_beat_E_window_gee_coefficients.csv",
    "beat_by_beat_E_window_case_control_effects.csv",
    "cardiac_phase_latency_consistency_E_effect_coefficients.csv",
    "cardiac_phase_latency_consistency_E_effect_case_control_contrasts.csv",
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def choose_sort_col(df: pd.DataFrame) -> Optional[str]:
    for col in P_VALUE_COLUMNS:
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            return col
    return None


def trim_table(df: pd.DataFrame, max_rows: int = 50) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    sort_col = choose_sort_col(out)
    if sort_col is not None:
        out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
        out = out.sort_values(sort_col, na_position="last")
    if len(out) > max_rows:
        out = out.head(max_rows)
    # Avoid enormous columns in the report.
    keep_cols = []
    for col in out.columns:
        if col.lower() in {"sequence_csv", "source_fif", "file_path", "raw_fif"}:
            continue
        keep_cols.append(col)
    out = out[keep_cols]
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.5g}")
    return out


def dataframe_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "<p><em>No rows available.</em></p>"
    small = trim_table(df, max_rows=max_rows)
    return small.to_html(index=False, escape=True, border=0, classes="data-table")


def count_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    parts = [f"Rows: <strong>{len(df):,}</strong>"]
    for col in ("study", "group", "participant", "condition", "microstate"):
        if col in df.columns:
            n = df[col].dropna().nunique()
            parts.append(f"{html.escape(col)}: <strong>{n:,}</strong>")
    return "<p>" + " &nbsp; | &nbsp; ".join(parts) + "</p>"


def metadata_html(stats_root: Path) -> str:
    meta_path = stats_root / "run_metadata.json"
    if not meta_path.exists():
        return ""
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return ""
    rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in payload.items()
    )
    return f"<h2>Run metadata</h2><table class='meta-table'>{rows}</table>"


def image_section(stats_root: Path) -> str:
    fig_dir = stats_root / "figures"
    if not fig_dir.exists():
        return ""
    imgs = [
        img for img in sorted(fig_dir.glob("*.png"))
        if img.name.startswith("cardiac_phase_") and "bayes" in img.name
    ]
    if not imgs:
        return ""
    chunks = ["<h2>Figures</h2>"]
    for img in imgs:
        rel = img.relative_to(stats_root).as_posix()
        title = img.stem.replace("_", " ")
        chunks.append(
            f"<figure><figcaption>{html.escape(title)}</figcaption>"
            f"<img src='{html.escape(rel)}' alt='{html.escape(title)}'></figure>"
        )
    return "\n".join(chunks)


def table_sections(stats_root: Path, max_rows: int) -> str:
    table_dir = stats_root / "tables"
    chunks = ["<h2>Tables</h2>"]
    for filename, title in KEY_TABLES:
        path = table_dir / filename
        if not path.exists():
            continue
        df = read_csv(path)
        chunks.append(f"<section class='table-section'><h3>{html.escape(title)}</h3>")
        chunks.append(f"<p class='path'>tables/{html.escape(filename)}</p>")
        chunks.append(count_summary(df))
        chunks.append(dataframe_html(df, max_rows=max_rows))
        chunks.append("</section>")
    # Include any extra CSVs not listed explicitly.
    known = {name for name, _ in KEY_TABLES}
    for path in sorted(table_dir.glob("*.csv")):
        if path.name in known or path.name in EXCLUDED_EXTRA_TABLES:
            continue
        df = read_csv(path)
        chunks.append(f"<section class='table-section'><h3>{html.escape(path.stem.replace('_', ' '))}</h3>")
        chunks.append(f"<p class='path'>tables/{html.escape(path.name)}</p>")
        chunks.append(count_summary(df))
        chunks.append(dataframe_html(df, max_rows=max_rows))
        chunks.append("</section>")
    return "\n".join(chunks)


def methods_text() -> str:
    return """
    <h2>Analysis synopsis</h2>
    <p>The report summarises the outputs of the HEPPy microstate downstream
    pipeline. Microstate coverage, GFP, occurrence rate, and duration are
    analysed at participant-condition-state level. Cardiac-cycle modulation is
    reduced to participant-condition Fourier features and tested with circular
    vector statistics. Heartbeat-tapping hit/miss rates are computed per
    participant-condition from valid taps around R peaks. Beat-level prediction
    of tapping success uses pooled/study GEE binomial models clustered by
    participant. A cardiac-period analysis repeats the beat-level model for
    systole, defined as 0.35-0.60 s after the R peak, and diastole, defined as
    the rest of the RR interval, for both the current R-to-next-R cycle and the
    previous R-to-R cycle. It then applies empirical-Bayes shrinkage to the GEE
    period estimates for stable Bayesian posterior intervals. The analysis also
    reports within-contrast FDR adjustment and omnibus GEE Wald tests. Tap timing is
    summarised as latency from the previous R peak globally and by group. The
    figures section is restricted to Bayesian
    cardiac-period plots; other generated PNGs remain in the figures
    directory.</p>
    <p>Tables are sorted by the most relevant available p-value or FDR-adjusted
    p-value when present. The displayed rows are truncated for readability; the
    full CSVs remain in the <code>tables</code> directory.</p>
    """


def build_html(stats_root: Path, max_rows: int) -> str:
    title = "HEPPy cardiac-cycle microstate analysis report"
    css = """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; line-height: 1.45; color: #222; }
    h1, h2, h3 { color: #111; }
    code { background: #f2f2f2; padding: 0.1rem 0.25rem; border-radius: 3px; }
    .path { font-size: 0.9rem; color: #555; font-family: monospace; }
    table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1.5rem 0; font-size: 0.86rem; }
    th, td { border-bottom: 1px solid #ddd; padding: 0.35rem 0.45rem; vertical-align: top; }
    th { background: #f7f7f7; text-align: left; position: sticky; top: 0; }
    .meta-table th { width: 16rem; }
    .table-section { margin-bottom: 2.4rem; }
    figure { margin: 1.5rem 0; padding: 1rem; border: 1px solid #ddd; border-radius: 6px; }
    figcaption { font-weight: 600; margin-bottom: 0.75rem; }
    img { max-width: 100%; height: auto; display: block; }
    .note { color: #555; }
    </style>
    """
    return "\n".join([
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        css,
        "</head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='note'>Stats root: <code>{html.escape(str(stats_root))}</code></p>",
        metadata_html(stats_root),
        methods_text(),
        table_sections(stats_root, max_rows=max_rows),
        image_section(stats_root),
        "</body></html>",
    ])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compile HEPPy microstate downstream results into an HTML report.")
    p.add_argument("--stats-root", type=Path, default="/Users/rohan/EEG/Microstates_and_Interoception/output/heppy_microstates_matlab/python_publication_stats", help="Output directory from heppy_microstate_publication_stats.py")
    p.add_argument("--output", type=Path, default=None, help="HTML output path; default is <stats-root>/microstate_publication_report.html")
    p.add_argument("--max-rows", type=int, default=50, help="Maximum rows displayed per table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stats_root = args.stats_root.resolve()
    out = args.output or (stats_root / "microstate_publication_report.html")
    html_text = build_html(stats_root, max_rows=int(args.max_rows))
    out.write_text(html_text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
