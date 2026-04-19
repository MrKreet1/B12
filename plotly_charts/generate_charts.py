from __future__ import annotations

from pathlib import Path
import re
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import write_html


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
STARTS_DIR = ROOT / "starts"
JOBS_DIR = ROOT / "jobs"
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "out"

HARTREE_TO_KJ_MOL = 2625.49962

MOTIF_ORDER = ["double_ring_6_6", "ring12", "zigzag_chain12", "random", "other"]
MOTIF_LABELS = {
    "double_ring_6_6": "Double ring 6+6",
    "ring12": "Ring12",
    "zigzag_chain12": "Zigzag chain12",
    "random": "Random",
    "other": "Other",
}
MOTIF_COLORS = {
    "Double ring 6+6": "#0F766E",
    "Ring12": "#B45309",
    "Zigzag chain12": "#1D4ED8",
    "Random": "#6B7280",
    "Other": "#374151",
}
SCREENING_COLORS = {
    "Converged minimum": "#0F766E",
    "Normal termination only": "#D97706",
    "Abnormal termination": "#B91C1C",
}
FINAL_STATUS_COLORS = {
    "OK_NO_IMAG": "#0F766E",
    "TINY_NEG_ONLY": "#D97706",
    "REJECT_IMAG": "#B91C1C",
}

warnings.filterwarnings(
    "ignore",
    message="When grouping with a length-1 list-like, you will need to pass a length-1 tuple",
    category=FutureWarning,
)


def load_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path)


def parse_job_name(job_name: str) -> dict[str, object]:
    text = str(job_name)
    motif = next(
        (candidate for candidate in MOTIF_ORDER if candidate != "other" and text.startswith(candidate)),
        "other",
    )
    distance_match = re.search(r"_d(\d+\.\d+)", text)
    seed_distance = float(distance_match.group(1)) if distance_match else None
    return {
        "motif": motif,
        "motif_label": MOTIF_LABELS[motif],
        "seed_distance_ang": seed_distance,
        "seed_distance_label": f"{seed_distance:.2f} A" if seed_distance is not None else "n/a",
        "display_name": text.replace("_r2scan", ""),
        "is_known_motif": motif != "other",
    }


def normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )


def format_mult(value: object) -> str:
    try:
        return f"M={int(float(value))}"
    except (TypeError, ValueError):
        return "M=?"


def add_common_columns(df: pd.DataFrame, energy_col: str) -> pd.DataFrame:
    out = df.copy()
    parsed = out["job_name"].map(parse_job_name).apply(pd.Series)
    out = pd.concat([out, parsed], axis=1)
    out[energy_col] = pd.to_numeric(out[energy_col], errors="coerce")
    out["mult_label"] = out.get("mult", pd.Series(index=out.index, dtype=object)).map(format_mult)
    out["rel_energy_kj_mol"] = (out[energy_col] - out[energy_col].min()) * HARTREE_TO_KJ_MOL
    return out


def write_plot(fig: go.Figure, filename: str) -> None:
    fig.update_layout(
        template="plotly_white",
        font={"family": "Arial, sans-serif", "size": 14},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 70, "r": 40, "t": 90, "b": 70},
    )
    write_html(
        fig,
        file=OUT_DIR / filename,
        full_html=True,
        include_plotlyjs="directory",
        auto_open=False,
    )


def build_xtb_landscape(xtb_all: pd.DataFrame) -> None:
    plot_df = xtb_all[xtb_all["is_known_motif"]].copy()
    plot_df["screening_status"] = "Abnormal termination"
    plot_df.loc[plot_df["normal_term"], "screening_status"] = "Normal termination only"
    plot_df.loc[plot_df["normal_term"] & plot_df["opt_converged"], "screening_status"] = "Converged minimum"

    fig = px.scatter(
        plot_df.sort_values(["mult_label", "motif_label", "seed_distance_ang", "rel_energy_kj_mol"]),
        x="seed_distance_ang",
        y="rel_energy_kj_mol",
        color="motif_label",
        symbol="screening_status",
        facet_col="mult_label",
        category_orders={"mult_label": ["M=1", "M=3"]},
        color_discrete_map=MOTIF_COLORS,
        hover_name="job_name",
        hover_data={
            "seed_distance_ang": ":.2f",
            "energy_hartree": ":.8f",
            "rel_energy_kj_mol": ":.2f",
            "normal_term": True,
            "opt_converged": True,
            "motif_label": False,
            "mult_label": False,
        },
        title="xTB energy landscape by seed distance",
        labels={
            "seed_distance_ang": "Seed distance (A)",
            "rel_energy_kj_mol": "Relative energy (kJ/mol)",
            "motif_label": "Motif",
            "screening_status": "xTB status",
        },
    )
    fig.update_traces(marker={"size": 10, "opacity": 0.8, "line": {"width": 0.5, "color": "white"}})
    fig.update_xaxes(tickmode="array", tickvals=[1.5, 1.7, 1.9, 2.1])
    fig.for_each_annotation(lambda ann: ann.update(text=ann.text.replace("mult_label=", "")))
    write_plot(fig, "01_xtb_landscape.html")


def build_xtb_unique(xtb_unique: pd.DataFrame) -> None:
    plot_df = xtb_unique.sort_values("rank").copy()
    plot_df["rel_label"] = plot_df["rel_energy_kj_mol"].map(lambda value: f"{value:.2f}")

    fig = px.bar(
        plot_df,
        x="display_name",
        y="rel_energy_kj_mol",
        color="motif_label",
        text="rel_label",
        color_discrete_map=MOTIF_COLORS,
        title="Unique xTB minima selected for DFT",
        labels={
            "display_name": "Structure",
            "rel_energy_kj_mol": "Relative energy (kJ/mol)",
            "motif_label": "Motif",
        },
        hover_name="job_name",
        hover_data={
            "energy_hartree": ":.8f",
            "rel_energy_kj_mol": ":.2f",
            "mult_label": True,
            "motif_label": False,
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(categoryorder="array", categoryarray=plot_df["display_name"].tolist())
    write_plot(fig, "02_xtb_unique_minima.html")


def build_screening_funnel(
    seed_count: int,
    planned_xtb_count: int,
    xtb_all: pd.DataFrame,
    xtb_unique: pd.DataFrame,
    final_numfreq: pd.DataFrame,
    final_good: pd.DataFrame,
) -> None:
    steps = [
        ("Seed geometries", seed_count),
        ("Planned xTB jobs", planned_xtb_count),
        ("xTB normal term", int(xtb_all["normal_term"].sum())),
        ("xTB opt converged", int(xtb_all["opt_converged"].sum())),
        ("Unique xTB minima", len(xtb_unique)),
        ("Final DFT/NumFreq set", len(final_numfreq)),
        ("No imaginary freqs", len(final_good)),
    ]

    fig = go.Figure(
        go.Funnel(
            y=[name for name, _ in steps],
            x=[count for _, count in steps],
            textinfo="value+percent initial",
            marker={
                "color": ["#1D4ED8", "#2563EB", "#0EA5E9", "#0F766E", "#059669", "#D97706", "#B45309"],
            },
            connector={"line": {"color": "#CBD5E1", "width": 1.5}},
        )
    )
    fig.update_layout(title="Pipeline funnel from seeds to validated minima")
    write_plot(fig, "03_screening_funnel.html")


def build_final_dft(final_numfreq: pd.DataFrame) -> None:
    plot_df = final_numfreq.sort_values("numfreq_energy_hartree").copy()
    plot_df["rel_label"] = plot_df["rel_energy_kj_mol"].map(lambda value: f"{value:.2f}")

    fig = px.bar(
        plot_df,
        x="display_name",
        y="rel_energy_kj_mol",
        color="status",
        pattern_shape="mult_label",
        color_discrete_map=FINAL_STATUS_COLORS,
        text="rel_label",
        title="Final DFT + NumFreq ranking",
        labels={
            "display_name": "Structure",
            "rel_energy_kj_mol": "Relative energy (kJ/mol)",
            "status": "Final status",
            "mult_label": "Multiplicity",
        },
        hover_name="job_name",
        hover_data={
            "numfreq_energy_hartree": ":.8f",
            "dft_energy_hartree": ":.8f",
            "min_freq_cm1": ":.2f",
            "n_imag_strict": True,
            "n_imag_significant": True,
            "status": False,
        },
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(categoryorder="array", categoryarray=plot_df["display_name"].tolist())
    write_plot(fig, "04_final_dft_numfreq.html")


def build_frequency_diagnostics(final_numfreq: pd.DataFrame) -> None:
    plot_df = final_numfreq.copy()
    plot_df["bubble_size"] = plot_df["n_imag_strict"].fillna(0).astype(float) + 1.0

    fig = px.scatter(
        plot_df,
        x="rel_energy_kj_mol",
        y="min_freq_cm1",
        size="bubble_size",
        color="status",
        symbol="mult_label",
        text="display_name",
        size_max=48,
        color_discrete_map=FINAL_STATUS_COLORS,
        title="Frequency diagnostics for final structures",
        labels={
            "rel_energy_kj_mol": "Relative energy (kJ/mol)",
            "min_freq_cm1": "Minimum frequency (cm^-1)",
            "status": "Final status",
            "mult_label": "Multiplicity",
        },
        hover_name="job_name",
        hover_data={
            "n_imag_strict": True,
            "n_imag_significant": True,
            "numfreq_energy_hartree": ":.8f",
            "dft_energy_hartree": ":.8f",
            "bubble_size": False,
            "display_name": False,
        },
    )
    fig.update_traces(textposition="top center", marker={"line": {"width": 1, "color": "white"}})
    fig.add_hline(y=0.0, line_dash="dash", line_color="#0F766E", annotation_text="Stable minimum threshold")
    fig.add_hline(y=-20.0, line_dash="dot", line_color="#B91C1C", annotation_text="Significant imag threshold")
    write_plot(fig, "05_frequency_diagnostics.html")


def build_index(seed_count: int, planned_xtb_count: int, final_good_count: int) -> None:
    cards = [
        ("01_xtb_landscape.html", "xTB Landscape", "All screened xTB structures by seed distance, motif, and convergence status."),
        ("02_xtb_unique_minima.html", "Unique xTB Minima", "The unique xTB minima that were promoted to the DFT stage."),
        ("03_screening_funnel.html", "Pipeline Funnel", "Compression of the search space from seed geometries to validated minima."),
        ("04_final_dft_numfreq.html", "Final Ranking", "Relative energies after the final DFT + NumFreq stage."),
        ("05_frequency_diagnostics.html", "Frequency Diagnostics", "Minimum frequency and imaginary-mode diagnostics for the final structures."),
    ]
    summary_html = "".join(
        [
            f"<div class='metric'><span>{label}</span><strong>{value}</strong></div>"
            for label, value in [
                ("Seed geometries", seed_count),
                ("Planned xTB jobs", planned_xtb_count),
                ("Stable minima", final_good_count),
            ]
        ]
    )
    cards_html = "".join(
        [
            (
                "<section class='card'>"
                f"<div class='card-head'><h2>{title}</h2><p>{description}</p>"
                f"<a href='{filename}' target='_blank' rel='noopener'>Open separately</a></div>"
                f"<iframe src='{filename}' loading='lazy' title='{title}'></iframe>"
                "</section>"
            )
            for filename, title, description in cards
        ]
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Plotly charts for the ORCA B12 workflow</title>
  <style>
    :root {{
      --bg: #f3f4f6;
      --panel: #ffffff;
      --text: #111827;
      --muted: #4b5563;
      --line: #d1d5db;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 24%),
        radial-gradient(circle at top right, rgba(29, 78, 216, 0.12), transparent 28%),
        var(--bg);
    }}
    main {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 36px;
      line-height: 1.1;
    }}
    .intro {{
      max-width: 860px;
      color: var(--muted);
      margin-bottom: 22px;
      font-size: 16px;
      line-height: 1.6;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric strong {{
      font-size: 28px;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
    }}
    .card-head {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 22px;
    }}
    .card-head p {{
      margin: 6px 0 0;
      color: var(--muted);
      max-width: 760px;
      line-height: 1.5;
    }}
    .card-head a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    iframe {{
      width: 100%;
      height: 640px;
      border: 0;
      border-radius: 16px;
      background: #fff;
    }}
    @media (max-width: 820px) {{
      main {{ padding: 24px 14px 36px; }}
      h1 {{ font-size: 28px; }}
      iframe {{ height: 520px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Plotly charts for the ORCA B12 workflow</h1>
    <p class="intro">
      This folder contains interactive charts built from the current CSV outputs in <code>results/</code>.
      Open the charts below or launch any standalone HTML file from this directory.
    </p>
    <section class="metrics">{summary_html}</section>
    <section class="grid">{cards_html}</section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    xtb_all = add_common_columns(load_csv("xtb_ranked_all.csv"), "energy_hartree")
    xtb_all = xtb_all[xtb_all["is_known_motif"]].copy()
    xtb_all["normal_term"] = normalize_bool(xtb_all["normal_term"])
    xtb_all["opt_converged"] = normalize_bool(xtb_all["opt_converged"])

    xtb_unique = add_common_columns(load_csv("xtb_ranked_unique.csv"), "energy_hartree")
    final_numfreq = add_common_columns(load_csv("final_numfreq_report.csv"), "numfreq_energy_hartree")
    final_good = load_csv("final_good_no_imag.csv")

    seed_count = len(list(STARTS_DIR.glob("*.xyz")))
    planned_xtb_count = sum(
        1
        for path in (JOBS_DIR / "01_xtb").rglob("*.inp")
        if not path.stem.lower().startswith("test_")
    )

    build_xtb_landscape(xtb_all)
    build_xtb_unique(xtb_unique)
    build_screening_funnel(seed_count, planned_xtb_count, xtb_all, xtb_unique, final_numfreq, final_good)
    build_final_dft(final_numfreq)
    build_frequency_diagnostics(final_numfreq)
    build_index(seed_count, planned_xtb_count, len(final_good))

    generated_files = sorted(path.name for path in OUT_DIR.glob("*.html"))
    print("Generated Plotly files:")
    for filename in generated_files:
        print(f" - {filename}")


if __name__ == "__main__":
    main()
