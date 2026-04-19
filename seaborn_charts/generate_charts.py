from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
STATUS_COLORS = {
    "Converged minimum": "#0F766E",
    "Normal termination only": "#D97706",
    "Abnormal termination": "#B91C1C",
}
FINAL_STATUS_COLORS = {
    "OK_NO_IMAG": "#0F766E",
    "TINY_NEG_ONLY": "#D97706",
    "REJECT_IMAG": "#B91C1C",
}

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "legend.title_fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    },
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


def finish_figure(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_visible_bars(ax: plt.Axes) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height):
            continue
        ax.annotate(
            f"{float(height):.2f}",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=10,
            xytext=(0, 5),
            textcoords="offset points",
        )


def build_xtb_landscape(xtb_all: pd.DataFrame) -> None:
    plot_df = xtb_all[xtb_all["is_known_motif"]].copy()
    plot_df["screening_status"] = "Abnormal termination"
    plot_df.loc[plot_df["normal_term"], "screening_status"] = "Normal termination only"
    plot_df.loc[plot_df["normal_term"] & plot_df["opt_converged"], "screening_status"] = "Converged minimum"

    grid = sns.relplot(
        data=plot_df.sort_values(["mult_label", "motif_label", "seed_distance_ang", "rel_energy_kj_mol"]),
        x="seed_distance_ang",
        y="rel_energy_kj_mol",
        hue="motif_label",
        style="screening_status",
        col="mult_label",
        col_order=["M=1", "M=3"],
        palette=MOTIF_COLORS,
        s=120,
        alpha=0.88,
        edgecolor="white",
        linewidth=0.6,
        height=5.5,
        aspect=1.15,
    )
    grid.set_axis_labels("Seed distance (A)", "Relative energy (kJ/mol)")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.set_xticks([1.5, 1.7, 1.9, 2.1])
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    grid.fig.subplots_adjust(top=0.82)
    grid.fig.suptitle("xTB energy landscape by seed distance")
    finish_figure(grid.fig, "01_xtb_landscape.png")


def build_xtb_unique(xtb_unique: pd.DataFrame) -> None:
    plot_df = xtb_unique.sort_values("rank").copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="display_name",
        y="rel_energy_kj_mol",
        hue="motif_label",
        palette=MOTIF_COLORS,
        dodge=False,
        ax=ax,
    )
    ax.set_title("Unique xTB minima selected for DFT")
    ax.set_xlabel("Structure")
    ax.set_ylabel("Relative energy (kJ/mol)")
    ymax = max(float(plot_df["rel_energy_kj_mol"].max()), 1.0)
    ax.set_ylim(0, ymax * 1.16 + 8)
    ax.tick_params(axis="x", rotation=20)
    annotate_visible_bars(ax)
    if ax.legend_:
        ax.legend(title="Motif", frameon=True)
    finish_figure(fig, "02_xtb_unique_minima.png")


def build_pipeline_summary(
    seed_count: int,
    planned_xtb_count: int,
    xtb_all: pd.DataFrame,
    xtb_unique: pd.DataFrame,
    final_numfreq: pd.DataFrame,
    final_good: pd.DataFrame,
) -> None:
    steps = pd.DataFrame(
        [
            ("Seed geometries", seed_count),
            ("Planned xTB jobs", planned_xtb_count),
            ("xTB normal term", int(xtb_all["normal_term"].sum())),
            ("xTB opt converged", int(xtb_all["opt_converged"].sum())),
            ("Unique xTB minima", len(xtb_unique)),
            ("Final DFT/NumFreq set", len(final_numfreq)),
            ("No imaginary freqs", len(final_good)),
        ],
        columns=["step", "count"],
    )
    steps["percent_initial"] = steps["count"] / steps["count"].iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=(11, 6.4))
    sns.barplot(data=steps, y="step", x="count", palette="crest", ax=ax)
    ax.set_title("Pipeline summary from seeds to validated minima")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    xmax = max(steps["count"]) * 1.18
    ax.set_xlim(0, xmax)
    for patch, (_, row) in zip(ax.patches, steps.iterrows()):
        ax.annotate(
            f"{int(row['count'])}  ({row['percent_initial']:.1f}%)",
            (patch.get_width(), patch.get_y() + patch.get_height() / 2),
            va="center",
            ha="left",
            fontsize=10,
            xytext=(8, 0),
            textcoords="offset points",
        )
    finish_figure(fig, "03_pipeline_summary.png")


def build_final_dft(final_numfreq: pd.DataFrame) -> None:
    plot_df = final_numfreq.sort_values("numfreq_energy_hartree").copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=plot_df,
        x="display_name",
        y="rel_energy_kj_mol",
        hue="status",
        palette=FINAL_STATUS_COLORS,
        dodge=False,
        ax=ax,
    )
    ax.set_title("Final DFT + NumFreq ranking")
    ax.set_xlabel("Structure")
    ax.set_ylabel("Relative energy (kJ/mol)")
    ymax = max(float(plot_df["rel_energy_kj_mol"].max()), 1.0)
    ax.set_ylim(0, ymax * 1.16 + 8)
    ax.tick_params(axis="x", rotation=20)
    annotate_visible_bars(ax)
    if ax.legend_:
        ax.legend(title="Final status", frameon=True)
    finish_figure(fig, "04_final_dft_numfreq.png")


def build_frequency_diagnostics(final_numfreq: pd.DataFrame) -> None:
    plot_df = final_numfreq.copy()
    plot_df["bubble_size"] = plot_df["n_imag_strict"].fillna(0).astype(float) * 220 + 180
    fig, ax = plt.subplots(figsize=(11, 6.6))
    sns.scatterplot(
        data=plot_df,
        x="rel_energy_kj_mol",
        y="min_freq_cm1",
        hue="status",
        style="mult_label",
        size="bubble_size",
        sizes=(180, 1500),
        palette=FINAL_STATUS_COLORS,
        ax=ax,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.axhline(0.0, linestyle="--", color="#0F766E", linewidth=1.4)
    ax.axhline(-20.0, linestyle=":", color="#B91C1C", linewidth=1.4)
    ax.set_title("Frequency diagnostics for final structures")
    ax.set_xlabel("Relative energy (kJ/mol)")
    ax.set_ylabel("Minimum frequency (cm^-1)")
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["display_name"],
            (row["rel_energy_kj_mol"], row["min_freq_cm1"]),
            fontsize=10,
            xytext=(8, 8),
            textcoords="offset points",
        )
    if ax.legend_:
        ax.legend(loc="upper right", frameon=True, title="Status / mult / size")
    finish_figure(fig, "05_frequency_diagnostics.png")


def build_status_heatmap(xtb_all: pd.DataFrame) -> None:
    plot_df = xtb_all[xtb_all["is_known_motif"]].copy()
    plot_df["screening_status"] = "Abnormal termination"
    plot_df.loc[plot_df["normal_term"], "screening_status"] = "Normal termination only"
    plot_df.loc[plot_df["normal_term"] & plot_df["opt_converged"], "screening_status"] = "Converged minimum"
    counts = (
        plot_df.groupby(["motif_label", "screening_status"])
        .size()
        .reset_index(name="count")
        .pivot(index="motif_label", columns="screening_status", values="count")
        .fillna(0)
        .reindex(
            index=["Double ring 6+6", "Ring12", "Zigzag chain12", "Random"],
            columns=["Converged minimum", "Normal termination only", "Abnormal termination"],
            fill_value=0,
        )
    )

    fig, ax = plt.subplots(figsize=(9, 5.6))
    sns.heatmap(counts, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5, cbar=True, ax=ax)
    ax.set_title("xTB screening status by motif")
    ax.set_xlabel("Screening outcome")
    ax.set_ylabel("Motif")
    finish_figure(fig, "06_xtb_status_heatmap.png")


def build_index(seed_count: int, planned_xtb_count: int, final_good_count: int) -> None:
    cards = [
        ("01_xtb_landscape.png", "xTB Landscape", "All screened xTB structures split by multiplicity and seed distance."),
        ("02_xtb_unique_minima.png", "Unique xTB Minima", "The unique xTB minima promoted to the DFT stage."),
        ("03_pipeline_summary.png", "Pipeline Summary", "Compression of the workflow from seeds to validated minima."),
        ("04_final_dft_numfreq.png", "Final Ranking", "Final relative energies after DFT + NumFreq."),
        ("05_frequency_diagnostics.png", "Frequency Diagnostics", "Minimum frequency versus relative energy for final structures."),
        ("06_xtb_status_heatmap.png", "xTB Status Heatmap", "How each motif family behaved during xTB screening."),
    ]
    metrics_html = "".join(
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
                f"<h2>{title}</h2>"
                f"<p>{description}</p>"
                f"<a href='{filename}' target='_blank' rel='noopener'>Open image</a>"
                f"<img src='{filename}' alt='{title}'>"
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
  <title>Seaborn charts for the ORCA B12 workflow</title>
  <style>
    :root {{
      --bg: #f7f7f5;
      --panel: #fffdf8;
      --text: #18181b;
      --muted: #52525b;
      --line: #d6d3d1;
      --accent: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(180, 83, 9, 0.12), transparent 24%),
        var(--bg);
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 30px 18px 42px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .intro {{
      max-width: 860px;
      color: var(--muted);
      margin-bottom: 20px;
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
      box-shadow: 0 14px 34px rgba(24, 24, 27, 0.06);
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
      box-shadow: 0 16px 40px rgba(24, 24, 27, 0.08);
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .card p {{
      margin: 0 0 10px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .card a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .card img {{
      width: 100%;
      margin-top: 12px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #fff;
    }}
    @media (max-width: 820px) {{
      main {{ padding: 24px 14px 32px; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Seaborn charts for the ORCA B12 workflow</h1>
    <p class="intro">
      This folder contains static Seaborn charts rendered from the current CSV outputs in <code>results/</code>.
      Open any PNG directly or use this gallery as the entry point.
    </p>
    <section class="metrics">{metrics_html}</section>
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
    build_pipeline_summary(seed_count, planned_xtb_count, xtb_all, xtb_unique, final_numfreq, final_good)
    build_final_dft(final_numfreq)
    build_frequency_diagnostics(final_numfreq)
    build_status_heatmap(xtb_all)
    build_index(seed_count, planned_xtb_count, len(final_good))

    generated_files = sorted(path.name for path in OUT_DIR.glob("*"))
    print("Generated Seaborn files:")
    for filename in generated_files:
        print(f" - {filename}")


if __name__ == "__main__":
    main()
