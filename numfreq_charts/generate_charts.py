from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
NUMFREQ_DIR = ROOT / "jobs" / "03_numfreq"
RESULTS_DIR = ROOT / "results"
THIS_DIR = Path(__file__).resolve().parent
OUT_DIR = THIS_DIR / "out"

MOTIF_LABELS = {
    "double_ring_6_6": "Double ring 6+6",
    "ring12": "Ring12",
    "zigzag_chain12": "Zigzag chain12",
    "random": "Random",
}
STRUCTURE_COLORS = ["#0F766E", "#B45309", "#1D4ED8", "#7C3AED", "#BE185D", "#0EA5E9"]
MODE_CLASS_COLORS = {
    "imaginary": "#B91C1C",
    "zero": "#9CA3AF",
    "real": "#2563EB",
}
STATUS_COLORS = {
    "OK_NO_IMAG": "#0F766E",
    "TINY_NEG_ONLY": "#D97706",
    "REJECT_IMAG": "#B91C1C",
    "NO_FREQS_PARSED": "#6B7280",
}

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
    },
)


def parse_job_name(job_name: str) -> dict[str, object]:
    text = str(job_name)
    motif = "other"
    for candidate in MOTIF_LABELS:
        if text.startswith(candidate):
            motif = candidate
            break
    distance_match = re.search(r"_d(\d+\.\d+)", text)
    seed_distance = float(distance_match.group(1)) if distance_match else None
    mult_match = re.search(r"_m(\d+)", text)
    mult = int(mult_match.group(1)) if mult_match else None
    base_label = MOTIF_LABELS.get(motif, text.replace("_r2scan", ""))
    short_label = base_label if mult is None else f"{base_label} (M={mult})"
    return {
        "motif": motif,
        "motif_label": MOTIF_LABELS.get(motif, text),
        "seed_distance_ang": seed_distance,
        "mult_from_name": mult,
        "short_label": short_label,
    }


def normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )


def parse_vibrational_frequencies(text: str) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    in_section = False
    for line in text.splitlines():
        if "VIBRATIONAL FREQUENCIES" in line:
            in_section = True
            continue
        if in_section and "NORMAL MODES" in line:
            break
        if not in_section:
            continue
        match = re.match(r"\s*(\d+):\s*(-?\d+\.\d+)\s*cm\*\*-1", line)
        if match:
            rows.append((int(match.group(1)), float(match.group(2))))
    return rows


def build_mode_table() -> pd.DataFrame:
    metadata = pd.read_csv(RESULTS_DIR / "final_numfreq_report.csv")
    metadata["normal_term"] = normalize_bool(metadata["normal_term"])
    metadata["sort_energy"] = pd.to_numeric(metadata["numfreq_energy_hartree"], errors="coerce")
    metadata = metadata.sort_values(["sort_energy", "job_name"], na_position="last").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for order, meta in enumerate(metadata.itertuples(index=False), start=1):
        out_path = NUMFREQ_DIR / f"{meta.job_name}_numfreq.out"
        if not out_path.exists():
            continue
        parsed = parse_job_name(meta.job_name)
        freqs = parse_vibrational_frequencies(out_path.read_text(encoding="utf-8", errors="replace"))
        for mode_index, frequency_cm1 in freqs:
            if frequency_cm1 < 0.0:
                mode_class = "imaginary"
            elif abs(frequency_cm1) <= 1.0:
                mode_class = "zero"
            else:
                mode_class = "real"
            rows.append(
                {
                    "structure_order": order,
                    "job_name": meta.job_name,
                    "short_label": parsed["short_label"],
                    "motif_label": parsed["motif_label"],
                    "seed_distance_ang": parsed["seed_distance_ang"],
                    "mult": int(meta.mult) if pd.notna(meta.mult) else parsed["mult_from_name"],
                    "status": meta.status,
                    "numfreq_energy_hartree": meta.numfreq_energy_hartree,
                    "min_freq_cm1": meta.min_freq_cm1,
                    "n_imag_strict": int(meta.n_imag_strict),
                    "mode_index": mode_index,
                    "mode_number": mode_index + 1,
                    "frequency_cm1": frequency_cm1,
                    "mode_class": mode_class,
                    "normal_term": bool(meta.normal_term),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No frequencies were parsed from jobs/03_numfreq/*.out")
    palette = {label: STRUCTURE_COLORS[idx % len(STRUCTURE_COLORS)] for idx, label in enumerate(df["short_label"].drop_duplicates())}
    df["structure_color"] = df["short_label"].map(palette)
    return df


def save_fig(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def mode_limits(df: pd.DataFrame) -> tuple[float, float]:
    ymin = float(df["frequency_cm1"].min())
    ymax = float(df["frequency_cm1"].max())
    lower = ymin - max(25.0, abs(ymin) * 0.08)
    upper = ymax + max(35.0, abs(ymax) * 0.05)
    return lower, upper


def build_overlay_plot(df: pd.DataFrame) -> None:
    ordered_labels = df.sort_values("structure_order")["short_label"].drop_duplicates().tolist()
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ymin, ymax = mode_limits(df)
    if ymin < 0:
        ax.axhspan(ymin, 0, color="#fee2e2", alpha=0.45, zorder=0)
    ax.axhline(0, color="#374151", linestyle="--", linewidth=1.0, zorder=1)

    for label in ordered_labels:
        part = df[df["short_label"] == label].sort_values("mode_index")
        color = part["structure_color"].iloc[0]
        ax.plot(part["mode_number"], part["frequency_cm1"], color=color, linewidth=1.8, alpha=0.9, zorder=2)
        regular = part[part["mode_class"] != "imaginary"]
        imag = part[part["mode_class"] == "imaginary"]
        ax.scatter(
            regular["mode_number"],
            regular["frequency_cm1"],
            s=28,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        if not imag.empty:
            ax.scatter(
                imag["mode_number"],
                imag["frequency_cm1"],
                s=84,
                marker="X",
                color=color,
                edgecolor="#7f1d1d",
                linewidth=0.8,
                zorder=4,
            )

    ax.set_title("NumFreq frequencies by mode number")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("Frequency (cm$^{-1}$)")
    ax.set_xlim(1, int(df["mode_number"].max()))
    ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", linestyle="--", alpha=0.28)
    ax.grid(False, axis="x")

    structure_handles = [
        Line2D([0], [0], color=df[df["short_label"] == label]["structure_color"].iloc[0], linewidth=2.2, label=label)
        for label in ordered_labels
    ]
    mode_handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor="#374151", markeredgecolor="white", markersize=7, linewidth=0, label="Real/zero mode"),
        Line2D([0], [0], marker="X", color="white", markerfacecolor="#374151", markeredgecolor="#7f1d1d", markersize=8, linewidth=0, label="Imaginary mode"),
    ]
    legend1 = ax.legend(handles=structure_handles, title="Structure", loc="upper left", frameon=True)
    ax.add_artist(legend1)
    ax.legend(handles=mode_handles, title="Marker meaning", loc="lower right", frameon=True)
    save_fig(fig, "01_mode_number_overlay.png")


def build_structure_panels(df: pd.DataFrame) -> None:
    ordered = list(df.sort_values("structure_order")["short_label"].drop_duplicates())
    fig, axes = plt.subplots(1, len(ordered), figsize=(5.0 * len(ordered), 5.6), sharey=True)
    if len(ordered) == 1:
        axes = [axes]
    ymin, ymax = mode_limits(df)

    for ax, label in zip(axes, ordered):
        part = df[df["short_label"] == label].sort_values("mode_number")
        if ymin < 0:
            ax.axhspan(ymin, 0, color="#fee2e2", alpha=0.45, zorder=0)
        ax.axhline(0, color="#374151", linestyle="--", linewidth=1.0, zorder=1)
        ax.plot(part["mode_number"], part["frequency_cm1"], color="#6b7280", linewidth=1.1, alpha=0.7, zorder=2)
        for mode_class, color in MODE_CLASS_COLORS.items():
            sub = part[part["mode_class"] == mode_class]
            if sub.empty:
                continue
            marker = {"real": "o", "zero": "o", "imaginary": "X"}[mode_class]
            size = {"real": 34, "zero": 34, "imaginary": 80}[mode_class]
            ax.scatter(
                sub["mode_number"],
                sub["frequency_cm1"],
                s=size,
                color=color,
                edgecolor="white" if mode_class != "imaginary" else "#7f1d1d",
                linewidth=0.6,
                marker=marker,
                zorder=3,
            )

        status = part["status"].iloc[0]
        imag_count = int(part["n_imag_strict"].iloc[0])
        ax.set_title(f"{label}\n{status} | imag={imag_count}", fontsize=12)
        ax.set_xlabel("Mode number")
        ax.set_xlim(1, int(part["mode_number"].max()))
        ax.set_ylim(ymin, ymax)
        ax.grid(True, axis="y", linestyle="--", alpha=0.24)
        ax.grid(False, axis="x")
    axes[0].set_ylabel("Frequency (cm$^{-1}$)")

    handles = [
        Line2D([0], [0], marker="o", color="white", markerfacecolor=MODE_CLASS_COLORS["real"], markeredgecolor="white", markersize=7, linewidth=0, label="Real"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor=MODE_CLASS_COLORS["zero"], markeredgecolor="white", markersize=7, linewidth=0, label="Zero / rigid-body"),
        Line2D([0], [0], marker="X", color="white", markerfacecolor=MODE_CLASS_COLORS["imaginary"], markeredgecolor="#7f1d1d", markersize=8, linewidth=0, label="Imaginary"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Mode comparison across final NumFreq structures", y=1.08, fontsize=18)
    save_fig(fig, "02_structure_mode_panels.png")


def build_spectra(df: pd.DataFrame) -> None:
    ordered = list(df.sort_values("structure_order")["short_label"].drop_duplicates())
    fig, axes = plt.subplots(len(ordered), 1, figsize=(12.5, 3.2 * len(ordered)), sharex=True)
    if len(ordered) == 1:
        axes = [axes]
    xmin = float(df["frequency_cm1"].min()) - 80
    xmax = float(df["frequency_cm1"].max()) + 40

    for ax, label in zip(axes, ordered):
        part = df[df["short_label"] == label].sort_values("frequency_cm1")
        ax.axvspan(xmin, 0, color="#fee2e2", alpha=0.55, zorder=0)
        ax.axvline(0, color="#374151", linestyle="--", linewidth=1.0, zorder=1)
        ax.axhline(0, color="#d1d5db", linewidth=0.8, zorder=1)
        for mode_class, color in MODE_CLASS_COLORS.items():
            sub = part[part["mode_class"] == mode_class]
            if sub.empty:
                continue
            ymax_local = 1.0 if mode_class == "real" else 0.9 if mode_class == "zero" else 1.08
            ax.vlines(sub["frequency_cm1"], 0, ymax_local, color=color, linewidth=1.3, alpha=0.95, zorder=2)
            ax.scatter(
                sub["frequency_cm1"],
                [ymax_local] * len(sub),
                s=18 if mode_class != "imaginary" else 40,
                color=color,
                edgecolor="white" if mode_class != "imaginary" else "#7f1d1d",
                linewidth=0.5,
                marker="o" if mode_class != "imaginary" else "X",
                zorder=3,
            )
        meta = part.iloc[0]
        ax.set_title(f"{label} | {meta['status']} | min={meta['min_freq_cm1']:.2f} cm$^{{-1}}$", loc="left", fontsize=12)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([])
        ax.grid(True, axis="x", linestyle="--", alpha=0.24)
        ax.grid(False, axis="y")

    axes[-1].set_xlabel("Frequency (cm$^{-1}$)")
    axes[0].set_xlim(xmin, xmax)
    fig.suptitle("Stick spectrum with imaginary frequencies highlighted", y=1.02, fontsize=18)
    save_fig(fig, "03_frequency_spectra.png")


def build_heatmap(df: pd.DataFrame) -> None:
    ordered = list(df.sort_values("structure_order")["short_label"].drop_duplicates())
    pivot = (
        df.pivot(index="short_label", columns="mode_number", values="frequency_cm1")
        .reindex(ordered)
    )
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=0,
        linewidths=0.25,
        linecolor="#f3f4f6",
        cbar_kws={"label": "Frequency (cm$^{-1}$)"},
        ax=ax,
    )
    ax.set_title("Mode-by-mode comparison heatmap")
    ax.set_xlabel("Mode number")
    ax.set_ylabel("")
    save_fig(fig, "04_mode_heatmap.png")


def build_index(df: pd.DataFrame) -> None:
    summary = (
        df.groupby(["short_label", "status"], as_index=False)
        .agg(
            n_modes=("mode_number", "count"),
            n_imag=("mode_class", lambda x: int((x == "imaginary").sum())),
            min_freq=("frequency_cm1", "min"),
        )
        .sort_values("short_label")
    )
    metrics_html = "".join(
        (
            "<tr>"
            f"<td>{row.short_label}</td>"
            f"<td><span style='color:{STATUS_COLORS.get(row.status, '#374151')};font-weight:700'>{row.status}</span></td>"
            f"<td>{int(row.n_modes)}</td>"
            f"<td>{int(row.n_imag)}</td>"
            f"<td>{row.min_freq:.2f}</td>"
            "</tr>"
        )
        for row in summary.itertuples(index=False)
    )

    cards = [
        ("01_mode_number_overlay.png", "Frequencies by mode number", "Overlay of all parsed modes across the final NumFreq structures."),
        ("02_structure_mode_panels.png", "Per-structure mode comparison", "Each structure in a separate panel, with real, zero, and imaginary modes separated visually."),
        ("03_frequency_spectra.png", "Stick spectra", "Frequency-stick spectra with the negative-frequency region highlighted."),
        ("04_mode_heatmap.png", "Mode heatmap", "Compact comparison of every mode across all final structures."),
    ]
    cards_html = "".join(
        (
            "<section class='card'>"
            f"<h2>{title}</h2>"
            f"<p>{description}</p>"
            f"<a href='{filename}' target='_blank' rel='noopener'>Open image</a>"
            f"<img src='{filename}' alt='{title}'>"
            "</section>"
        )
        for filename, title, description in cards
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NumFreq charts</title>
  <style>
    :root {{
      --bg: #f5f5f4;
      --panel: #fffdfa;
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
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 23%),
        radial-gradient(circle at top right, rgba(185, 28, 28, 0.08), transparent 26%),
        var(--bg);
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 32px;
    }}
    p {{
      margin: 0 0 16px;
      color: var(--muted);
      max-width: 900px;
      line-height: 1.6;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      box-shadow: 0 12px 28px rgba(0,0,0,0.05);
      margin-bottom: 20px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
    }}
    th {{
      background: #fafaf9;
    }}
    .grid {{
      display: grid;
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 12px 32px rgba(0,0,0,0.06);
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 21px;
    }}
    .card p {{
      margin-bottom: 10px;
    }}
    .card a {{
      color: var(--accent);
      font-weight: 700;
      text-decoration: none;
    }}
    .card img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid var(--line);
      margin-top: 12px;
      background: white;
    }}
  </style>
</head>
<body>
  <main>
    <h1>NumFreq mode charts</h1>
    <p>
      These charts are built from the current ORCA <code>NumFreq</code> outputs in <code>jobs/03_numfreq</code>.
      Imaginary modes are emphasized explicitly, so unstable structures are easy to spot.
    </p>
    <table>
      <thead>
        <tr>
          <th>Structure</th>
          <th>Status</th>
          <th>Modes</th>
          <th>Imaginary</th>
          <th>Min freq (cm^-1)</th>
        </tr>
      </thead>
      <tbody>{metrics_html}</tbody>
    </table>
    <section class="grid">{cards_html}</section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_mode_table()
    df.to_csv(OUT_DIR / "parsed_numfreq_modes.csv", index=False)

    build_overlay_plot(df)
    build_structure_panels(df)
    build_spectra(df)
    build_heatmap(df)
    build_index(df)

    generated = sorted(path.name for path in OUT_DIR.glob("*"))
    print("Generated NumFreq chart files:")
    for filename in generated:
        print(f" - {filename}")


if __name__ == "__main__":
    main()
