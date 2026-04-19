from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
EXAMPLES_DIR = ROOT / "examples"
OUT_DIR = ROOT / "out"

sns.set_theme(
    style="white",
    context="paper",
    rc={
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "font.family": "DejaVu Serif",
    },
)


def ensure_dirs() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def face_or_none(value: str) -> str:
    return "none" if str(value).lower() == "none" else value


def write_layout(path: Path, layout: dict) -> None:
    path.write_text(json.dumps(layout, indent=2), encoding="utf-8")


def save_data(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def add_line(
    rows: list[dict],
    panel_id: str,
    series_id: str,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    color: str,
    line_style: str = "-",
    line_width: float = 1.1,
    alpha: float = 1.0,
    zorder: float = 2.0,
) -> None:
    for x, y in zip(xs, ys):
        rows.append(
            {
                "panel_id": panel_id,
                "kind": "line",
                "series_id": series_id,
                "q": float(x),
                "energy_mev": float(y),
                "line_color": color,
                "line_style": line_style,
                "line_width": line_width,
                "alpha": alpha,
                "zorder": zorder,
                "marker": "",
                "marker_size": "",
                "marker_face": "",
                "marker_edge": "",
                "edge_width": "",
            }
        )


def add_scatter(
    rows: list[dict],
    panel_id: str,
    series_id: str,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    marker: str,
    marker_size: float,
    marker_face: str,
    marker_edge: str,
    edge_width: float = 0.0,
    alpha: float = 1.0,
    zorder: float = 4.0,
) -> None:
    for x, y in zip(xs, ys):
        rows.append(
            {
                "panel_id": panel_id,
                "kind": "scatter",
                "series_id": series_id,
                "q": float(x),
                "energy_mev": float(y),
                "line_color": "",
                "line_style": "",
                "line_width": "",
                "alpha": alpha,
                "zorder": zorder,
                "marker": marker,
                "marker_size": marker_size,
                "marker_face": marker_face,
                "marker_edge": marker_edge,
                "edge_width": edge_width,
            }
        )


def build_triptych_demo() -> tuple[Path, Path]:
    layout = {
        "filename": "demo_triptych.png",
        "figsize": [6.6, 3.9],
        "nrows": 1,
        "ncols": 3,
        "bottom": 0.22,
        "top": 0.97,
        "wspace": 0.12,
        "hspace": 0.1,
        "panels": [
            {
                "id": "100",
                "row": 0,
                "col": 0,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 62.0],
                "xticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "yticks": [0, 10, 20, 30, 40, 50, 60],
                "ylabel": "Energy [meV]",
                "texts": [{"x": 0.5, "y": 0.97, "text": "[100]", "ha": "center", "va": "top", "size": 8}],
            },
            {
                "id": "110",
                "row": 0,
                "col": 1,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 62.0],
                "xticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "yticks": [0, 10, 20, 30, 40, 50, 60],
                "xlabel": "q [r.l.u.]",
                "texts": [{"x": 0.5, "y": 0.97, "text": "[110]", "ha": "center", "va": "top", "size": 8}],
            },
            {
                "id": "111",
                "row": 0,
                "col": 2,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 62.0],
                "xticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "yticks": [0, 10, 20, 30, 40, 50, 60],
                "texts": [{"x": 0.5, "y": 0.97, "text": "[111]", "ha": "center", "va": "top", "size": 8}],
            },
        ],
        "figure_texts": [{"x": 0.5, "y": 0.02, "text": "a", "ha": "center", "va": "bottom", "style": "italic", "size": 10}],
    }

    q_line = np.linspace(0.0, 0.5, 32)
    q_pts = np.array([0.03, 0.08, 0.14, 0.20, 0.28, 0.35, 0.44])

    rows: list[dict] = []
    panel_specs = {
        "100": {
            "u_long": lambda q: 24 + 33 / (1 + np.exp(-(q - 0.19) / 0.08)),
            "u_trans": lambda q: 23 + 9 * (q / 0.5) ** 1.3,
            "l_long": lambda q: 1 + 12.5 * (1 - np.exp(-q / 0.08)),
            "l_trans": lambda q: 12.5 * (1 - np.exp(-q / 0.13)) ** 1.3,
        },
        "110": {
            "u_long": lambda q: 22 + 40 / (1 + np.exp(-(q - 0.21) / 0.08)),
            "u_trans": lambda q: 23 + 18 * (1 - np.exp(-((q - 0.05).clip(min=0)) / 0.18)),
            "l_long": lambda q: 0.6 + 12.5 * (1 - np.exp(-q / 0.06)),
            "l_trans": lambda q: 11.5 * (1 - np.exp(-q / 0.07)) ** 0.95,
        },
        "111": {
            "u_long": lambda q: 22 + 44 / (1 + np.exp(-(q - 0.18) / 0.07)),
            "u_trans": lambda q: 22 + 18 * np.sin(np.pi * np.clip(q / 0.45, 0, 1) / 2) ** 1.7 - 6 * np.maximum(q - 0.32, 0),
            "l_long": lambda q: 0.8 + 12.0 * (1 - np.exp(-q / 0.05)),
            "l_trans": lambda q: 11.8 * (1 - np.exp(-q / 0.06)) ** 0.9,
        },
    }

    for panel_id, funcs in panel_specs.items():
        add_line(rows, panel_id, "u_long", q_line, funcs["u_long"](q_line), color="#444444", line_style="-", line_width=1.0)
        add_line(rows, panel_id, "u_trans", q_line, funcs["u_trans"](q_line), color="#444444", line_style="--", line_width=1.0)
        add_line(rows, panel_id, "l_long", q_line, funcs["l_long"](q_line), color="#444444", line_style="-", line_width=1.0)
        add_line(rows, panel_id, "l_trans", q_line, funcs["l_trans"](q_line), color="#444444", line_style="--", line_width=1.0)

        upper_long_pts = funcs["u_long"](q_pts) + np.array([1.4, 0.3, 1.1, -1.0, 1.2, -2.4, -1.4])
        upper_trans_pts = funcs["u_trans"](q_pts) + np.array([0.3, -0.2, 0.5, -1.5, 1.2, 0.8, -1.0])
        lower_long_pts = funcs["l_long"](q_pts) + np.array([0.4, -0.4, 0.2, 0.4, -0.3, 0.1, -0.2])
        lower_trans_pts = funcs["l_trans"](q_pts) + np.array([-0.5, -0.1, 0.6, 0.8, -0.4, 0.3, 0.1])

        add_scatter(
            rows,
            panel_id,
            "exp_u_long",
            q_pts,
            upper_long_pts,
            marker="s",
            marker_size=10,
            marker_face="#ef4444",
            marker_edge="#ef4444",
        )
        add_scatter(
            rows,
            panel_id,
            "exp_u_trans",
            q_pts,
            upper_trans_pts,
            marker="^",
            marker_size=14,
            marker_face="#2563eb",
            marker_edge="#2563eb",
        )
        add_scatter(
            rows,
            panel_id,
            "exp_l_long",
            q_pts,
            lower_long_pts,
            marker="s",
            marker_size=10,
            marker_face="#ef4444",
            marker_edge="#ef4444",
        )
        add_scatter(
            rows,
            panel_id,
            "exp_l_trans",
            q_pts,
            lower_trans_pts,
            marker="^",
            marker_size=14,
            marker_face="#2563eb",
            marker_edge="#2563eb",
        )

    layout_path = EXAMPLES_DIR / "triptych_layout.json"
    data_path = EXAMPLES_DIR / "triptych_data.csv"
    write_layout(layout_path, layout)
    save_data(data_path, rows)
    return layout_path, data_path


def build_six_panel_demo() -> tuple[Path, Path]:
    layout = {
        "filename": "demo_six_panel.png",
        "figsize": [7.5, 8.6],
        "nrows": 2,
        "ncols": 3,
        "wspace": 0.05,
        "hspace": 0.1,
        "panels": [
            {
                "id": "a_001",
                "row": 0,
                "col": 0,
                "xlim": [0.0, 1.0],
                "ylim": [0.0, 46.0],
                "xticks": [0.0, 0.5, 1.0],
                "yticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "ylabel": "Energy (meV)",
                "texts": [
                    {"x": 0.02, "y": 1.02, "text": "G", "ha": "left", "va": "bottom", "size": 8},
                    {"x": 0.98, "y": 1.02, "text": "X", "ha": "right", "va": "bottom", "size": 8},
                    {"x": 0.83, "y": 0.08, "text": "[001]", "ha": "center", "va": "bottom", "size": 8},
                ],
            },
            {
                "id": "a_110",
                "row": 0,
                "col": 1,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 46.0],
                "xticks": [0.0, 0.25, 0.5],
                "yticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "vlines": [0.18],
                "xlabel": "q (r.l.u)",
                "texts": [
                    {"x": 0.02, "y": 1.02, "text": "X", "ha": "left", "va": "bottom", "size": 8},
                    {"x": 0.50, "y": 1.02, "text": "K/U", "ha": "center", "va": "bottom", "size": 8},
                    {"x": 0.98, "y": 1.02, "text": "G", "ha": "right", "va": "bottom", "size": 8},
                    {"x": 0.83, "y": 0.08, "text": "[110]", "ha": "center", "va": "bottom", "size": 8},
                ],
            },
            {
                "id": "a_111",
                "row": 0,
                "col": 2,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 46.0],
                "xticks": [0.0, 0.25, 0.5],
                "yticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "texts": [
                    {"x": 0.02, "y": 1.02, "text": "G", "ha": "left", "va": "bottom", "size": 8},
                    {"x": 0.98, "y": 1.02, "text": "L", "ha": "right", "va": "bottom", "size": 8},
                    {"x": 0.83, "y": 0.08, "text": "[111]", "ha": "center", "va": "bottom", "size": 8},
                ],
            },
            {
                "id": "b_001",
                "row": 1,
                "col": 0,
                "xlim": [0.0, 1.0],
                "ylim": [0.0, 46.0],
                "xticks": [0.0, 0.5, 1.0],
                "yticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "ylabel": "Energy (meV)",
                "texts": [
                    {"x": 0.02, "y": 1.02, "text": "G", "ha": "left", "va": "bottom", "size": 8},
                    {"x": 0.98, "y": 1.02, "text": "X", "ha": "right", "va": "bottom", "size": 8},
                    {"x": 0.83, "y": 0.08, "text": "[001]", "ha": "center", "va": "bottom", "size": 8},
                ],
            },
            {
                "id": "b_110",
                "row": 1,
                "col": 1,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 46.0],
                "xticks": [0.0, 0.25, 0.5],
                "yticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "vlines": [0.18],
                "xlabel": "q (r.l.u)",
                "texts": [
                    {"x": 0.02, "y": 1.02, "text": "X", "ha": "left", "va": "bottom", "size": 8},
                    {"x": 0.50, "y": 1.02, "text": "K/U", "ha": "center", "va": "bottom", "size": 8},
                    {"x": 0.98, "y": 1.02, "text": "G", "ha": "right", "va": "bottom", "size": 8},
                    {"x": 0.83, "y": 0.08, "text": "[110]", "ha": "center", "va": "bottom", "size": 8},
                ],
            },
            {
                "id": "b_111",
                "row": 1,
                "col": 2,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 46.0],
                "xticks": [0.0, 0.25, 0.5],
                "yticks": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45],
                "texts": [
                    {"x": 0.02, "y": 1.02, "text": "G", "ha": "left", "va": "bottom", "size": 8},
                    {"x": 0.98, "y": 1.02, "text": "L", "ha": "right", "va": "bottom", "size": 8},
                    {"x": 0.83, "y": 0.08, "text": "[111]", "ha": "center", "va": "bottom", "size": 8},
                ],
            },
        ],
        "figure_texts": [
            {"x": 0.5, "y": 0.50, "text": "a", "ha": "center", "va": "center", "style": "italic", "size": 10},
            {"x": 0.5, "y": 0.01, "text": "b", "ha": "center", "va": "bottom", "style": "italic", "size": 10},
        ],
    }

    rows: list[dict] = []

    def acoustic_family(u: np.ndarray, scale: float, bend: float) -> np.ndarray:
        return scale * np.sin(np.pi * u / 2) ** bend

    def hill_family(u: np.ndarray, base: float, peak: float, skew: float) -> np.ndarray:
        return base + peak * np.sin(np.pi * u) ** 2 * (1 - skew * u)

    def rising_family(u: np.ndarray, base: float, gain: float, center: float) -> np.ndarray:
        return base + gain / (1 + np.exp(-(u - center) / 0.14))

    six_specs = {
        "a_001": {"xmax": 1.0, "red_shift": 1.0, "blue_shift": -0.6, "panel": "001"},
        "a_110": {"xmax": 0.5, "red_shift": 0.8, "blue_shift": -0.5, "panel": "110"},
        "a_111": {"xmax": 0.5, "red_shift": 0.9, "blue_shift": -0.4, "panel": "111"},
        "b_001": {"xmax": 1.0, "red_shift": 0.2, "blue_shift": -0.2, "panel": "001b"},
        "b_110": {"xmax": 0.5, "red_shift": 0.1, "blue_shift": -0.1, "panel": "110b"},
        "b_111": {"xmax": 0.5, "red_shift": 0.0, "blue_shift": 0.0, "panel": "111b"},
    }

    for panel_id, spec in six_specs.items():
        xmax = spec["xmax"]
        q_line = np.linspace(0.0, xmax, 48)
        u = q_line / xmax

        if spec["panel"] in {"001", "001b"}:
            ref_curves = [
                rising_family(u, 24.0, 18.0, 0.52),
                hill_family(u, 26.0, 12.0, 0.9),
                hill_family(u, 21.0, 8.0, 0.35),
                acoustic_family(u, 15.0, 0.9),
                acoustic_family(u, 14.0, 0.95) + 0.8,
                acoustic_family(u, 13.0, 1.05) + 1.5,
            ]
        elif spec["panel"] in {"110", "110b"}:
            ref_curves = [
                24.0 + 19.0 * np.sin(np.pi * u) ** 2 * (1 - 0.65 * u),
                25.0 + 10.0 * np.sin(np.pi * u) ** 2 * (1 - 0.20 * u),
                21.0 + 7.0 * np.sin(np.pi * u) ** 2 * (1 - 0.35 * u),
                acoustic_family(np.clip(1 - u, 0, 1), 15.5, 0.85),
                acoustic_family(np.clip(1 - u, 0, 1), 14.8, 0.95) + 0.6,
                acoustic_family(np.clip(1 - u, 0, 1), 13.9, 1.05) + 1.2,
            ]
        else:
            ref_curves = [
                rising_family(u, 23.5, 18.5, 0.56),
                hill_family(u, 24.5, 9.5, 0.7),
                hill_family(u, 21.5, 6.0, 0.3),
                acoustic_family(u, 15.0, 0.85) + 0.4,
                acoustic_family(u, 14.5, 0.95) + 0.7,
                acoustic_family(u, 13.7, 1.05) + 1.2,
            ]

        for index, curve in enumerate(ref_curves, start=1):
            add_line(
                rows,
                panel_id,
                f"ref_{index}",
                q_line,
                curve,
                color="#8b8b8b",
                line_style="--",
                line_width=0.8,
                alpha=0.95,
                zorder=1.0,
            )

        red_top = ref_curves[0] + spec["red_shift"]
        blue_top = ref_curves[1] + spec["blue_shift"]
        red_low = ref_curves[3] + 0.25
        blue_low = ref_curves[4] - 0.15

        add_line(rows, panel_id, "red_top", q_line, red_top, color="#ef4444", line_width=1.0, zorder=2.3)
        add_line(rows, panel_id, "blue_top", q_line, blue_top, color="#2563eb", line_width=1.0, zorder=2.3)
        add_line(rows, panel_id, "red_low", q_line, red_low, color="#ef4444", line_width=1.0, zorder=2.3)
        add_line(rows, panel_id, "blue_low", q_line, blue_low, color="#2563eb", line_width=1.0, zorder=2.3)

        q_pts = np.linspace(0.06 * xmax, 0.88 * xmax, 7)
        u_pts = q_pts / xmax
        red_fill_top = np.interp(q_pts, q_line, red_top) + np.array([0.6, -1.0, 1.3, -0.4, 1.0, -1.1, 0.9])
        blue_fill_top = np.interp(q_pts, q_line, blue_top) + np.array([0.3, 0.9, -0.8, 1.2, -1.0, 0.7, -1.1])
        red_fill_low = np.interp(q_pts, q_line, red_low) + np.array([0.2, -0.2, 0.3, -0.1, 0.2, -0.2, 0.1])
        blue_open_low = np.interp(q_pts, q_line, blue_low) + np.array([0.1, -0.3, 0.2, -0.2, 0.1, -0.2, 0.0])

        add_scatter(
            rows,
            panel_id,
            "red_fill_top",
            q_pts,
            red_fill_top,
            marker="o",
            marker_size=14,
            marker_face="#ef4444",
            marker_edge="#ef4444",
        )
        add_scatter(
            rows,
            panel_id,
            "blue_fill_top",
            q_pts,
            blue_fill_top,
            marker="^",
            marker_size=18,
            marker_face="#2563eb",
            marker_edge="#2563eb",
        )
        add_scatter(
            rows,
            panel_id,
            "red_fill_low",
            q_pts,
            red_fill_low,
            marker="o",
            marker_size=14,
            marker_face="#ef4444",
            marker_edge="#ef4444",
        )
        add_scatter(
            rows,
            panel_id,
            "blue_open_low",
            q_pts,
            blue_open_low,
            marker="^",
            marker_size=18,
            marker_face="none",
            marker_edge="#2563eb",
            edge_width=0.8,
        )

        if spec["panel"] in {"001", "110", "111"}:
            pale_pts = 18.5 + 0.8 * np.sin(np.pi * u_pts)
            add_scatter(
                rows,
                panel_id,
                "open_gray",
                q_pts,
                pale_pts,
                marker="o",
                marker_size=8,
                marker_face="none",
                marker_edge="#b0b0b0",
                edge_width=0.6,
                alpha=0.9,
                zorder=1.8,
            )

    layout_path = EXAMPLES_DIR / "six_panel_layout.json"
    data_path = EXAMPLES_DIR / "six_panel_data.csv"
    write_layout(layout_path, layout)
    save_data(data_path, rows)
    return layout_path, data_path


def write_templates() -> None:
    template_layout = {
        "filename": "your_figure.png",
        "figsize": [6.8, 4.0],
        "nrows": 1,
        "ncols": 3,
        "wspace": 0.12,
        "hspace": 0.1,
        "panels": [
            {
                "id": "panel_1",
                "row": 0,
                "col": 0,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 60.0],
                "xticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "yticks": [0, 10, 20, 30, 40, 50, 60],
                "ylabel": "Energy [meV]",
                "texts": [{"x": 0.5, "y": 0.97, "text": "[100]", "ha": "center", "va": "top", "size": 8}],
            },
            {
                "id": "panel_2",
                "row": 0,
                "col": 1,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 60.0],
                "xticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "yticks": [0, 10, 20, 30, 40, 50, 60],
                "xlabel": "q [r.l.u.]",
                "texts": [{"x": 0.5, "y": 0.97, "text": "[110]", "ha": "center", "va": "top", "size": 8}],
            },
            {
                "id": "panel_3",
                "row": 0,
                "col": 2,
                "xlim": [0.0, 0.5],
                "ylim": [0.0, 60.0],
                "xticks": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "yticks": [0, 10, 20, 30, 40, 50, 60],
                "texts": [{"x": 0.5, "y": 0.97, "text": "[111]", "ha": "center", "va": "top", "size": 8}],
            },
        ],
        "figure_texts": [{"x": 0.5, "y": 0.02, "text": "a", "ha": "center", "va": "bottom", "style": "italic", "size": 10}],
    }
    template_rows = [
        {
            "panel_id": "panel_1",
            "kind": "line",
            "series_id": "branch_1",
            "q": 0.00,
            "energy_mev": 2.0,
            "line_color": "#444444",
            "line_style": "-",
            "line_width": 1.0,
            "alpha": 1.0,
            "zorder": 2.0,
            "marker": "",
            "marker_size": "",
            "marker_face": "",
            "marker_edge": "",
            "edge_width": "",
        },
        {
            "panel_id": "panel_1",
            "kind": "scatter",
            "series_id": "exp_1",
            "q": 0.10,
            "energy_mev": 7.0,
            "line_color": "",
            "line_style": "",
            "line_width": "",
            "alpha": 1.0,
            "zorder": 4.0,
            "marker": "s",
            "marker_size": 12,
            "marker_face": "#ef4444",
            "marker_edge": "#ef4444",
            "edge_width": 0.0,
        },
    ]
    write_layout(EXAMPLES_DIR / "template_layout.json", template_layout)
    save_data(EXAMPLES_DIR / "template_data.csv", template_rows)


def render_figure(layout_path: Path, data_path: Path) -> Path:
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    data = pd.read_csv(data_path)
    data["panel_id"] = data["panel_id"].astype(str)
    panels = layout["panels"]

    fig, axes = plt.subplots(
        layout["nrows"],
        layout["ncols"],
        figsize=tuple(layout["figsize"]),
        squeeze=False,
    )
    fig.subplots_adjust(
        wspace=layout.get("wspace", 0.1),
        hspace=layout.get("hspace", 0.1),
        left=layout.get("left", 0.125),
        right=layout.get("right", 0.9),
        bottom=layout.get("bottom", 0.11),
        top=layout.get("top", 0.88),
    )

    for panel in panels:
        ax = axes[panel["row"]][panel["col"]]
        panel_df = data[data["panel_id"] == str(panel["id"])].copy()

        for _, series_df in panel_df.groupby(["kind", "series_id"], sort=False):
            kind = series_df["kind"].iloc[0]
            if kind == "line":
                series_df = series_df.sort_values("q")
                ax.plot(
                    series_df["q"],
                    series_df["energy_mev"],
                    color=series_df["line_color"].iloc[0],
                    linestyle=series_df["line_style"].iloc[0],
                    linewidth=float(series_df["line_width"].iloc[0]),
                    alpha=float(series_df["alpha"].iloc[0]),
                    zorder=float(series_df["zorder"].iloc[0]),
                )
            else:
                ax.scatter(
                    series_df["q"],
                    series_df["energy_mev"],
                    marker=series_df["marker"].iloc[0],
                    s=float(series_df["marker_size"].iloc[0]),
                    facecolors=face_or_none(str(series_df["marker_face"].iloc[0])),
                    edgecolors=series_df["marker_edge"].iloc[0],
                    linewidths=float(series_df["edge_width"].fillna(0.0).iloc[0]),
                    alpha=float(series_df["alpha"].iloc[0]),
                    zorder=float(series_df["zorder"].iloc[0]),
                )

        ax.set_xlim(panel["xlim"])
        ax.set_ylim(panel["ylim"])
        ax.set_xticks(panel["xticks"])
        ax.set_yticks(panel["yticks"])
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=7, length=3)
        ax.minorticks_on()
        ax.tick_params(which="minor", length=1.8)

        if panel.get("ylabel"):
            ax.set_ylabel(panel["ylabel"], fontsize=8)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if panel.get("xlabel"):
            ax.set_xlabel(panel["xlabel"], fontsize=8)
        else:
            ax.set_xlabel("")

        for x_value in panel.get("vlines", []):
            ax.axvline(x_value, color="#bdbdbd", linestyle=":", linewidth=0.7, zorder=0.5)

        for text in panel.get("texts", []):
            ax.text(
                text["x"],
                text["y"],
                text["text"],
                transform=ax.transAxes,
                ha=text.get("ha", "center"),
                va=text.get("va", "center"),
                fontsize=text.get("size", 8),
                style=text.get("style", "normal"),
            )

    for text in layout.get("figure_texts", []):
        fig.text(
            text["x"],
            text["y"],
            text["text"],
            ha=text.get("ha", "center"),
            va=text.get("va", "center"),
            fontsize=text.get("size", 10),
            style=text.get("style", "normal"),
        )

    out_path = OUT_DIR / layout["filename"]
    fig.savefig(out_path, dpi=280, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_index(image_paths: list[Path]) -> None:
    cards_html = "".join(
        (
            "<section class='card'>"
            f"<h2>{path.stem}</h2>"
            f"<a href='{path.name}' target='_blank' rel='noopener'>Open image</a>"
            f"<img src='{path.name}' alt='{path.stem}'>"
            "</section>"
        )
        for path in image_paths
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dispersion-style figures</title>
  <style>
    :root {{
      --bg: #f6f5f2;
      --panel: #fffdf9;
      --text: #161616;
      --muted: #555;
      --line: #d7d4cf;
      --accent: #8b1e3f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(139, 30, 63, 0.12), transparent 22%),
        radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 24%),
        var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 18px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
    }}
    p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.55;
      max-width: 860px;
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
      box-shadow: 0 12px 32px rgba(0, 0, 0, 0.06);
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .card a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .card img {{
      width: 100%;
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Dispersion-style figure demos</h1>
    <p>
      These figures are demo layouts in the style of publication phonon-dispersion plots.
      Replace the CSV values in <code>dispersion_style_charts/examples/</code> with your real q-dependent data.
    </p>
    <section class="grid">{cards_html}</section>
  </main>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    triptych_layout, triptych_data = build_triptych_demo()
    six_layout, six_data = build_six_panel_demo()
    write_templates()

    outputs = [
        render_figure(triptych_layout, triptych_data),
        render_figure(six_layout, six_data),
    ]
    build_index(outputs)

    print("Generated dispersion-style figures:")
    for path in outputs:
        print(f" - {path.name}")
    print(" - index.html")
    print("Example inputs:")
    for name in [
        "triptych_layout.json",
        "triptych_data.csv",
        "six_panel_layout.json",
        "six_panel_data.csv",
        "template_layout.json",
        "template_data.csv",
    ]:
        print(f" - {name}")


if __name__ == "__main__":
    main()
