# Dispersion-Style Charts

This folder contains reusable publication-style layouts for phonon-dispersion-like figures.

Run:

```bash
python dispersion_style_charts/generate_figures.py
```

Generated outputs:

```text
dispersion_style_charts/out/
```

Generated example inputs:

```text
dispersion_style_charts/examples/
```

Main files:

- `triptych_layout.json` + `triptych_data.csv`
- `six_panel_layout.json` + `six_panel_data.csv`
- `template_layout.json` + `template_data.csv`

Input CSV columns:

- `panel_id`
- `kind` (`line` or `scatter`)
- `series_id`
- `q`
- `energy_mev`
- `line_color`
- `line_style`
- `line_width`
- `alpha`
- `zorder`
- `marker`
- `marker_size`
- `marker_face`
- `marker_edge`
- `edge_width`

The JSON file defines the panel grid, limits, ticks, captions, panel labels, and optional vertical guide lines.
