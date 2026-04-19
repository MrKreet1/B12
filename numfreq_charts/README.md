# NumFreq Charts

This folder contains charts built directly from the current ORCA `NumFreq` outputs.

Run:

```bash
python numfreq_charts/generate_charts.py
```

Outputs:

```text
numfreq_charts/out/
```

Main entry:

```text
numfreq_charts/out/index.html
```

Source inputs:

- `jobs/03_numfreq/*.out`
- `results/final_numfreq_report.csv`

Generated table:

- `numfreq_charts/out/parsed_numfreq_modes.csv`
