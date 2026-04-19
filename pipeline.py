#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import math
import os
import random
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

ROOT = Path(__file__).resolve().parent

def load_cfg():
    with open(ROOT / "settings.json", "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_xyz(path: Path, atoms: List[Tuple[str, float, float, float]], comment: str = ""):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for el, x, y, z in atoms:
            f.write(f"{el:2s}  {x: .10f}  {y: .10f}  {z: .10f}\n")

def read_xyz(path: Path) -> List[Tuple[str, float, float, float]]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip() for ln in f]
    n = int(lines[0].strip())
    atoms = []
    for ln in lines[2:2+n]:
        if not ln.strip():
            continue
        parts = ln.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return atoms

def dist(a, b):
    return math.sqrt((a[1]-b[1])**2 + (a[2]-b[2])**2 + (a[3]-b[3])**2)

def fingerprint_sorted_pair_dists(atoms):
    vals = []
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            vals.append(dist(atoms[i], atoms[j]))
    vals.sort()
    return vals

def fp_rms(fp1, fp2):
    if len(fp1) != len(fp2):
        raise ValueError("Fingerprint length mismatch")
    if not fp1:
        return 0.0
    s = 0.0
    for a, b in zip(fp1, fp2):
        s += (a-b)**2
    return math.sqrt(s / len(fp1))

def centered(atoms):
    n = len(atoms)
    cx = sum(a[1] for a in atoms)/n
    cy = sum(a[2] for a in atoms)/n
    cz = sum(a[3] for a in atoms)/n
    return [(el, x-cx, y-cy, z-cz) for el, x, y, z in atoms]

def ring12(d):
    # circle radius from nearest-neighbor chord length
    R = d / (2.0 * math.sin(math.pi / 12.0))
    atoms = []
    for k in range(12):
        ang = 2.0 * math.pi * k / 12.0
        atoms.append(("B", R*math.cos(ang), R*math.sin(ang), 0.0))
    return centered(atoms)

def double_ring_6_6(d):
    # two stacked hexagons
    R = d
    z = 0.65 * d
    atoms = []
    for k in range(6):
        ang = 2.0 * math.pi * k / 6.0
        atoms.append(("B", R*math.cos(ang), R*math.sin(ang), -z))
    for k in range(6):
        ang = 2.0 * math.pi * (k + 0.5) / 6.0
        atoms.append(("B", 0.92*R*math.cos(ang), 0.92*R*math.sin(ang), z))
    return centered(atoms)

def zigzag_chain12(d):
    atoms = []
    for k in range(12):
        x = k * 0.92 * d
        y = (0.35 * d) if (k % 2 == 0) else (-0.35 * d)
        z = 0.0
        atoms.append(("B", x, y, z))
    return centered(atoms)

def random_compact(n, d, min_sep, radius_factor, rng):
    # random compact packing in sphere
    r = max(2.5, radius_factor * d)
    pts = []
    attempts = 0
    while len(pts) < n:
        attempts += 1
        if attempts > 200000:
            raise RuntimeError("Не удалось сгенерировать случайную компактную геометрию. Уменьшите min_separation_ang.")
        x = rng.uniform(-r, r)
        y = rng.uniform(-r, r)
        z = rng.uniform(-r, r)
        if x*x + y*y + z*z > r*r:
            continue
        ok = True
        for _, px, py, pz in pts:
            dd = math.sqrt((x-px)**2 + (y-py)**2 + (z-pz)**2)
            if dd < min_sep:
                ok = False
                break
        if ok:
            pts.append(("B", x, y, z))
    return centered(pts)

def make_starts():
    cfg = load_cfg()
    outdir = ROOT / "starts"
    ensure_dir(outdir)
    # clear only xyz starts
    for p in outdir.glob("*.xyz"):
        p.unlink()
    rng = random.Random(cfg["random_seed"])
    dists = cfg["seed_distances_ang"]
    n_atoms = cfg["n_atoms"]
    min_sep = cfg["min_separation_ang"]
    radius_factor = cfg["random_pack_radius_factor"]
    for d in dists:
        write_xyz(outdir / f"ring12_d{d:.2f}.xyz", ring12(d), comment=f"ring12 d={d:.2f}")
        write_xyz(outdir / f"double_ring_6_6_d{d:.2f}.xyz", double_ring_6_6(d), comment=f"double_ring_6_6 d={d:.2f}")
        write_xyz(outdir / f"zigzag_chain12_d{d:.2f}.xyz", zigzag_chain12(d), comment=f"zigzag_chain12 d={d:.2f}")
        for i in range(1, cfg["n_random_starts_per_distance"] + 1):
            atoms = random_compact(n_atoms, d, min_sep, radius_factor, rng)
            write_xyz(outdir / f"random_d{d:.2f}_{i:03d}.xyz", atoms, comment=f"random compact d={d:.2f} idx={i}")
    print(f"Сгенерировано стартов: {len(list(outdir.glob('*.xyz')))}")

def orca_input(method: str, nprocs: int, maxcore: int, charge: int, mult: int, xyz_rel: str, do_numfreq=False):
    keywords = f"{method} TightSCF"
    if do_numfreq:
        keywords += " NumFreq"
    else:
        keywords += " TightOpt"
    return f"""! {keywords}

%pal
  nprocs {nprocs}
end

%maxcore {maxcore}

* xyzfile {charge} {mult} {xyz_rel}
"""

def make_xtb_inputs():
    cfg = load_cfg()
    starts = sorted((ROOT / "starts").glob("*.xyz"))
    if not starts:
        raise SystemExit("Сначала запустите: python3 pipeline.py make-starts")
    base = ROOT / "jobs" / "01_xtb"
    ensure_dir(base)
    for mult in cfg["multiplicities"]:
        mdir = base / f"mult_{mult}"
        ensure_dir(mdir)
        for xyz in starts:
            job = mdir / f"{xyz.stem}_m{mult}.inp"
            rel_xyz = os.path.relpath(xyz, start=mdir)
            txt = orca_input(cfg["xtb_method"], cfg["xtb_nprocs"], cfg["xtb_maxcore_mb"], cfg["charge"], mult, rel_xyz, do_numfreq=False)
            job.write_text(txt, encoding="utf-8")
    print("Созданы input-файлы xTB.")

def parse_final_energy(text: str) -> Optional[float]:
    vals = re.findall(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)", text)
    if vals:
        return float(vals[-1])
    return None

def parse_last_cartesian_coords(text: str) -> Optional[List[Tuple[str, float, float, float]]]:
    lines = text.splitlines()
    starts = []
    for i, ln in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in ln:
            starts.append(i)
    if not starts:
        return None
    for start in reversed(starts):
        atoms = []
        i = start + 1
        # skip headers and dashes/blank lines
        while i < len(lines) and (not lines[i].strip() or set(lines[i].strip()) == {"-"}):
            i += 1
        while i < len(lines):
            s = lines[i].strip()
            if not s:
                break
            parts = s.split()
            if len(parts) < 4:
                break
            el = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except Exception:
                break
            atoms.append((el, x, y, z))
            i += 1
        if atoms:
            return atoms
    return None

def job_terminated_normally(text: str) -> bool:
    return "ORCA TERMINATED NORMALLY" in text

def optimization_converged(text: str) -> bool:
    return ("THE OPTIMIZATION HAS CONVERGED" in text) or ("OPTIMIZATION RUN DONE" in text)

def parse_freqs(text: str) -> List[float]:
    freqs = []
    # robust parser for lines like "  1:      -23.45 cm**-1" or similar
    for ln in text.splitlines():
        if "cm" not in ln.lower():
            continue
        m = re.search(r"(-?\d+\.\d+)\s*cm", ln, flags=re.IGNORECASE)
        if m:
            val = float(m.group(1))
            freqs.append(val)
    # fall back: keep only plausible vibrational values
    freqs = [v for v in freqs if -5000.0 < v < 10000.0]
    return freqs

def extract_xyz_from_out(out_path: Path, xyz_out_path: Path) -> Optional[List[Tuple[str, float, float, float]]]:
    text = out_path.read_text(encoding="utf-8", errors="replace")
    atoms = parse_last_cartesian_coords(text)
    if atoms:
        write_xyz(xyz_out_path, atoms, comment=out_path.stem)
    return atoms

def rank_xtb():
    cfg = load_cfg()
    out_root = ROOT / "jobs" / "01_xtb"
    xyz_dir = ROOT / "results" / "01_xtb_xyz"
    ensure_dir(xyz_dir)
    rows = []
    for out in sorted(out_root.rglob("*.out")):
        text = out.read_text(encoding="utf-8", errors="replace")
        energy = parse_final_energy(text)
        atoms = parse_last_cartesian_coords(text)
        mult_match = re.search(r"_m(\d+)$", out.stem)
        mult = int(mult_match.group(1)) if mult_match else None
        converged = optimization_converged(text)
        normal = job_terminated_normally(text)
        xyz_path = ""
        if atoms:
            xyz_file = xyz_dir / f"{out.stem}.xyz"
            write_xyz(xyz_file, atoms, comment=out.stem)
            xyz_path = str(xyz_file.relative_to(ROOT))
        rows.append({
            "job_name": out.stem,
            "mult": mult,
            "energy_hartree": energy,
            "normal_term": normal,
            "opt_converged": converged,
            "xyz_path": xyz_path,
        })
    all_csv = ROOT / "results" / "xtb_ranked_all.csv"
    with open(all_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["job_name","mult","energy_hartree","normal_term","opt_converged","xyz_path"])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (999 if x["energy_hartree"] is None else x["energy_hartree"])):
            w.writerow(r)
    # unique filter
    good = [r for r in rows if r["energy_hartree"] is not None and r["normal_term"] and r["opt_converged"] and r["xyz_path"]]
    good.sort(key=lambda x: x["energy_hartree"])
    accepted = []
    accepted_fps = []
    tol = cfg["fingerprint_rms_tolerance_ang"]
    for r in good:
        atoms = read_xyz(ROOT / r["xyz_path"])
        fp = fingerprint_sorted_pair_dists(atoms)
        is_dup = False
        for afp in accepted_fps:
            if fp_rms(fp, afp) < tol:
                is_dup = True
                break
        if not is_dup:
            accepted.append(r)
            accepted_fps.append(fp)
    uniq_csv = ROOT / "results" / "xtb_ranked_unique.csv"
    with open(uniq_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","job_name","mult","energy_hartree","xyz_path"])
        w.writeheader()
        for i, r in enumerate(accepted, 1):
            w.writerow({
                "rank": i,
                "job_name": r["job_name"],
                "mult": r["mult"],
                "energy_hartree": r["energy_hartree"],
                "xyz_path": r["xyz_path"],
            })
    topn = accepted[:cfg["top_n_xtb_for_dft"]]
    top_csv = ROOT / "results" / "xtb_top_for_dft.csv"
    with open(top_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","job_name","mult","energy_hartree","xyz_path"])
        w.writeheader()
        for i, r in enumerate(topn, 1):
            w.writerow({
                "rank": i,
                "job_name": r["job_name"],
                "mult": r["mult"],
                "energy_hartree": r["energy_hartree"],
                "xyz_path": r["xyz_path"],
            })
    print(f"xTB: всего good={len(good)}, unique={len(accepted)}, top_for_dft={len(topn)}")

def make_dft_inputs():
    cfg = load_cfg()
    top_csv = ROOT / "results" / "xtb_top_for_dft.csv"
    if not top_csv.exists():
        raise SystemExit("Сначала запустите: python3 pipeline.py rank-xtb")
    outdir = ROOT / "jobs" / "02_r2scan"
    ensure_dir(outdir)
    with open(top_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        xyz = ROOT / r["xyz_path"]
        mult = int(r["mult"])
        inp = outdir / f"{r['job_name']}_r2scan.inp"
        rel_xyz = os.path.relpath(xyz, start=outdir)
        txt = orca_input(cfg["dft_method"], cfg["dft_nprocs"], cfg["dft_maxcore_mb"], cfg["charge"], mult, rel_xyz, do_numfreq=False)
        inp.write_text(txt, encoding="utf-8")
    print("Созданы input-файлы DFT.")

def rank_dft_internal() -> List[Dict]:
    xyz_dir = ROOT / "results" / "02_r2scan_xyz"
    ensure_dir(xyz_dir)
    rows = []
    for out in sorted((ROOT / "jobs" / "02_r2scan").glob("*.out")):
        text = out.read_text(encoding="utf-8", errors="replace")
        energy = parse_final_energy(text)
        atoms = parse_last_cartesian_coords(text)
        normal = job_terminated_normally(text)
        converged = optimization_converged(text)
        mult_match = re.search(r"_m(\d+)_r2scan$", out.stem)
        if not mult_match:
            # fallback if original stem survives differently
            mm = re.search(r"_m(\d+)", out.stem)
            mult = int(mm.group(1)) if mm else None
        else:
            mult = int(mult_match.group(1))
        xyz_path = ""
        if atoms:
            xyz_file = xyz_dir / f"{out.stem}.xyz"
            write_xyz(xyz_file, atoms, comment=out.stem)
            xyz_path = str(xyz_file.relative_to(ROOT))
        rows.append({
            "job_name": out.stem,
            "mult": mult,
            "energy_hartree": energy,
            "normal_term": normal,
            "opt_converged": converged,
            "xyz_path": xyz_path,
        })
    good = [r for r in rows if r["energy_hartree"] is not None and r["normal_term"] and r["opt_converged"] and r["xyz_path"]]
    good.sort(key=lambda x: x["energy_hartree"])
    cfg = load_cfg()
    tol = cfg["fingerprint_rms_tolerance_ang"]
    accepted, accepted_fps = [], []
    for r in good:
        atoms = read_xyz(ROOT / r["xyz_path"])
        fp = fingerprint_sorted_pair_dists(atoms)
        is_dup = False
        for afp in accepted_fps:
            if fp_rms(fp, afp) < tol:
                is_dup = True
                break
        if not is_dup:
            accepted.append(r)
            accepted_fps.append(fp)
    rank_csv = ROOT / "results" / "final_dft_ranked.csv"
    with open(rank_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank","job_name","mult","energy_hartree","xyz_path"])
        w.writeheader()
        for i, r in enumerate(accepted, 1):
            w.writerow({
                "rank": i,
                "job_name": r["job_name"],
                "mult": r["mult"],
                "energy_hartree": r["energy_hartree"],
                "xyz_path": r["xyz_path"],
            })
    return accepted

def make_numfreq_inputs():
    cfg = load_cfg()
    ranked = rank_dft_internal()
    topn = ranked[:cfg["top_n_dft_for_numfreq"]]
    outdir = ROOT / "jobs" / "03_numfreq"
    ensure_dir(outdir)
    for r in topn:
        xyz = ROOT / r["xyz_path"]
        mult = int(r["mult"])
        inp = outdir / f"{r['job_name']}_numfreq.inp"
        rel_xyz = os.path.relpath(xyz, start=outdir)
        txt = orca_input(cfg["dft_method"], cfg["dft_nprocs"], cfg["dft_maxcore_mb"], cfg["charge"], mult, rel_xyz, do_numfreq=True)
        inp.write_text(txt, encoding="utf-8")
    print(f"Созданы input-файлы NumFreq для {len(topn)} лучших DFT-структур.")

def final_report():
    cfg = load_cfg()
    ranked = rank_dft_internal()
    rank_map = {r["job_name"]: r for r in ranked}
    rows = []
    for out in sorted((ROOT / "jobs" / "03_numfreq").glob("*.out")):
        text = out.read_text(encoding="utf-8", errors="replace")
        normal = job_terminated_normally(text)
        energy = parse_final_energy(text)
        freqs = parse_freqs(text)
        strict_imag = sum(1 for v in freqs if v < 0.0)
        signif_imag = sum(1 for v in freqs if v < cfg["imag_threshold_significant_cm1"])
        min_freq = min(freqs) if freqs else None
        base_name = out.stem.replace("_numfreq", "")
        dft_row = rank_map.get(base_name)
        xyz_path = dft_row["xyz_path"] if dft_row else ""
        mult = dft_row["mult"] if dft_row else ""
        dft_energy = dft_row["energy_hartree"] if dft_row else None
        status = "NO_FREQS_PARSED"
        if freqs:
            if strict_imag == 0:
                status = "OK_NO_IMAG"
            elif strict_imag > 0 and signif_imag == 0:
                status = "TINY_NEG_ONLY"
            else:
                status = "REJECT_IMAG"
        rows.append({
            "job_name": base_name,
            "mult": mult,
            "dft_energy_hartree": dft_energy,
            "numfreq_energy_hartree": energy,
            "normal_term": normal,
            "n_freqs_parsed": len(freqs),
            "min_freq_cm1": min_freq,
            "n_imag_strict": strict_imag,
            "n_imag_significant": signif_imag,
            "status": status,
            "xyz_path": xyz_path,
        })
    rep = ROOT / "results" / "final_numfreq_report.csv"
    with open(rep, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "job_name","mult","dft_energy_hartree","numfreq_energy_hartree","normal_term",
            "n_freqs_parsed","min_freq_cm1","n_imag_strict","n_imag_significant","status","xyz_path"
        ])
        w.writeheader()
        for r in sorted(rows, key=lambda x: (999 if x["dft_energy_hartree"] is None else x["dft_energy_hartree"])):
            w.writerow(r)
    good = [r for r in rows if r["status"] == "OK_NO_IMAG"]
    good.sort(key=lambda x: x["dft_energy_hartree"])
    good_csv = ROOT / "results" / "final_good_no_imag.csv"
    with open(good_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank","job_name","mult","dft_energy_hartree","min_freq_cm1","status","xyz_path"
        ])
        w.writeheader()
        for i, r in enumerate(good, 1):
            w.writerow({
                "rank": i,
                "job_name": r["job_name"],
                "mult": r["mult"],
                "dft_energy_hartree": r["dft_energy_hartree"],
                "min_freq_cm1": r["min_freq_cm1"],
                "status": r["status"],
                "xyz_path": r["xyz_path"],
            })
    print(f"Итоговый отчёт готов. Структур без мнимых частот: {len(good)}")

def main():
    ap = argparse.ArgumentParser(description="B12 ORCA pipeline for VPS")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make-starts")
    sub.add_parser("make-xtb-inputs")
    sub.add_parser("rank-xtb")
    sub.add_parser("make-dft-inputs")
    sub.add_parser("make-numfreq-inputs")
    sub.add_parser("final-report")
    args = ap.parse_args()

    if args.cmd == "make-starts":
        make_starts()
    elif args.cmd == "make-xtb-inputs":
        make_xtb_inputs()
    elif args.cmd == "rank-xtb":
        rank_xtb()
    elif args.cmd == "make-dft-inputs":
        make_dft_inputs()
    elif args.cmd == "make-numfreq-inputs":
        make_numfreq_inputs()
    elif args.cmd == "final-report":
        final_report()

if __name__ == "__main__":
    main()
