#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository-root resolution shared by the figure and analysis scripts.

Every path in this project is expressed relative to the repository root so the
scripts run from any working directory and on any checkout.  `git rev-parse`
is the primary mechanism; the marker-file fallback matters because a reader who
downloads the repository as a Zenodo tarball or a GitHub zip has no `.git`.
"""

from pathlib import Path
import subprocess

ROOT_MARKERS = ("input", "output")


def repo_root(markers=ROOT_MARKERS):
    """Absolute path to the repository root."""
    try:
        top = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if top:
            return Path(top)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    here = Path(__file__).resolve()
    for candidate in (here.parent, *here.parents):
        if all((candidate / m).is_dir() for m in markers):
            return candidate
    raise RuntimeError(
        "cannot locate the repository root: no .git and no directory "
        f"containing all of {markers}"
    )


ROOT = repo_root()

INPUT = ROOT / "input"
DERIVED = ROOT / "derived"
OUTPUT = ROOT / "output"

# ── Indexed inputs (see directory_structure.txt) ─────────────────────────────
TOPOLOGY = INPUT / "md" / "topology" / "built-protein.pdb"
EQUILIBRIUM_XTC = INPUT / "md" / "equilibrium-xtc"
SAMPLED_FRAMES = (INPUT / "state-model" / "t250_k50" / "sampled-frames"
                  / "sampled_frames.xtc")
REPEX_DTRAJS = INPUT / "state-model" / "t250_k50" / "repex-dtrajs"
BMSM_LAG400 = INPUT / "state-model" / "t250_k50" / "bmsm" / "lag400"
COOLING_MANIFESTS = INPUT / "cooling" / "endpoints" / "manifests"
COOLING_ENDPOINT_XTC = INPUT / "cooling" / "endpoints" / "xtc"
COOLING_PAIRS = INPUT / "cooling" / "pairs"

# ── Derived products, tracked in Git (small) ─────────────────────────────────
OBSERVABLES = DERIVED / "observables"
COOLING_BOOTSTRAPS = DERIVED / "cooling_bootstraps_k50_t250_lag_400.npz"
PIPELINE_CONFIG = DERIVED / "pipeline-config.yaml"

# ── Derived products rebuilt from coordinates, not tracked (large) ───────────
COOLED_BY_STATE = DERIVED / "cooled-by-state"
TREX_230_BY_STATE = DERIVED / "trex-230-by-state"
