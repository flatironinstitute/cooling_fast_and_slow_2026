#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample up to N frames per state from the 230 K replica, one DCD per state.

This is rip-frames-repex.py from the analysis tree, repathed onto the
repository layout.  The sampling logic is unchanged; the only addition is an
optional --seed, because the original used an unseeded default_rng and so did
not reproduce frame-for-frame between runs.  Leaving --seed unset preserves the
original behaviour exactly.

Inputs (both indexed in directory_structure.txt):
    input/state-model/t250_k50/repex-dtrajs/230k-dtraj.h5
    input/md/equilibrium-xtc/trp_whole_dir0.xtc
Output:
    derived/trex-230-by-state/NN.dcd
"""

import argparse
import numpy as np
from pathlib import Path

import paths

N_STATES = 50
N_SAMPLES = 100          # ceiling on frames drawn per state
DTRAJ = paths.REPEX_DTRAJS / '230k-dtraj.h5'
TRAJECTORY = paths.EQUILIBRIUM_XTC / 'trp_whole_dir0.xtc'
OUT_DIR = paths.TREX_230_BY_STATE


def load_dtraj(path=DTRAJ):
    """State assignment per frame.  Single-dataset h5, as written by enspara."""
    import h5py
    with h5py.File(path, 'r') as fh:
        keys = list(fh.keys())
        if keys != ['arr_00']:
            raise ValueError(f'{path.name} holds {keys}, expected ["arr_00"]')
        return fh['arr_00'][:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-samples', type=int, default=N_SAMPLES)
    ap.add_argument('--seed', type=int, default=None,
                    help='seed the sampler so the draw is reproducible; '
                         'omit for the original unseeded behaviour')
    args = ap.parse_args()

    import loos
    from loos import pyloos as pl

    for required in (DTRAJ, TRAJECTORY, paths.TOPOLOGY):
        if not required.is_file():
            raise FileNotFoundError(
                f'{required} is missing; stage the inputs listed in '
                'directory_structure.txt first')

    assigns = load_dtraj()
    counts = np.bincount(assigns, minlength=N_STATES)
    pad = int(np.ceil(np.log10(N_STATES)))
    model = loos.createSystem(str(paths.TOPOLOGY))
    traj = pl.Trajectory(str(TRAJECTORY), model)
    rng = np.random.default_rng(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'{len(assigns)} frames, {int((counts > 0).sum())} of {N_STATES} '
          f'states occupied; seed={args.seed}')
    written = 0
    for state, count in enumerate(counts):
        if count == 0:
            continue
        out_path = OUT_DIR / f'{state:>0{pad}}.dcd'
        writer = loos.DCDWriter(str(out_path))
        candidates = np.where(assigns == state)[0]
        to_pick = min(args.n_samples, int(count))
        for frame_index in rng.choice(candidates, size=to_pick, replace=False):
            traj.readFrame(int(frame_index))
            writer.writeFrame(model)
        written += to_pick
    print(f'wrote {written} frames across {int((counts > 0).sum())} DCDs '
          f'-> {OUT_DIR}')


if __name__ == '__main__':
    main()
