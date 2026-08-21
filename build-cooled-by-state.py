#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge the cooling endpoint frames into one DCD per Markov state.

Grouping follows grab-frames-coolings.py from the analysis tree: manifest line
i belongs to state pairs[i, 1], the end state of cooling run i, and frames are
appended in manifest order.  That reproduces the 6.4e9 and 6.4e8 per-state sets
bit-for-bit (verified to 5e-7 nm) and 49 of 50 states for 6.4e7.

Inputs (indexed in directory_structure.txt):
    input/cooling/endpoints/manifests/end-{rate}-traj-list.txt
    input/cooling/endpoints/xtc/{rate}/...
    input/cooling/pairs/{rate}_pairs_t250_k50.npy
Output:
    derived/cooled-by-state/{rate}/NN.dcd
"""

import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import mdtraj as md

import paths

RATES = ['6e9', '6e8', '6e7', '6e6']
TICA_LAG, K_CHOICE, N_STATES = 250, 50, 50
N_PROCS = min(16, cpu_count())    # the endpoint reads are IO-bound


def resolve(entry, rate, endpoint_root=paths.COOLING_ENDPOINT_XTC):
    """Locate one endpoint file named by a manifest line.

    Manifests as generated carry absolute paths to the original scratch
    locations; directory_structure.txt asks that the staged copies be rewritten
    to point into input/cooling/endpoints/xtc/{rate}/.  Accept either.

    NOTE: basenames are NOT unique within a rate -- the manifests draw from up
    to nine source directories and 163 basenames repeat for 6.4e7, 127 for
    6.4e6.  A flat staging directory therefore cannot be addressed by basename
    alone, which is why the rewritten manifest must carry whatever
    disambiguating prefix the staging used.
    """
    candidate = Path(entry)
    if candidate.is_file():
        return candidate
    relative = endpoint_root / rate / candidate.name
    if relative.is_file():
        return relative
    raise FileNotFoundError(
        f'cannot resolve manifest entry {entry!r} for rate {rate}: not present '
        f'at that path, and no {candidate.name} under {endpoint_root / rate}. '
        'If the staged manifest was rewritten, check that it disambiguates '
        'repeated basenames.')


def _load(args):
    path, topology = args
    return md.load(str(path), top=topology)


def build_rate(rate, topology, n_states=N_STATES, out_root=paths.COOLED_BY_STATE,
               n_procs=N_PROCS):
    manifest_path = paths.COOLING_MANIFESTS / f'end-{rate}-traj-list.txt'
    pairs_path = paths.COOLING_PAIRS / f'{rate}_pairs_t{TICA_LAG}_k{K_CHOICE}.npy'
    entries = manifest_path.read_text().split()
    pairs = np.load(pairs_path)
    if len(entries) != pairs.shape[0]:
        raise ValueError(
            f'{rate}: manifest has {len(entries)} entries but pairs has '
            f'{pairs.shape[0]} rows; they must correspond line-for-row')
    end_state = pairs[:, 1]
    files = [resolve(e, rate) for e in entries]

    out_dir = out_root / rate
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for state in range(n_states):
        idx = np.flatnonzero(end_state == state)
        if idx.size == 0:
            continue
        with Pool(n_procs) as pool:
            frames = pool.map(_load, [(files[i], str(topology)) for i in idx])
        md.join(frames).save(str(out_dir / f'{state:02d}.dcd'))
        written += idx.size
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rates', nargs='+', default=RATES)
    args = ap.parse_args()
    if not paths.TOPOLOGY.is_file():
        raise FileNotFoundError(f'{paths.TOPOLOGY} is missing')
    for rate in args.rates:
        n = build_rate(rate, paths.TOPOLOGY)
        print(f'  {rate}: {n} frames -> {paths.COOLED_BY_STATE / rate}', flush=True)


if __name__ == '__main__':
    main()
