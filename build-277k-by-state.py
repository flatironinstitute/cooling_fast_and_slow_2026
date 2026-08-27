#!/usr/bin/env python
"""Write the 277 K picked frames per state, with original coordinates and boxes.

sampled_frames.xtc holds the same 50000 frames but centred, rotated onto a
common reference, and with no box, which makes it unusable for anything that
depends on orientation relative to the lattice. This pulls the same frames
(via frame_metadata.txt, a mapping verified exact to xtc precision) straight
from the unbiased trajectories and writes one DCD per state, matching the
layout of the cooled and 230 K sets.

DCD rather than XTC on purpose: the source frames are already xtc-quantised, and
re-compressing would add a second round of loss.
"""
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import mdtraj as md

import paths

# The unbiased trajectories these frames come from are ~61 GB and are NOT part
# of the deposit; the per-state set this script writes is what is deposited.
# Running it therefore needs access to the original scratch data.
TRAJ_LIST = paths.INPUT / 'state-model' / 't250_k50' / 'sampled-frames' / 'traj-list.txt'
METADATA = (paths.INPUT / 'state-model' / 't250_k50' / 'sampled-frames'
            / 'frame_metadata.txt')
TOPOLOGY = paths.TOPOLOGY
OUT_DIR = paths.SAMPLED_277K_BY_STATE
N_STATES = 50
N_PROCS = min(16, cpu_count())

_CTX = {}


def _init(paths, topology):
    _CTX['paths'] = paths
    _CTX['top'] = md.load(str(topology)).top


def _one(job):
    traj_idx, rows, frames = job
    traj = md.load(_CTX['paths'][traj_idx], top=_CTX['top'])
    return [(r, traj.xyz[f].copy(),
             None if traj.unitcell_lengths is None else traj.unitcell_lengths[f].copy(),
             None if traj.unitcell_angles is None else traj.unitcell_angles[f].copy())
            for r, f in zip(rows, frames)]


def main(traj_list=TRAJ_LIST, metadata=METADATA, topology=TOPOLOGY,
         out_dir=OUT_DIR, n_states=N_STATES, n_procs=N_PROCS):
    for required in (traj_list, metadata, topology):
        if not required.is_file():
            raise FileNotFoundError(
                f'{required} is missing. This rebuild needs the original '
                'unbiased trajectories and their manifest, which are upstream '
                'of the deposit; see directory_structure.txt.')
    source_paths = traj_list.read_text().split()
    meta = np.loadtxt(metadata, dtype=int)
    top = md.load(str(topology)).top
    n_frames, n_atoms = meta.shape[0], top.n_atoms
    print(f'{n_frames} frames, {n_atoms} atoms, {n_procs} processes', flush=True)

    jobs = {}
    for row, (_, _, ti, fi) in enumerate(meta):
        jobs.setdefault(int(ti), ([], []))
        jobs[int(ti)][0].append(row)
        jobs[int(ti)][1].append(int(fi))
    jobs = [(ti, r, f) for ti, (r, f) in jobs.items()]

    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    lengths = np.zeros((n_frames, 3), dtype=np.float32)
    angles = np.zeros((n_frames, 3), dtype=np.float32)
    with Pool(n_procs, initializer=_init, initargs=(source_paths, topology)) as pool:
        for n, res in enumerate(pool.imap_unordered(_one, jobs, 64), 1):
            for row, coords, ln, an in res:
                xyz[row] = coords
                lengths[row] = ln if ln is not None else 0.0
                angles[row] = an if an is not None else 90.0
            if n % 3000 == 0:
                print(f'  {n}/{len(jobs)} trajectories', flush=True)

    if not np.all(lengths > 0):
        raise RuntimeError(f'{int((lengths <= 0).any(axis=1).sum())} frames lack a box')
    print(f'box edge: mean {lengths[:,0].mean()*10:.3f} A, '
          f'range {lengths[:,0].min()*10:.3f}-{lengths[:,0].max()*10:.3f}', flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    states = meta[:, 1]
    for s in range(n_states):
        rows = np.flatnonzero(states == s)          # ascending, as picked
        traj = md.Trajectory(xyz[rows], top,
                             unitcell_lengths=lengths[rows],
                             unitcell_angles=angles[rows])
        traj.save(str(out_dir / f'{s:02d}.dcd'))
    print(f'wrote {n_states} per-state DCDs -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
