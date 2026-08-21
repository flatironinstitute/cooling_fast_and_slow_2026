#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the observables from coordinates and compare to the tracked copies.

This is the reproducibility check: it does not overwrite anything, it reports
the largest deviation per product.  npz archives are zip files carrying entry
timestamps, so they are not byte-reproducible and `git diff` cannot serve as
the comparison -- the numbers have to be compared directly.

Requires the coordinate-derived inputs (see `make coordinates`).
"""

import sys
import numpy as np

import paths
import fig5_panels as fig5
import table_s3_pbc_self_image as s3
import fig_s4_trex_populations as s4

TOLERANCE = 1e-4          # A / dimensionless; float32 DCD round-trip is ~1e-6


def compare(name, tracked, recomputed, tolerance=TOLERANCE):
    missing = sorted(set(tracked) ^ set(recomputed))
    worst_key, worst = None, 0.0
    for key in sorted(set(tracked) & set(recomputed)):
        a = np.asarray(tracked[key], dtype=float)
        b = np.asarray(recomputed[key], dtype=float)
        if a.shape != b.shape:
            print(f'  {name}: {key} shape {a.shape} vs {b.shape}')
            return False
        finite = np.isfinite(a) & np.isfinite(b)
        if finite.any():
            d = float(np.abs(a[finite] - b[finite]).max())
            if d > worst:
                worst_key, worst = key, d
    ok = worst <= tolerance and not missing
    verdict = 'MATCH' if ok else 'DIFFERS'
    print(f'  {name:28s} {verdict}  worst |delta| = {worst:.2e}'
          + (f' at {worst_key}' if worst_key else '')
          + (f'; keys only on one side: {missing[:4]}' if missing else ''))
    return ok


def main():
    results = []

    print('Figure 5 observables:')
    tracked = dict(np.load(fig5.CACHE_NPZ, allow_pickle=True))
    recomputed = fig5.compute_observables(cache_npz=paths.ROOT / '.verify-fig5.npz')
    results.append(compare('fig5-observables.npz', tracked, recomputed))
    (paths.ROOT / '.verify-fig5.npz').unlink(missing_ok=True)

    print('Table S3 observables:')
    tracked = dict(np.load(s3.CACHE_NPZ, allow_pickle=True))
    recomputed = s3.compute(cache_npz=paths.ROOT / '.verify-s3.npz')
    results.append(compare('table-s3-pbc-self-image.npz', tracked, recomputed))
    (paths.ROOT / '.verify-s3.npz').unlink(missing_ok=True)

    print('Figure S4 populations:')
    tracked = dict(np.load(s4.CACHE_NPZ))
    t, p, n = s4.load_trex_populations()
    results.append(compare('fig-s4-trex-populations.npz', tracked,
                           {'temperatures': t, 'populations': p, 'n_frames': n}))

    print()
    if all(results):
        print('all products reproduce within tolerance')
        return 0
    print('at least one product does not reproduce; see above')
    return 1


if __name__ == '__main__':
    sys.exit(main())
