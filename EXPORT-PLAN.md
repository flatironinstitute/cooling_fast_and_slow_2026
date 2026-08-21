# Export plan: Figure 5, Figure S4, Table S3

This branch adds the analysis behind the main-text multipanel observables figure
and two supplementary items, in a form that rebuilds inside this repository.

Nothing belonging to another author is modified. Two files are appended to and
one new input slot is requested; those are listed under
[Needs the repo owner's sign-off](#needs-the-repo-owners-sign-off).

## Two-tier rebuild

The coordinate data lives in the companion Zenodo deposit, but the derived
observables are small enough to track here (~8 MB total), so the figures rebuild
with no download at all.

| | command | needs | time |
|---|---|---|---|
| **Tier 1** | `make figures` | numpy, scipy, matplotlib | seconds |
| **Tier 2** | `make observables` | + mdtraj, h5py, LOOS, and the staged Zenodo inputs | ~30 min |
| **Check** | `make verify` | as tier 2 | ~30 min |

`make verify` recomputes from coordinates and reports the largest deviation per
product without overwriting anything. It exists because npz archives are zip
files carrying entry timestamps, so they are not byte-reproducible and
`git diff` cannot serve as the comparison.

## What this branch adds

| file | role |
|---|---|
| `paths.py` | repository-root resolution; every path is anchored here |
| `fig5_panels.py` | Figure 5: Asn 1 and Trp 6 $\chi_1$, $R_g$, Trp 6 contacts |
| `fig_s4_trex_populations.py` | Figure S4: $\log_{10}\pi$ across the T-REX ladder |
| `table_s3_pbc_self_image.py` | Table S3: minimum-image self-interaction audit |
| `build-cooled-by-state.py` | tier 2: cooling endpoints → one DCD per state |
| `build-trex-230-by-state.py` | tier 2: 230 K replica → one DCD per state |
| `verify-observables.py` | tier 2: recompute and compare, no overwrite |
| `Makefile`, `environment.yml` | orchestration and dependencies |
| `derived/observables/*.npz` | the tracked observables (~1.7 MB) |
| `derived/cooling_bootstraps_*.npz` | state-population bootstrap replicates (5.5 MB) |
| `derived/pipeline-config.yaml` | config of the pipeline that produced them |

`paths.py` resolves the root via `git rev-parse --show-toplevel`, falling back to
walking up for a directory containing both `input/` and `output/`. The fallback
matters: a reader who downloads a tarball or zip has no `.git`.

## Error model, in one paragraph

Every error bar is a two-level bootstrap, $N_\mathrm{boot} = 2000$, 95%
percentile intervals. The outer level resamples the state populations using the
replicates already behind the other figures (`cooled_<rate>`,
`pi_230K_samples`, `pi_samples`). The inner level resamples frames within each
state multinomially. **The inner term dominates the outer by 4–6×**, so
population uncertainty alone would have been misleadingly tight.

Two caveats belong in the methods:

- `pi_samples` provides **50** distinct posterior draws of $\pi(277\,\mathrm{K})$,
  not 5000. The upstream `P-samples-lag-400.npz` stores 100 transition-matrix
  draws per stationary vector — $\pi$ is held fixed within each block to
  $4.7\times10^{-14}$ — so the 100× multiplicity carries kinetic variation at
  fixed populations and adds nothing to an equilibrium observable. The copy in
  `derived/` stores the 50 distinct draws directly.
- Replicates for different ensembles came from independent RNG streams, so
  between-ensemble differences are computed unpaired. The true estimators are
  positively correlated, which makes those difference bands conservative.

## Required action: five numbers in the text

The 6.4e6 per-state frames in the analysis tree were built from a manifest that
was **superseded six weeks later**. Their DCDs date from 2026-02-22; the current
`end-6e6-traj-list.txt` dates from 2026-04-03. Regenerating from the current
manifests changes ~4% of the 6.4e6 frames and nothing else:

| quantity | published | regenerated | shift | 95% CI half-width |
|---|---|---|---|---|
| Asn 1 $\chi_1$ *g+* | 0.4817 | **0.4762** | −0.0055 | 0.030 |
| frac $R_g < 8$ Å | 0.7199 | **0.7162** | −0.0037 | 0.024 |
| $\langle R_g\rangle$ | 7.8725 Å | **7.8791 Å** | +0.0066 Å | 0.013 Å |

Every shift is a fraction of its own error bar, and no ordering or conclusion
changes. 6.4e9, 6.4e8, 277 K and 230 K are unchanged to the last digit; 6.4e7
shifts by 0.0001 because the single state affected there carries 0.35% of that
ensemble's weight. **The observables committed here are the regenerated set**,
so the text needs those three values updated.

Verification that the regeneration method is right, rather than merely
plausible: rebuilding from `manifest` + `pairs[:,1]` reproduces 6.4e9 and 6.4e8
**bit-for-bit** across all 50 states (worst deviation $4.8\times10^{-7}$ nm,
float32 round-trip), and 49 of 50 states for 6.4e7. That is the same grouping
rule as `grab-frames-coolings.py` in the analysis tree.

Table S3 is unaffected in substance: 6.4e6's closest approach to its own image
is 7.75 Å either way, and its counts stay zero at every tier from 6 Å down.

## Needs the repo owner's sign-off

1. **New input slot** `input/md/topology/built-protein.pdb` (28 KB, protein-only,
   304 atoms). `input/md/topology/` currently indexes only the solvated
   `test.gro`. Every observable here needs the protein-only topology. Wants a
   SOURCE/ZENODO line in `directory_structure.txt`.
2. **`unbiased_200ns.mdp`** supplies `rvdw = 1.2` nm, `dt = 0.002` ps and
   `nstxtcout = 50000` for Table S3. These are currently named constants citing
   the file; shipping it would be better provenance.
3. **`.gitignore`** — appended two lines to exclude `derived/cooled-by-state/`
   and `derived/trex-230-by-state/`, which are rebuilt coordinate sets (~210 MB).
   Existing rules untouched.
4. **Basename collisions in the endpoint staging — please read before anyone
   copies files.** `directory_structure.txt` specifies
   `input/cooling/endpoints/xtc/{rate}/<original-basename>.xtc`, but the
   manifests draw from up to nine source directories per rate and basenames
   repeat:

   | rate | paths | unique basenames | **collisions** |
   |---|---|---|---|
   | 6e9 | 7550 | 7550 | 0 |
   | 6e8 | 7550 | 7550 | 0 |
   | 6e7 | 13539 | 13376 | **163** |
   | 6e6 | 2500 | 2373 | **127** |

   Staged flat, 290 files would be silently overwritten and the deposit would be
   quietly wrong. Prefixing with the manifest line index (`{i:05d}_{basename}`)
   fixes it and makes the manifest↔file correspondence explicit.
   `build-cooled-by-state.py` accepts either absolute original paths or
   rewritten repo-relative ones, so it works whichever convention is chosen —
   but it cannot disambiguate a flat directory by basename alone.

   Consider instead depositing the **per-state DCDs**: 200 files rather than
   31139, no namespace problem, and archives handle a few large files far better
   than tens of thousands of tiny ones.

## Provenance notes worth keeping

- **All 31139 cooling endpoint frames decompress cleanly**, so there are no
  corrupt files in the endpoint set for these four rates. A run truncated
  mid-cooling that still wrote a valid endpoint would not be caught this way.
- **`mdtraj` mis-types 26 hydrogens in `built-protein.pdb`.** The file has no
  element records, so element inference from atom names reads the 19 `HG*`
  hydrogens as **mercury** and the 7 `HE*` as **helium**. That inflates
  mdtraj's `sidechain-heavy` Trp 6 contacts by up to 0.14 (worst at the
  sequence-adjacent Leu 7) and corrupts any mass-weighted quantity. $R_g$ and
  contacts therefore go through LOOS. `mdtraj.compute_chi1` is unaffected
  because it matches atom names, not elements.
- **`enspara` is not required.** It was used for one `ra.load()` call on the
  repex dtrajs. All 90 files hold exactly one dataset named `arr_00`, which
  `h5py` reads byte-identically, so the dependency is gone. The scripts assert
  the single-dataset layout rather than assume it, because a multi-row ragged
  array would make `arr_00` a silent truncation.
- **The bootstrap npz no longer needs `allow_pickle`.** The upstream file stores
  `labels` and `pi_source_mode` as object arrays, so loading it requires
  `allow_pickle=True` — an arbitrary-code-execution hazard on a public artifact.
  The copy in `derived/` stores them as plain string arrays. Nothing indexed in
  `directory_structure.txt` is affected: `P-samples-lag-400.npz` and
  `its-from-P-samples.npz` both load clean with zero object arrays.
- The 277 K state assignment is **derived, not shipped**. `sampled_frames.xtc`
  is blocked by state, 1000 consecutive frames per state in order, so the
  assignment is `np.repeat(np.arange(50), 1000)`. `fig5_panels.py` uses
  `frame_metadata.txt` when present and otherwise derives it, asserting the
  frame count first so a changed picked-frame set fails loudly. Shipping the
  sidecar would additionally preserve source-trajectory provenance.

## Deviation from repo convention

The three product scripts use underscores (`fig5_panels.py`) rather than the
repo's hyphens (`compute-sasa.py`, `fig-s1.ipynb`). `table_s3_pbc_self_image.py`
imports `fig5_panels` for the shared bootstrap and style code, and a hyphenated
filename is not importable. Non-imported scripts keep hyphens
(`build-cooled-by-state.py`). Extracting the shared code into a neutral module
would let all of them use hyphens; that is a reasonable follow-up rather than
something done here, to keep this branch's diff small.

## Open items

- Zenodo DOI and per-file names are not yet assigned, so
  `directory_structure.txt` additions cannot cite them. Placeholder until the
  deposit is prepped.
- A checksum manifest for the deposited files would give corruption detection,
  download verification, and the index-prefix mapping from item 4 in one CSV.
- `build-trex-230-by-state.py` takes `--seed` (the Makefile passes one) because
  the original sampler was unseeded and so did not reproduce frame-for-frame.
  Omitting the flag preserves the original behaviour exactly.
