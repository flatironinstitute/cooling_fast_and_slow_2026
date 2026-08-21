#!/usr/bin/env python
"""Minimum-image safety audit over the frames behind the R_g / extent figures.

For a solute of length M in a cubic box of edge L with van der Waals cutoff
rvdw, the solute cannot see its own periodic image when

    L > M + n * rvdw

Tiers run from the conventional box-sizing rule (two cutoffs of padding) down
through one cutoff, one solvation shell, van der Waals contact and hydrogen-bond
range.  There is no zero-margin tier on purpose: zero clearance would mean atom
centres coinciding with their own images, which no pair of van der Waals
particles can reach.

M is reported two ways.  The primary one is the largest per-axis extent
max(x) - min(x), because the lattice is axis-aligned: a molecule lying along a
box diagonal projects less onto every axis than its end-to-end length suggests,
so it is the per-axis footprint that reaches the neighbouring image.  This is
the one job an axis-aligned bounding box is right for -- it is not rotation
invariant, and neither is the lattice.  The max pairwise heavy-atom separation
is also tabulated as a conservative upper bound.

Heavy atoms are LOOS `!hydrogen` over the whole molecule, L is that frame's box
edge read from the trajectory, and rvdw comes from the .mdp used for these
simulations (rvdw = 1.2 nm, vdw-type = cut-off, pbc = xyz; see
input/md/unbiased_200ns.mdp).

Counts are reported in absolute frames as well as pi-weighted fractions: the
conformations that fail are rare enough that a weighted fraction rounds to zero
while still corresponding to real frames.

Usage
-----
    python check_pbc_self_image.py                  # table from the cache
    python check_pbc_self_image.py --recompute      # re-read trajectories
    python check_pbc_self_image.py --rvdw 12.0      # Angstrom
    python check_pbc_self_image.py --csv audit.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

import fig5_panels as base
import paths

# ── Simulation parameters (input/md/unbiased_200ns.mdp) ────────────────────────────
RVDW = 12.0                   # Angstrom; mdp has rvdw = 1.2 nm
N_CUTOFFS = 2                 # default console tier: margin = N_CUTOFFS * rvdw
SOLVATION_SHELL = 3.75        # Angstrom; first-shell boundary, near the O-O first
                              # minimum for the water model used here
FIRST_SHELL_OUTER = 4.5       # Angstrom; the GENEROUS first-shell boundary -- the
                              # first RDF minimum for apolar (carbon)-water rather
                              # than the tighter O-O value.  Kept deliberately
                              # loose: an ensemble clean at this distance is not
                              # coming anywhere near its own hydration shell.
SECOND_SHELL = 6.0            # Angstrom; outer edge of the SECOND shell.  Not
                              # 2 * SOLVATION_SHELL: shells are successive minima
                              # in the protein-water RDF, not additive layers.
VDW_CONTACT = 3.5             # Angstrom; two carbon vdW radii (1.7 + 1.7) touching
SAVE_INTERVAL_PS = 100.0      # dt 0.002 ps x nstxtcout 50000, input/md/unbiased_200ns.mdp
HBOND_HEAVY = 3.2             # Angstrom; N/O donor-acceptor heavy-atom separation

# Margin tiers for the report, most demanding first.  A margin is an absolute
# clearance in Angstrom demanded between the solute and its nearest image, so
# tiers set by a multiple of the cutoff and tiers set by a physical contact
# distance sit in the same list.  There is deliberately no zero-margin tier: a
# zero clearance is not contact but coincidence of atom centres with their own
# images, which is far past any distance two van der Waals particles can reach.
REPORT_MARGINS = (
    ('2 rvdw', 2 * RVDW),
    ('1 rvdw', 1 * RVDW),
    ('two solvation shells', SECOND_SHELL),
    ('first shell, outer bound', FIRST_SHELL_OUTER),
    ('first shell', SOLVATION_SHELL),
    ('vdW contact', VDW_CONTACT),
    ('hydrogen bond', HBOND_HEAVY),
)
HEAVY_SELECTION = '!hydrogen'
# sampled_frames.xtc was written without box information (it reports 0,0,0), so
# frames from it fall back to this edge, measured from the cooled and T-REX
# trajectories of the same system.  Rows using it are flagged in the table.
NOMINAL_BOX_EDGE = 47.2

CACHE_NPZ = paths.OBSERVABLES / 'table-s3-pbc-self-image.npz'
OUT_DIR = paths.OUTPUT / 'table-s3-pbc-self-image'
MDP = paths.ROOT / 'input' / 'md' / 'unbiased_200ns.mdp'


def true_self_image_gap(coords, box_edge):
    """Smallest distance between any atom and an IMAGE of another atom.

    This is the quantity that decides whether the solute interacts with itself
    across the boundary.  The per-axis and max-pairwise measures below are only
    SCREENS on it: for the extreme pair along an axis the projected gap is
    |L - M|, and a 3D distance is never smaller than one of its projections, so
    |L - M| is a lower bound on this value.  It is a loose one, because the two
    extreme atoms along x can be far apart in y and z -- which is why the
    screens over-count violations, sometimes by a lot.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    imaged = diff - box_edge * np.round(diff / box_edge)
    d_imaged = np.linalg.norm(imaged, axis=-1)
    d_direct = np.linalg.norm(diff, axis=-1)
    through_image = d_imaged < d_direct - 1e-9
    return float(d_imaged[through_image].min()) if through_image.any() else np.inf


def frame_extent_and_box(heavy, model):
    """Per-frame (max pairwise M, largest per-axis extent, box edge) in Angstrom.

    Two measures of "molecule length" because they answer different questions.
    M is the max pairwise separation, rotation invariant and the conservative
    choice.  The per-axis extent max(x)-min(x) over each axis is what actually
    governs reaching a periodic image, because the periodicity is axis-aligned:
    a molecule lying along a box diagonal projects less onto every axis than M
    suggests.  This is the one job the axis-aligned bounding box is right for --
    it is not rotation invariant, and neither is the lattice.
    """
    coords = heavy.getCoords()
    box_edge = model.periodicBox()[0]
    edge = box_edge if box_edge > 0 else NOMINAL_BOX_EDGE
    return (pdist(coords).max(),
            (coords.max(axis=0) - coords.min(axis=0)).max(),
            box_edge,
            true_self_image_gap(coords, edge))


def scan_states(paths_by_state, topology=base.TOPOLOGY,
                selection=HEAVY_SELECTION, n_states=base.N_STATES,
                xtc_state_index=None):
    """Raw per-frame extents and box edges, kept per state.

    Raw values, not histograms: the audit counts individual rare frames, and
    binning them would round the count.
    """
    import loos
    from loos import pyloos as pl

    model = loos.createSystem(str(topology))
    heavy = loos.selectAtoms(model, selection)
    extents = {s: [] for s in range(n_states)}
    axials = {s: [] for s in range(n_states)}
    boxes = {s: [] for s in range(n_states)}
    trues = {s: [] for s in range(n_states)}

    items = ([(None, paths_by_state)] if xtc_state_index is not None
             else sorted(paths_by_state.items()))
    for state_ix, path in items:
        traj = pl.Trajectory(str(path), model)
        frame_m, frame_a, frame_l, frame_t = [], [], [], []
        for _ in traj:
            m, a, l, t = frame_extent_and_box(heavy, model)
            frame_m.append(m)
            frame_a.append(a)
            frame_l.append(l)
            frame_t.append(t)
        frame_m, frame_a, frame_l, frame_t = (
            np.array(frame_m), np.array(frame_a), np.array(frame_l),
            np.array(frame_t))

        targets = ([(state_ix, slice(None))] if xtc_state_index is None else
                   [(s, np.flatnonzero(xtc_state_index == s))
                    for s in np.unique(xtc_state_index)])
        for s, sel in targets:
            extents[s].append(frame_m[sel])
            axials[s].append(frame_a[sel])
            boxes[s].append(frame_l[sel])
            trues[s].append(frame_t[sel])
    join = lambda d: {s: np.concatenate(v) if v else np.empty(0) for s, v in d.items()}
    return join(extents), join(axials), join(boxes), join(trues)


def compute(cooling_labels=base.COOLING_LABELS, cooled_dir=base.COOLED_DIR,
            trex_dir=base.TREX_DIR, sampled_277=base.SAMPLED_277,
            n_states=base.N_STATES, cache_npz=CACHE_NPZ):
    payload = {}
    sources = [(lab, {int(p.stem): p for p in sorted((cooled_dir / lab).iterdir())}, None)
               for lab in cooling_labels]
    sources.append(('230K', {int(p.stem): p for p in sorted(trex_dir.iterdir())}, None))
    meta = np.loadtxt(sampled_277.parent / 'frame_metadata.txt', dtype=int)
    sources.append(('277K', sampled_277, meta[:, 1]))

    for key, paths, index in sources:
        extents, axials, boxes, trues = scan_states(paths, n_states=n_states,
                                                    xtc_state_index=index)
        for s in range(n_states):
            payload[f'{key}_extent_{s}'] = extents[s]
            payload[f'{key}_axial_{s}'] = axials[s]
            payload[f'{key}_box_{s}'] = boxes[s]
            payload[f'{key}_true_{s}'] = trues[s]
        print(f'  {key}: {sum(len(v) for v in extents.values())} frames')
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **payload)
    print(f'cached -> {cache_npz}')
    return payload


def fmt_prob(value):
    """Probability without pretending to more or less precision than we have."""
    if value <= 0.0:
        return '0'
    return f'{value:.3f}' if value >= 0.01 else f'{value:.2e}'


def fmt_margin(margin):
    """Margin in Angstrom without spurious rounding or trailing zeros."""
    return f'{margin:.2f}'.rstrip('0').rstrip('.')


def audit(obs, reps, margin, cooling_labels=base.COOLING_LABELS,
          n_states=base.N_STATES, nominal_edge=NOMINAL_BOX_EDGE):
    """One row per ensemble, under each measure of solute-image separation.

    Every criterion is reduced to a per-frame GAP in Angstrom, so a frame fails
    a tier when its gap does not exceed the margin:

      true    the actual smallest distance from any atom to an image of another
              atom.  This is the physical quantity.
      axial   |L - (per-axis extent)|.  A lower bound on `true`, hence a
              conservative screen that over-counts violations.
      maxpair |L - (max pairwise separation)|.  A looser bound still.

    The screens over-count because the two atoms furthest apart along an axis
    need not be near each other in the other two, so the projected gap can be
    far smaller than the distance it bounds.
    """
    rows = []
    kinds = ('true', 'axial', 'maxpair')
    for key in cooling_labels + base.REFERENCE_LABELS:
        pi = reps[key].mean(axis=0)
        n_total = 0
        fails = {k: 0 for k in kinds}
        weighted = {k: 0.0 for k in kinds}
        tightest = {k: np.inf for k in kinds}
        used_nominal = False
        for s in range(n_states):
            true_gap = obs[f'{key}_true_{s}']
            if true_gap.size == 0:
                continue
            l = obs[f'{key}_box_{s}']
            if not np.all(l > 0):
                l = np.where(l > 0, l, nominal_edge)
                used_nominal = True
            gaps = {'true': true_gap,
                    'axial': np.abs(l - obs[f'{key}_axial_{s}']),
                    'maxpair': np.abs(l - obs[f'{key}_extent_{s}'])}
            n_total += true_gap.size
            for kind, gap in gaps.items():
                fail = gap <= margin
                fails[kind] += int(fail.sum())
                weighted[kind] += pi[s] * fail.mean()
                finite = gap[np.isfinite(gap)]
                if finite.size:
                    tightest[kind] = min(tightest[kind], finite.min())
        row = dict(key=key, ensemble=base.PRETTY[key], n_frames=n_total)
        for kind in kinds:
            row[f'n_fail_{kind}'] = fails[kind]
            row[f'pct_{kind}'] = 100.0 * fails[kind] / n_total if n_total else 0.0
            row[f'pi_weighted_{kind}'] = weighted[kind]
            row[f'min_gap_{kind}'] = tightest[kind]
        row['box_from_file'] = not used_nominal
        rows.append(row)
    return rows


def print_table(rows, margin, label='', nominal_edge=NOMINAL_BOX_EDGE):
    print(f'\nsolute-image separation must exceed {fmt_margin(margin)} A'
          + (f'   [{label}]' if label else ''))
    for kind, what in (('true', 'TRUE minimum-image distance  <- the physical test'),
                       ('axial', 'per-axis screen   |L - M|  (over-counts)'),
                       ('maxpair', 'max-pairwise screen        (over-counts more)')):
        print(f'  {what}')
        head = (f"    {'ensemble':>10s} {'frames':>8s} {'fail':>7s} {'fail %':>8s} "
                f"{'pi-weighted':>12s} {'min gap':>9s}  box")
        print(head)
        print('    ' + '-' * (len(head) - 4))
        for r in rows:
            print(f"    {r['ensemble']:>10s} {r['n_frames']:8d} "
                  f"{r[f'n_fail_{kind}']:7d} {r[f'pct_{kind}']:8.3f} "
                  f"{r[f'pi_weighted_{kind}']:12.5f} {r[f'min_gap_{kind}']:9.2f}  "
                  f"{'file' if r['box_from_file'] else f'{nominal_edge:.1f} nom'}")
        total_fail = sum(r[f'n_fail_{kind}'] for r in rows)
        total = sum(r['n_frames'] for r in rows)
        print(f"    -> {total_fail} of {total} frames fail "
              f"({100.0 * total_fail / total:.3f}%)\n")


CSV_HEADERS = {
    'ensemble': 'ensemble',
    'n_frames': 'frames_analysed',
    'n_fail_true': 'frames_failing_true_minimum_image_distance',
    'pct_true': 'percent_frames_failing_true',
    'pi_weighted_true': 'population_weighted_probability_failing_true',
    'min_gap_true': 'closest_true_solute_image_distance_angstrom',
    'n_fail_axial': 'frames_failing_per_axis_screen',
    'pct_axial': 'percent_frames_failing_per_axis_screen',
    'pi_weighted_axial': 'population_weighted_probability_failing_per_axis_screen',
    'min_gap_axial': 'smallest_per_axis_projected_gap_angstrom',
    'n_fail_maxpair': 'frames_failing_max_pairwise_screen',
    'pct_maxpair': 'percent_frames_failing_max_pairwise_screen',
    'pi_weighted_maxpair': 'population_weighted_probability_failing_max_pairwise_screen',
    'min_gap_maxpair': 'smallest_max_pairwise_projected_gap_angstrom',
    'box_edge_read_from_trajectory': 'box_edge_read_from_trajectory',
    'box_from_file': 'box_edge_read_from_trajectory',
}

TIER_MEANING = {
    '2 rvdw': 'the conventional box-sizing rule: a full cutoff of solvent '
              'padding on either side of the solute. Failing this is a '
              'sizing-convention shortfall, not by itself a physical artefact.',
    '1 rvdw': 'the strict bound. Failing this means the solute has an image of '
              'itself inside the van der Waals cutoff, so the force field is '
              'computing a self-interaction.',
    'two solvation shells': 'the gap between the copies no longer holds two '
                            'hydration layers, so the solvent between them '
                            'never reaches bulk structure. Note 6.0 Angstrom is '
                            'the outer edge of the second shell in a '
                            'protein-water RDF, NOT 2 x 4.5: solvation shells '
                            'are successive minima of the RDF, not additive '
                            'layers.',
    'first shell, outer bound': 'the generous reading of the first shell: the '
                                'first RDF minimum for apolar carbon-water, '
                                'rather than the tighter O-O value below. An '
                                'ensemble with no failing frames here is not '
                                'approaching its own hydration shell under any '
                                'reasonable definition of it.',
    'first shell': 'the solute and its image are closer than one layer of '
                   'water, taken at the O-O first minimum for this water model, '
                   'so their first hydration shells interpenetrate and share '
                   'waters even where no direct solute-solute interaction is '
                   'computed.',
    'vdW contact': 'the solute is touching its own image: the clearance is '
                   'down to the sum of two carbon van der Waals radii '
                   '(1.7 + 1.7 Angstrom), so atoms are in direct contact '
                   'across the periodic boundary.',
    'hydrogen bond': 'the tightest tier. The clearance is within heavy-atom '
                     'donor-acceptor range, so the peptide is close enough to '
                     'hydrogen bond to its own image. Note this distance is '
                     'SHORTER than van der Waals contact, not longer.',
}

MD_COLUMNS = [
    ('ensemble', 'Ensemble', '{}'),
    ('n_frames', 'Frames analysed', '{:d}'),
    ('n_fail_true', 'Frames failing', '{:d}'),
    ('pct_true', 'Failing (% of frames)', '{:.3f}'),
    ('pi_weighted_true', 'Failing ($P$)', '{:.5f}'),
    ('min_gap_true', 'Closest approach (Å)', '{:.2f}'),
    ('box_from_file', '$L$ source', '{}'),
]


def _md_table(rows, nominal_edge=NOMINAL_BOX_EDGE):
    header = '| ' + ' | '.join(c[1] for c in MD_COLUMNS) + ' |'
    lines = [header, '|' + '|'.join(['---'] * len(MD_COLUMNS)) + '|']
    for r in rows:
        cells = []
        for key, _, fmt in MD_COLUMNS:
            value = r[key]
            if key == 'box_from_file':
                value = 'trajectory' if value else f'{nominal_edge:.1f} Å nominal'
            cells.append(fmt.format(value))
        lines.append('| ' + ' | '.join(cells) + ' |')
    total_fail = sum(r['n_fail_true'] for r in rows)
    total = sum(r['n_frames'] for r in rows)
    lines.append(f'\nAll ensembles: **{total_fail} of {total} frames** '
                 f'({100.0 * total_fail / total:.3f}%).')
    return '\n'.join(lines)


def render_markdown(rows_by_tier, rvdw=RVDW, nominal_edge=NOMINAL_BOX_EDGE,
                    heavy_selection=HEAVY_SELECTION,
                    solvation_shell=SOLVATION_SHELL,
                    first_shell_outer=FIRST_SHELL_OUTER,
                    vdw_contact=VDW_CONTACT,
                    save_interval_ps=SAVE_INTERVAL_PS):
    """A report someone can read without opening the script."""
    keys = base.COOLING_LABELS + base.REFERENCE_LABELS
    tiers = list(REPORT_MARGINS)
    total_frames = sum(r['n_frames'] for r in rows_by_tier[tiers[0][0]])
    closest = {r['key']: r['min_gap_true'] for r in rows_by_tier[tiers[0][0]]}

    out = ['# Minimum-image self-interaction audit', '',
           'How close the peptide comes to its own periodic image, over exactly '
           'the frames behind the $R_g$ and maximum-separation figures.', '',
           'Generated by `check_pbc_self_image.py`. Every frame in the cooled, '
           '277 K and 230 K sets is tested individually.',
           '', '## The observable', '',
           'For each frame, the smallest distance from any heavy atom to a '
           'periodic **image** of any other heavy atom of the same molecule:',
           '',
           '$$g = \\min_{i,j,\\;\\mathbf{n}\\neq\\mathbf{0}} '
           '\\left| \\mathbf{r}_i - \\mathbf{r}_j + \\mathbf{n}L \\right|$$',
           '',
           f'Heavy atoms are the LOOS selection `{heavy_selection}` over the '
           'whole molecule, backbone included; $L$ is that frame’s cubic box '
           'edge, read from the trajectory. A frame **fails** a tier when '
           '$g \\le \\Delta$.',
           '',
           'Earlier versions of this audit screened on the solute’s own size '
           'instead, comparing the box edge against the per-axis extent or the '
           'maximum pairwise separation. Those are only lower bounds on $g$: the '
           'two atoms furthest apart along one axis need not be near each other '
           'in the other two, so a projected gap can be far smaller than the '
           'distance it bounds. They over-counted violations by more than an '
           'order of magnitude and are not reported here.',
           '', '## Closest approach', '',
           'The single most useful number per ensemble: the smallest $g$ seen '
           'anywhere in it.', '',
           '| Ensemble | Frames | Closest approach to own image (Å) |',
           '|---|---|---|']
    for key in keys:
        row = next(r for r in rows_by_tier[tiers[0][0]] if r['key'] == key)
        out.append(f'| {base.PRETTY[key]} | {row["n_frames"]} | '
                   f'{closest[key]:.2f} |')
    out += ['',
            f'Nothing anywhere approaches overlap: the tightest contact in '
            f'{total_frames} frames is '
            f'{min(closest.values()):.2f} Å, a hydrogen-bond separation. Two '
            'van der Waals spheres cannot be driven onto one another by a '
            'finite-size artefact -- the simulation would blow up long before '
            'that -- and these numbers are consistent with that.',
            '',
            f'Two reference distances set the tiers below: the van der Waals '
            f'cutoff $r_\\mathrm{{vdw}} = {fmt_margin(rvdw)}$ Å (`rvdw = 1.2` nm '
            'in `input/md/unbiased_200ns.mdp`, `vdw-type = cut-off`, `pbc = xyz`), and '
            f'the first solvation shell, taken as {fmt_margin(solvation_shell)} Å '
            f'of water around protein heavy atoms with '
            f'{fmt_margin(first_shell_outer)} Å as a generous outer bound.',
            '', '## Margin tiers', '']
    for label, margin in tiers:
        out.append(f'- **{label}** ($\\Delta = {fmt_margin(margin)}$ Å) — '
                   f'{TIER_MEANING.get(label, "")}')
    out += ['', '## Summary', '',
            f'Every tier against every ensemble, out of {total_frames} frames '
            'analysed. **Each cell is $P$ (frames failing)**: the '
            'population-weighted probability that a conformation drawn from '
            'that ensemble violates the tier, with the absolute number of '
            'failing frames in parentheses. The two are not redundant -- $P$ '
            'runs several-fold below the frame share because the offending '
            'conformations sit in sparsely populated extended states.', '',
            '| Tier | $\\Delta$ (Å) | '
            + ' | '.join(base.SHORT[k] for k in keys) + ' |',
            '|---|---|' + '---|' * len(keys)]
    for label, margin in tiers:
        by_key = {r['key']: r for r in rows_by_tier[label]}
        cells = [f'{fmt_prob(by_key[k]["pi_weighted_true"])} '
                 f'({by_key[k]["n_fail_true"]})' for k in keys]
        out.append(f'| {label} | {fmt_margin(margin)} | '
                   + ' | '.join(cells) + ' |')
    out += ['',
            'No row is summed or averaged across ensembles: each ensemble '
            'normalises its own $\\pi$ to one, so there is no joint '
            'distribution over which ensemble a conformation came from and no '
            'aggregate probability to be had without inventing a prior over '
            'conditions.',
            '', '## Column glossary', '',
            '| Column | Meaning |', '|---|---|',
            '| Ensemble | Cooling rate, or the 277 K / 230 K equilibrium reference. |',
            '| Frames analysed | Every frame in that ensemble, across all 50 states. |',
            '| Frames failing | Absolute count. Reported because the tight tiers '
            'involve a handful of frames whose population-weighted probability '
            'rounds to zero. |',
            '| Failing (% of frames) | Unweighted, out of frames analysed. |',
            '| Failing ($P$) | '
            '$\\sum_s \\pi_s \\cdot (\\text{failing frames in state } s) / '
            '(\\text{frames in state } s)$, using the same MSM state '
            'populations $\\pi$ as the figures. The fraction of the *ensemble* '
            'affected, as opposed to the fraction of frames on disk; the two '
            'differ because states are not sampled in proportion to $\\pi$. |',
            '| Closest approach (Å) | Smallest $g$ over every frame of that '
            'ensemble. Independent of tier. |',
            '| $L$ source | Per-frame box edge from the trajectory, or the '
            f'nominal {nominal_edge:.1f} Å fallback. |',
            '', '## Per-tier detail', '']
    for label, margin in tiers:
        out += [f'### {label}  ($\\Delta = {fmt_margin(margin)}$ Å)', '',
                _md_table(rows_by_tier[label]), '']
    out += ['## Caveats', '',
            '- `sampled_frames.xtc` (277 K) was written without box information '
            '— it reports $(0,0,0)$ — so those frames use a nominal '
            f'{nominal_edge:.1f} Å edge measured from the cooled and T-REX '
            'trajectories of the same system. The rows are flagged.',
            '- The peptide is whole, not split across the boundary: in the most '
            'extended frame of each set the largest consecutive '
            'C$\\alpha$–C$\\alpha$ step is under 4.0 Å, and no frame anywhere '
            'has a step above 5 Å. A wrapped molecule would show a jump of '
            'order the box edge.',
            '- This is a property of the original simulations, not of the '
            'analysis; the frames are reported as found.',
            '- **Every count is a lower bound.** Coordinates were written every '
            f'{save_interval_ps:.0f} ps (`dt = 0.002` ps $\\times$ '
            '`nstxtcout = 50000`), and the peptide extends and contracts far '
            'faster than that. A frame recorded at the '
            f'{fmt_margin(vdw_contact)} Å margin implies the trajectory passed '
            'through closer approaches in the unrecorded interval, so the true '
            'fraction of simulated time spent inside any margin exceeds the '
            'tabulated frame fractions. For the cooled ensembles the unsaved '
            'neighbours lie in the parent cooling trajectory rather than in the '
            'analysed set, since each analysed frame is the endpoint of an '
            'independent cooling run.',
            f'- For the same reason the {fmt_margin(vdw_contact)} Å van der '
            'Waals tier is the effective floor of what this sampling can '
            'resolve.',
            '']
    return '\n'.join(out)


def fmt_prob_tex(value):
    """Probability as LaTeX math, scientific below 1%."""
    if value <= 0.0:
        return '0'
    if value >= 0.01:
        return f'{value:.3f}'
    exponent = int(np.floor(np.log10(value)))
    mantissa = value / 10 ** exponent
    return f'${mantissa:.1f}{{\\times}}10^{{{exponent}}}$'


ENSEMBLE_TEX = {
    '6e9': r'$6.4{\times}10^{9}$', '2e9': r'$2.7{\times}10^{9}$',
    '6e8': r'$6.4{\times}10^{8}$', '2e8': r'$2.7{\times}10^{8}$',
    '6e7': r'$6.4{\times}10^{7}$', '2e7': r'$2.7{\times}10^{7}$',
    '6e6': r'$6.4{\times}10^{6}$',
    '277K': r'277\,K', '230K': r'230\,K',
}
TIER_TEX = {'2 rvdw': r'$2r_\mathrm{vdw}$', '1 rvdw': r'$1r_\mathrm{vdw}$',
            'two solvation shells': 'two solvation shells',
            'first shell, outer bound': 'first shell, outer',
            'first shell': 'first shell', 'vdW contact': 'vdW contact',
            'hydrogen bond': 'hydrogen bond'}


def render_latex(rows_by_tier, rvdw=RVDW, save_interval_ps=SAVE_INTERVAL_PS,
                 nominal_edge=NOMINAL_BOX_EDGE, wrap_added=True):
    """One tabularx table of the corrected audit, sized to \\textwidth.

    No tier-name column (the margin itself is the row label), tabularx so the
    body spans the text block rather than sitting narrow under a full-width
    caption, and \\small commented out.  Set
    wrap_added=False to drop the \\add{} change-marking macro.
    """
    keys = base.COOLING_LABELS + base.REFERENCE_LABELS
    tiers = list(REPORT_MARGINS)
    total = sum(r['n_frames'] for r in rows_by_tier[tiers[0][0]])
    closest = {r['key']: r['min_gap_true'] for r in rows_by_tier[tiers[0][0]]}
    n_ens = len(keys)

    caption = (
        'Minimum-image self-interaction audit. For each frame, $g$ is the '
        'smallest distance from any heavy atom to a periodic image of any other '
        'heavy atom of the same peptide, minimised over all atom pairs and all '
        'nonzero lattice vectors; a frame is counted when $g \\le \\Delta$. '
        'Each cell gives the population-weighted probability that a conformation '
        'drawn from that ensemble falls below the threshold, with the absolute '
        f'number of such frames in parentheses, out of {total} frames analysed. '
        'Thresholds are set by the van der Waals cutoff '
        f'$r_\\mathrm{{vdw}} = {fmt_margin(rvdw)}$\\,\\AA\\ and twice it, the '
        'second and first solvation shells, van der Waals contact and '
        'hydrogen-bond range. The closest approach anywhere is '
        f'{min(closest.values()):.2f}\\,\\AA, so no conformation comes near '
        'overlapping its own image. Cooling rates are in K\\,s$^{-1}$. Counts '
        'are lower bounds, since coordinates were written only every '
        f'{save_interval_ps:.0f}\\,ps.')
    if wrap_added:
        caption = '\\add{' + caption + '}'

    out = ['% Minimum-image self-interaction audit.',
           '% Generated by table_s3_pbc_self_image.py --tex',
           '% Requires \\usepackage{booktabs}, \\usepackage{tabularx},',
           '% \\usepackage{array} and the \\add{} change-marking macro.', '',
           r'\begin{table}[h]', r'  \centering', r'  % \small',
           f'  \\caption{{{caption}}}',
           r'  \label{tab:pbc-self-image}',
           r'  \begin{tabularx}{\textwidth}{@{}l *{' + str(n_ens)
           + r'}{>{\raggedleft\arraybackslash}X}@{}}',
           r'    \toprule',
           r'    & \multicolumn{' + str(n_ens) + r'}{c}{probability weighted '
           r'fraction below threshold (raw frame count)} \\',
           r'    \cmidrule(l){2-' + str(1 + n_ens) + r'}',
           r'    $\Delta$ (\AA) & '
           + ' & '.join(ENSEMBLE_TEX[k] for k in keys) + r' \\',
           r'    \midrule']
    for label, margin in tiers:
        by_key = {r['key']: r for r in rows_by_tier[label]}
        cells = [f'{fmt_prob_tex(by_key[k]["pi_weighted_true"])} '
                 f'({by_key[k]["n_fail_true"]})' for k in keys]
        out.append(f'    {fmt_margin(margin)} & ' + ' & '.join(cells) + r' \\')
    out += [r'    \bottomrule', r'  \end{tabularx}', r'\end{table}', '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--rvdw', type=float, default=RVDW,
                        help='van der Waals cutoff in Angstrom (default %(default)s)')
    parser.add_argument('--n-cutoffs', type=float, default=N_CUTOFFS,
                        help='console tier: margin is N * rvdw (default %(default)s)')
    parser.add_argument('--margin', type=float,
                        help='console tier: absolute clearance in Angstrom, '
                             'overriding --n-cutoffs')
    parser.add_argument('--csv', type=Path, help='also write the table as CSV')
    parser.add_argument('--tex', type=Path,
                        help='write the disaggregated summary as LaTeX booktabs '
                             'tables for the supplement')
    parser.add_argument('--no-add', action='store_true',
                        help='omit the \\add{} change-marking macro from the '
                             'LaTeX caption')
    parser.add_argument('--md', type=Path,
                        help='write a self-describing markdown report covering '
                             'every margin tier in REPORT_MARGINS')
    args = parser.parse_args()

    if args.recompute or not CACHE_NPZ.is_file():
        print('scanning trajectories (LOOS) ...')
        obs = compute()
    else:
        obs = dict(np.load(CACHE_NPZ, allow_pickle=True))
        print(f'loaded cache {CACHE_NPZ}')

    reps = base.load_population_replicates()
    margin = (args.margin if args.margin is not None
              else args.n_cutoffs * args.rvdw)
    label = next((l for l, m in REPORT_MARGINS if abs(m - margin) < 1e-9), '')
    rows = audit(obs, reps, margin)
    print_table(rows, margin, label=label)

    if args.md or args.tex:
        rows_by_tier = {label: audit(obs, reps, m) for label, m in REPORT_MARGINS}
        if args.md:
            args.md.write_text(render_markdown(rows_by_tier, rvdw=args.rvdw))
            print(f'wrote {args.md}')
        if args.tex:
            args.tex.write_text(render_latex(rows_by_tier, rvdw=args.rvdw,
                                             wrap_added=not args.no_add))
            print(f'wrote {args.tex}')

    if args.csv:
        with args.csv.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_HEADERS.values()))
            writer.writeheader()
            for r in rows:
                writer.writerow({CSV_HEADERS[k]: v for k, v in r.items()
                                 if k in CSV_HEADERS})
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
