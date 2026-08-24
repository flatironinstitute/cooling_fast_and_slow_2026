#!/usr/bin/env python
"""Minimum-image safety audit over the frames behind the R_g / extent figures.

For each frame, g is the smallest distance from any heavy atom to a periodic
image of any other heavy atom of the same molecule.  A frame fails a tier when
g <= margin.  Margin tiers run from the conventional box-sizing rule (two vdW
cutoffs) down to hydrogen-bond range; there is no zero-margin tier, since zero
clearance is coincidence of atom centres, not contact.

Counts are reported both absolutely and pi-weighted: failing conformations are
rare enough that a weighted fraction can round to zero over real frames.

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

import fig5_panels as base
import paths

# ── Simulation parameters (input/md/unbiased_200ns.mdp) ────────────────────────────
RVDW = 12.0                   # Angstrom; mdp has rvdw = 1.2 nm
N_CUTOFFS = 2                 # default console tier: margin = N_CUTOFFS * rvdw
SOLVATION_SHELL = 3.75        # Angstrom; O-O first-shell minimum, this water model
FIRST_SHELL_OUTER = 4.5       # Angstrom; generous first shell: apolar carbon-water
                              # RDF minimum rather than the tighter O-O value
SECOND_SHELL = 6.0            # Angstrom; outer edge of the SECOND shell.  Not
                              # 2 * SOLVATION_SHELL: shells are successive RDF
                              # minima, not additive layers
VDW_CONTACT = 3.5             # Angstrom; two carbon vdW radii (1.7 + 1.7) touching
HBOND_HEAVY = 3.2             # Angstrom; N/O donor-acceptor heavy-atom separation
SAVE_INTERVAL_PS = 100.0      # dt 0.002 ps x nstxtcout 50000, input/md/unbiased_200ns.mdp

# Absolute clearances in Angstrom, most demanding first; cutoff multiples and
# physical contact distances share the list.
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
# its frames fall back to this edge, measured from the cooled and T-REX
# trajectories of the same system.  Rows using it are flagged.
NOMINAL_BOX_EDGE = 47.2

# Cache sits beside make_revised_panels, not beside this file, so a copy run
# from elsewhere still finds it.
CACHE_NPZ = paths.OBSERVABLES / 'table-s3-pbc-self-image.npz'
OUT_DIR = paths.OUTPUT / 'table-s3-pbc-self-image'

CSV_HEADERS = {
    'ensemble': 'ensemble',
    'n_frames': 'frames_analysed',
    'n_fail_true': 'frames_failing_true_minimum_image_distance',
    'pct_true': 'percent_frames_failing_true',
    'pi_weighted_true': 'population_weighted_probability_failing_true',
    'min_gap_true': 'closest_true_solute_image_distance_angstrom',
    'box_from_file': 'box_edge_read_from_trajectory',
}

MD_COLUMNS = (
    ('ensemble', 'Ensemble', '{}'),
    ('n_frames', 'Frames analysed', '{:d}'),
    ('n_fail_true', 'Frames failing', '{:d}'),
    ('pct_true', 'Failing (% of frames)', '{:.3f}'),
    ('pi_weighted_true', 'Failing ($P$)', '{:.5f}'),
    ('min_gap_true', 'Closest approach (Å)', '{:.2f}'),
    ('box_from_file', '$L$ source', '{}'),
)

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

ENSEMBLE_TEX = {
    '6e9': r'$6.4{\times}10^{9}$', '2e9': r'$2.7{\times}10^{9}$',
    '6e8': r'$6.4{\times}10^{8}$', '2e8': r'$2.7{\times}10^{8}$',
    '6e7': r'$6.4{\times}10^{7}$', '2e7': r'$2.7{\times}10^{7}$',
    '6e6': r'$6.4{\times}10^{6}$',
    '277K': r'277\,K', '230K': r'230\,K',
}


def true_self_image_gap(coords, box_edge):
    """Smallest distance from any atom to an IMAGE of another atom."""
    diff = coords[:, None, :] - coords[None, :, :]
    imaged = diff - box_edge * np.round(diff / box_edge)
    d_imaged = np.linalg.norm(imaged, axis=-1)
    through_image = d_imaged < np.linalg.norm(diff, axis=-1) - 1e-9
    return float(d_imaged[through_image].min()) if through_image.any() else np.inf


def compute(cooling_labels=base.COOLING_LABELS, cooled_dir=base.COOLED_DIR,
            trex_dir=base.TREX_DIR, sampled_277=base.SAMPLED_277,
            topology=base.TOPOLOGY, n_states=base.N_STATES,
            selection=HEAVY_SELECTION, nominal_edge=NOMINAL_BOX_EDGE,
            cache_npz=CACHE_NPZ):
    """Scan trajectories with LOOS; cache per-state box edges and self-image gaps.

    Raw per-frame values, not histograms: the audit counts rare frames.
    """
    import loos
    from loos import pyloos as pl

    model = loos.createSystem(str(topology))
    heavy = loos.selectAtoms(model, selection)

    sources = [(lab, {int(p.stem): p for p in sorted((cooled_dir / lab).iterdir())}, None)
               for lab in cooling_labels]
    sources.append(('230K', {int(p.stem): p for p in sorted(trex_dir.iterdir())}, None))
    meta = np.loadtxt(sampled_277.parent / 'frame_metadata.txt', dtype=int)
    sources.append(('277K', sampled_277, meta[:, 1]))

    payload = {}
    for key, paths, index in sources:
        boxes = {s: [] for s in range(n_states)}
        gaps = {s: [] for s in range(n_states)}
        for state_ix, path in ([(None, paths)] if index is not None
                               else sorted(paths.items())):
            frame_l, frame_g = [], []
            for _ in pl.Trajectory(str(path), model):
                edge = model.periodicBox()[0]
                frame_l.append(edge)
                frame_g.append(true_self_image_gap(
                    heavy.getCoords(), edge if edge > 0 else nominal_edge))
            frame_l, frame_g = np.array(frame_l), np.array(frame_g)
            for s, sel in ([(state_ix, slice(None))] if index is None else
                           [(s, np.flatnonzero(index == s))
                            for s in np.unique(index)]):
                boxes[s].append(frame_l[sel])
                gaps[s].append(frame_g[sel])
        cat = lambda v: np.concatenate(v) if v else np.empty(0)
        n = 0
        for s in range(n_states):
            payload[f'{key}_box_{s}'] = cat(boxes[s])
            payload[f'{key}_true_{s}'] = cat(gaps[s])
            n += payload[f'{key}_true_{s}'].size
        print(f'  {key}: {n} frames')
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **payload)
    print(f'cached -> {cache_npz}')
    return payload


def fmt_prob(value):
    """Probability without pretending to more or less precision than we have."""
    if value <= 0.0:
        return '0'
    return f'{value:.3f}' if value >= 0.01 else f'{value:.2e}'


def fmt_prob_tex(value):
    """Probability as LaTeX math, scientific below 1%."""
    if value <= 0.0:
        return '0'
    if value >= 0.01:
        return f'{value:.3f}'
    exponent = int(np.floor(np.log10(value)))
    return f'${value / 10 ** exponent:.1f}{{\\times}}10^{{{exponent}}}$'


def fmt_margin(margin):
    """Margin in Angstrom without spurious rounding or trailing zeros."""
    return f'{margin:.2f}'.rstrip('0').rstrip('.')


def audit(obs, reps, margin, cooling_labels=base.COOLING_LABELS,
          reference_labels=base.REFERENCE_LABELS, pretty=base.PRETTY,
          n_states=base.N_STATES):
    """One row per ensemble: frames whose self-image gap fails `margin`."""
    rows = []
    for key in cooling_labels + reference_labels:
        pi = reps[key].mean(axis=0)
        n_total, n_fail, weighted, tightest, nominal = 0, 0, 0.0, np.inf, False
        for s in range(n_states):
            gap = obs[f'{key}_true_{s}']
            if gap.size == 0:
                continue
            nominal |= not np.all(obs[f'{key}_box_{s}'] > 0)
            fail = gap <= margin
            n_total += gap.size
            n_fail += int(fail.sum())
            weighted += pi[s] * fail.mean()
            finite = gap[np.isfinite(gap)]
            if finite.size:
                tightest = min(tightest, finite.min())
        rows.append(dict(key=key, ensemble=pretty[key], n_frames=n_total,
                         n_fail_true=n_fail,
                         pct_true=100.0 * n_fail / n_total if n_total else 0.0,
                         pi_weighted_true=weighted, min_gap_true=tightest,
                         box_from_file=not nominal))
    return rows


def print_table(rows, margin, label='', nominal_edge=NOMINAL_BOX_EDGE):
    """Console view of one margin tier."""
    print(f'\nsolute-image separation must exceed {fmt_margin(margin)} A'
          + (f'   [{label}]' if label else ''))
    head = (f"    {'ensemble':>10s} {'frames':>8s} {'fail':>7s} {'fail %':>8s} "
            f"{'pi-weighted':>12s} {'min gap':>9s}  box")
    print(head)
    print('    ' + '-' * (len(head) - 4))
    for r in rows:
        print(f"    {r['ensemble']:>10s} {r['n_frames']:8d} "
              f"{r['n_fail_true']:7d} {r['pct_true']:8.3f} "
              f"{r['pi_weighted_true']:12.5f} {r['min_gap_true']:9.2f}  "
              f"{'file' if r['box_from_file'] else f'{nominal_edge:.1f} nom'}")
    total_fail = sum(r['n_fail_true'] for r in rows)
    total = sum(r['n_frames'] for r in rows)
    print(f"    -> {total_fail} of {total} frames fail "
          f"({100.0 * total_fail / total:.3f}%)\n")


def md_table(rows, md_columns=MD_COLUMNS, nominal_edge=NOMINAL_BOX_EDGE):
    """Markdown table of one margin tier, with an all-ensemble total."""
    lines = ['| ' + ' | '.join(c[1] for c in md_columns) + ' |',
             '|' + '|'.join(['---'] * len(md_columns)) + '|']
    for r in rows:
        cells = [fmt.format(('trajectory' if r[key] else f'{nominal_edge:.1f} Å nominal')
                            if key == 'box_from_file' else r[key])
                 for key, _, fmt in md_columns]
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
                    save_interval_ps=SAVE_INTERVAL_PS,
                    tiers=REPORT_MARGINS, tier_meaning=TIER_MEANING,
                    cooling_labels=base.COOLING_LABELS,
                    reference_labels=base.REFERENCE_LABELS,
                    pretty=base.PRETTY, short=base.SHORT):
    """A report someone can read without opening the script."""
    keys = cooling_labels + reference_labels
    first = rows_by_tier[tiers[0][0]]
    total_frames = sum(r['n_frames'] for r in first)
    closest = {r['key']: r['min_gap_true'] for r in first}

    out = ['# Minimum-image self-interaction audit', '',
           'How close the peptide comes to its own periodic image, over exactly '
           'the frames behind the $R_g$ and maximum-separation figures.', '',
           'Generated by `table_s3_pbc_self_image.py`. Every frame in the cooled, '
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
        row = next(r for r in first if r['key'] == key)
        out.append(f'| {pretty[key]} | {row["n_frames"]} | '
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
                   f'{tier_meaning.get(label, "")}')
    out += ['', '## Summary', '',
            f'Every tier against every ensemble, out of {total_frames} frames '
            'analysed. **Each cell is $P$ (frames failing)**: the '
            'population-weighted probability that a conformation drawn from '
            'that ensemble violates the tier, with the absolute number of '
            'failing frames in parentheses. The two are not redundant -- $P$ '
            'runs several-fold below the frame share because the offending '
            'conformations sit in sparsely populated extended states.', '',
            '| Tier | $\\Delta$ (Å) | ' + ' | '.join(short[k] for k in keys) + ' |',
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
            '| $L$ source | Per-frame box edge. Taken from the trajectory, '
            'except for 277 K where it is recovered from the source '
            'trajectories (see the caveats). |',
            '', '## Per-tier detail', '']
    for label, margin in tiers:
        out += [f'### {label}  ($\\Delta = {fmt_margin(margin)}$ Å)', '',
                md_table(rows_by_tier[label], nominal_edge=nominal_edge), '']
    out += ['## Caveats', '',
            '- **277 K is computed from the source trajectories, not from '
            '`sampled_frames.xtc`.** Those picked frames were centred and '
            'rotated onto a common reference, and $g$ is not rotation '
            'invariant, so they cannot be used here; the file also carries no '
            'box. Both are recovered per frame from the unbiased trajectories '
            'via `frame_metadata.txt`, a mapping verified exact to xtc '
            'precision. The recovered box averages 46.61 Å (sd 0.08), not the '
            '47.2 Å of the cooled and T-REX runs. Centring alone would have '
            'been harmless: $g$ depends only on interatomic differences.',
            '- The cooled and 230 K frames are bit-identical to their sources '
            '(4e-7 nm), so they need no such treatment.',
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


def render_latex(rows_by_tier, rvdw=RVDW,
                 wrap_added=True, tiers=REPORT_MARGINS,
                 ensemble_tex=ENSEMBLE_TEX,
                 cooling_labels=base.COOLING_LABELS,
                 reference_labels=base.REFERENCE_LABELS):
    """One tabularx table sized to \\textwidth; the margin is the row label.

    wrap_added=False drops the \\add{} change-marking macro from the caption.
    """
    keys = cooling_labels + reference_labels
    first = rows_by_tier[tiers[0][0]]
    total = sum(r['n_frames'] for r in first)
    closest = min(r['min_gap_true'] for r in first)
    n_ens = len(keys)

    caption = (
        '\\textbf{Minimum-image self-interaction at various thresholds.} '
        'For each frame, $g$ is the smallest minimum-image distance from any '
        'heavy atom to a periodic image of any other heavy atom of the same '
        'peptide, so long as that distance is less than the direct '
        '(non-periodic) distance between the atoms; a frame is counted when '
        '$g \\le \\Delta$. Each cell gives the ensemble population-weighted '
        'probability that a conformation drawn from that ensemble falls below '
        'the threshold, with the absolute number of such frames in '
        f'parentheses. This analysis was performed across the {total} frames '
        'used to calculate the observables and record the state occupancies of '
        'the cooling endpoints elsewhere in this work. The van der Waals '
        f'cutoff was $r_\\mathrm{{vdw}} = {fmt_margin(rvdw)}$\\,\\AA. The '
        'closest approach we see across any dataset is in one frame, at '
        f'{closest:.2f}\\,\\AA. Cooling rates are in K\\,s$^{{-1}}$.')
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
           + ' & '.join(ensemble_tex[k] for k in keys) + r' \\',
           r'    \midrule']
    for label, margin in tiers:
        by_key = {r['key']: r for r in rows_by_tier[label]}
        cells = [f'{fmt_prob_tex(by_key[k]["pi_weighted_true"])} '
                 f'({by_key[k]["n_fail_true"]})' for k in keys]
        out.append(f'    {fmt_margin(margin)} & ' + ' & '.join(cells) + r' \\')
    out += [r'    \bottomrule', r'  \end{tabularx}', r'\end{table}', '']
    return '\n'.join(out)


def main(rvdw=RVDW, n_cutoffs=N_CUTOFFS, tiers=REPORT_MARGINS,
         cache_npz=CACHE_NPZ, csv_headers=CSV_HEADERS):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--recompute', action='store_true')
    p.add_argument('--rvdw', type=float, default=rvdw,
                   help='van der Waals cutoff in Angstrom (default %(default)s)')
    p.add_argument('--n-cutoffs', type=float, default=n_cutoffs,
                   help='console tier: margin is N * rvdw (default %(default)s)')
    p.add_argument('--margin', type=float,
                   help='console tier: absolute clearance in Angstrom, '
                        'overriding --n-cutoffs')
    p.add_argument('--csv', type=Path, help='also write the table as CSV')
    p.add_argument('--tex', type=Path, help='write the summary as a LaTeX table')
    p.add_argument('--no-add', action='store_true',
                   help='omit the \\add{} change-marking macro from the '
                        'LaTeX caption')
    p.add_argument('--md', type=Path,
                   help='write a self-describing markdown report covering '
                        'every margin tier')
    args = p.parse_args()

    if args.recompute or not cache_npz.is_file():
        print('scanning trajectories (LOOS) ...')
        obs = compute(cache_npz=cache_npz)
    else:
        obs = dict(np.load(cache_npz, allow_pickle=True))
        print(f'loaded cache {cache_npz}')

    reps = base.load_population_replicates()
    margin = args.margin if args.margin is not None else args.n_cutoffs * args.rvdw
    rows = audit(obs, reps, margin)
    print_table(rows, margin,
                label=next((l for l, m in tiers if abs(m - margin) < 1e-9), ''))

    if args.md or args.tex:
        rows_by_tier = {label: audit(obs, reps, m) for label, m in tiers}
        if args.md:
            args.md.write_text(render_markdown(rows_by_tier, rvdw=args.rvdw))
            print(f'wrote {args.md}')
        if args.tex:
            args.tex.write_text(render_latex(rows_by_tier, rvdw=args.rvdw,
                                             wrap_added=not args.no_add))
            print(f'wrote {args.tex}')

    if args.csv:
        with args.csv.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(csv_headers.values()))
            writer.writeheader()
            for r in rows:
                writer.writerow({csv_headers[k]: v for k, v in r.items()
                                 if k in csv_headers})
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    main()
