#!/usr/bin/env python
"""Regenerate the chi1 / Rg / Trp6-contact figure panels with statistical errors.

Self-contained replacement for cell 25 of figs-t250-k50.ipynb, written so the
revised figure and the data behind it can be submitted together.  Addresses the
referee comments on the submitted version (archived in ./submitted/):

  1. cooling rates hard to separate  -> optional residual (delta-from-277 K)
     strips under panels A and B; see --ab-variant
  2. no statistical errors           -> two-level bootstrap on every panel
  3. shaded vs saturated mismatch    -> one STYLE registry drives panels *and*
                                        legend, references always carry a
                                        saturated outline over their tint
  4. solid in A/B vs dashed in C/D   -> one encoding everywhere: coolings are
                                        solid lines, equilibrium references are
                                        tinted fills with saturated outlines
  5. unexplained panel-C inset       -> peak-height inset (a bin-width-dependent
                                        quantity) replaced by compactness vs
                                        cooling rate with 95% CIs
  6. "Asn" vs "ASN"                  -> residue labels title-cased at the source

Error model
-----------
Two nested sources, both resampled:

  * population uncertainty -- the bootstrap replicates of the state populations
    already used for the FES / dynamical-activity figures, read straight from
    cooling_bootstraps_k50_t250_lag_400.npz (`pi_samples` for 277 K,
    `cooled_<rate>` for each cooling rate, `pi_230K_samples` for 230 K).
  * within-state sampling uncertainty -- each state's conditional histogram is
    a multinomial draw over that state's frames.  This term dominates the
    population term by 4-6x and cannot be neglected.

Each cooled frame is the endpoint of an independent cooling run, so the
multinomial frame model is exact for the cooled ensembles.  The 230 K frames
come from replica-exchange trajectories and are time-correlated, so their frame
term is a lower bound.

Bootstrap replicates for different ensembles were drawn from independent RNG
streams, so differences between ensembles are treated as unpaired; the true
estimators are positively correlated, which makes the plotted difference bands
conservative (they over-cover).

Note on topology: built-protein.pdb carries no element records.  mdtraj infers
elements from atom names and reads the 19 HG* hydrogens as mercury and the 7
HE* hydrogens as helium, so mdtraj masses and its "sidechain-heavy" contact
scheme are both wrong on this file.  Rg and Trp6 contacts therefore go through
LOOS, matching the submitted figure.  chi1 comes from mdtraj, whose
compute_chi1 is atom-name-template based and unaffected.

Usage
-----
    python make_revised_panels.py                 # plot from cached observables
    python make_revised_panels.py --recompute     # re-read trajectories first
    python make_revised_panels.py --ab-variant residual
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── Model / dataset selection ────────────────────────────────────────────────
K_CHOICE = 50
TICA_LAG = 250
CHOSEN_LAG = 400
N_STATES = 50
FRAMES_PER_STATE_277 = 1000   # picked frames per state in sampled_frames.xtc
N_PI_DRAWS = 50               # distinct posterior draws of pi(277 K)
N_TPM_PER_PI = 100            # TPM draws stored per pi in P-samples-lag-400.npz
COOLING_LABELS = ['6e9', '6e8', '6e7', '6e6']       # '2e*' dropped for readability
REFERENCE_LABELS = ['277K', '230K']
PRETTY = {'6e9': '6.4e9 K/s', '2e9': '2.7e9 K/s', '6e8': '6.4e8 K/s',
          '2e8': '2.7e8 K/s', '6e7': '6.4e7 K/s', '2e7': '2.7e7 K/s',
          '6e6': '6.4e6 K/s', '277K': '277 K', '230K': '230 K'}
# Legend labels drop the unit -- six entries with "K/s" overrun the 8.7 cm
# column.  The panel-C inset axis carries the unit instead.
SHORT = {k: v.replace(' K/s', '') for k, v in PRETTY.items()}
RATE_VALUES = {'6e9': 6.4e9, '2e9': 2.7e9, '6e8': 6.4e8, '2e8': 2.7e8,
               '6e7': 6.4e7, '2e7': 2.7e7, '6e6': 6.4e6}

# ── Paths: all anchored at the repository root, see paths.py ─────────────────
import paths

BOOT_NPZ = paths.COOLING_BOOTSTRAPS
COOLED_DIR = paths.COOLED_BY_STATE
TREX_DIR = paths.TREX_230_BY_STATE
SAMPLED_277 = paths.SAMPLED_FRAMES
TOPOLOGY = paths.TOPOLOGY
OUT_DIR = paths.OUTPUT / 'fig5-panels'
OUTPUT_PREFIX = 'fig5-'
CACHE_NPZ = paths.OBSERVABLES / 'fig5-observables.npz'

# ── Observable definitions ───────────────────────────────────────────────────
ASN1_CHI1_IX = 0
TRP6_CHI1_IX = 5
CHI_BINWIDTH = np.pi / 12
CHI_BINS = np.arange(0, 2 * np.pi, CHI_BINWIDTH)
# Bins must span the DATA, not the display range: max Rg is 17.47 A (277 K) and
# the fastest coolings reach 17.06 A.  Stopping the bins at 11 A dropped up to
# 3.5% of the 6.4e9 weight, and since the inset divides by the in-range total it
# renormalised onto the truncated range and inflated frac(Rg < 8 A), worst for
# the rates with the most extended tail.  Display range is RG_XLIM.
RG_BINS = np.arange(7, 18.0, 0.125)
RG_XLIM = (7.0, 11.0)         # None shows the full measured range
RG_INSET_RECT = (0.52, 0.42, 0.44, 0.55)      # paper layout
RG_INSET_RECT_SLIDE = (0.55, 0.10, 0.40, 0.52)  # clears an upper-right legend
CONTACT_RADIUS = 4.5          # Angstrom, LOOS AtomicGroup.contactWith
CHI_PANEL_FILL = True         # fill under the equilibrium chi1 profiles in A/B
COMPACT_CUTOFF = 8.0          # Angstrom, "compact" threshold for the panel-C inset

# ── Bootstrap ────────────────────────────────────────────────────────────────
N_BOOT = 2000
CI_PERCENTILES = (2.5, 97.5)
RNG_SEED = 42

# ── Style: hues are fixed by the rest of the manuscript and must not change ──
COLORS = {'6e9': '#fde725', '2e9': '#90d743', '6e8': '#35b779', '2e8': '#21918c',
          '6e7': '#31688e', '2e7': '#443983', '6e6': '#440154',
          '277K': 'tab:red', '230K': 'tab:blue'}
# Reference fills are OPAQUE tints of their own hue, never alpha-blended: two
# translucent fills stacking into a third colour was a specific referee
# complaint.  Layering is by FIXED series order (see REFERENCE_DRAW_ORDER), so
# each ensemble keeps one colour across the whole axis; both saturated dashed
# outlines are drawn over the fills so the occluded profile stays traceable.
REFERENCE_TINT = 0.60         # how far the fill is blended toward white
BAND_ALPHA = 0.26             # CI ribbon fill on the cooling-rate lines
BAND_EDGE_LINEWIDTH = 0.35    # hairline edges so the ribbon reads over a fill
BAND_TINT_STEP = 0.22         # reference CI ribbon is this much darker than its fill
TINT_TOWARD_WHITE = 0.60      # legend swatch tinting, matches REFERENCE_TINT
REFERENCE_DRAW_ORDER = ['230K', '277K']  # back to front; FIXED, never per-x
REFERENCE_LINESTYLE = (0, (3.2, 1.4))   # dashed => equilibrium, in every panel
COOLING_LINESTYLE = 'solid'             # solid  => cooled, in every panel
LINEWIDTH = 1.0
OUTLINE_LINEWIDTH = 0.9
STROKE_IN = 0.01
FONT_SIZE = 8
LEGEND_FONTSIZE = 6.5
PANEL_LABEL_FONTSIZE = 12
INSET_FONTSIZE = 6
YLIM_HEADROOM = 0.08          # keep CI band tops off the axis ceiling
RESIDUAL_SHARED_YLIM = True   # one y-scale across all residual strips

# ── Geometry: the composite must not grow, the draft layout depends on it ────
SINGLE_COLWIDTH_CM = 8.7
CM_TO_INCH = 0.393701
COMPOSITE_HEIGHT_SCALE = 1.7
HEIGHT_RATIOS_OVERLAY = [0.55, 0.55, 0.90]                  # sums to 2.0
HEIGHT_RATIOS_RESIDUAL = [0.36, 0.19, 0.36, 0.19, 0.90]     # also sums to 2.0
WSPACE = 0.10

# ── Slide variant: one panel per 16:9 canvas, finer x ticks, bigger type ─────
SLIDE_WIDTH_CM = 25.4
SLIDE_HEIGHT_CM = SLIDE_WIDTH_CM * 9 / 16
SLIDE_FONT_SIZE = 11
SLIDE_LEGEND_FONTSIZE = 10
CHI_TICKS_FINE = [i * np.pi / 3 for i in range(7)]
CHI_TICKLABELS_FINE = ['0', r'$\pi/3$', r'$2\pi/3$', r'$\pi$',
                       r'$4\pi/3$', r'$5\pi/3$', r'$2\pi$']
RG_TICK_STEP_SLIDE = 1.0      # major; minors fall halfway between
CONTACT_TICK_STEP_SLIDE = 0.1


def tint(color, toward_white=TINT_TOWARD_WHITE):
    """Lighten a colour toward white without changing its hue assignment."""
    r, g, b = to_rgb(color)
    f = toward_white
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def format_axes(ax, minor_ticks_x=2, minor_ticks_y=2, stroke_in=STROKE_IN):
    """Tick/spine styling shared with the rest of the manuscript figures."""
    lw = stroke_in * 72
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(minor_ticks_x))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(minor_ticks_y))
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_color('black')
    ax.tick_params(which='major', width=lw, length=4.5, direction='in',
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(which='minor', width=lw * 0.75, length=2.25, direction='in',
                   top=True, bottom=True, left=True, right=True)
    return ax


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 1 — per-state conditional observables
# ═══════════════════════════════════════════════════════════════════════════

def wrap_angle_360(angles):
    two_pi = 2 * np.pi
    return angles - np.floor(angles / two_pi) * two_pi


def _loos_selections(topology=TOPOLOGY):
    import loos
    model = loos.createSystem(str(topology))
    trp = loos.selectAtoms(model, '!backbone && !hydrogen && resname == "TRP"')
    other = loos.selectAtoms(model, '!backbone && !hydrogen && resname != "TRP"')
    residues = other.splitByResidue()
    labels = [f'{r[0].resname().capitalize()} {r[0].resid()}' for r in residues]
    return model, trp, residues, labels


def _state_observables(frame_paths_by_state, topology=TOPOLOGY,
                       chi_bins=CHI_BINS, rg_bins=RG_BINS,
                       contact_radius=CONTACT_RADIUS, n_states=N_STATES,
                       xtc_state_index=None):
    """Per-state conditional histograms and contact fractions.

    frame_paths_by_state : dict state_ix -> Path of a single-state trajectory,
                           or, when xtc_state_index is given, a single Path
                           whose frames are labelled by xtc_state_index.
    """
    import loos
    import mdtraj as md
    from loos import pyloos as pl

    model, trp, residues, res_labels = _loos_selections(topology)
    n_chi, n_rg, n_res = len(chi_bins) - 1, len(rg_bins) - 1, len(residues)
    h_asn = np.zeros((n_states, n_chi))
    h_trp = np.zeros((n_states, n_chi))
    h_rg = np.zeros((n_states, n_rg))
    contacts = np.zeros((n_states, n_res))
    counts = np.zeros(n_states, dtype=int)

    if xtc_state_index is not None:
        items = [(None, frame_paths_by_state)]
    else:
        items = sorted(frame_paths_by_state.items())

    for state_ix, path in items:
        # --- LOOS: Rg and Trp6 sidechain contacts (element-safe) -------------
        traj = pl.Trajectory(str(path), model)
        rgs, contact_rows = [], []
        for _ in traj:
            rgs.append(model.radiusOfGyration())
            contact_rows.append([1 if r.contactWith(contact_radius, trp) else 0
                                 for r in residues])
        rgs = np.asarray(rgs)
        contact_rows = np.asarray(contact_rows, dtype=float)

        # --- mdtraj: chi1 (atom-name templates, element-independent) ---------
        chi1 = md.compute_chi1(md.load(str(path), top=str(topology)))[1]
        asn = wrap_angle_360(chi1[:, ASN1_CHI1_IX])
        trp6 = wrap_angle_360(chi1[:, TRP6_CHI1_IX])

        if xtc_state_index is None:
            targets = [(state_ix, slice(None))]
        else:
            targets = [(s, np.flatnonzero(xtc_state_index == s))
                       for s in np.unique(xtc_state_index)]

        for s, sel in targets:
            n = len(rgs[sel])
            if n == 0:
                continue
            counts[s] = n
            h_asn[s] = np.histogram(asn[sel], bins=chi_bins)[0] / n
            h_trp[s] = np.histogram(trp6[sel], bins=chi_bins)[0] / n
            h_rg[s] = np.histogram(rgs[sel], bins=rg_bins)[0] / n
            contacts[s] = contact_rows[sel].mean(axis=0)

    return dict(asn=h_asn, trp=h_trp, rg=h_rg, contact=contacts,
                counts=counts), res_labels


def sampled_277_states(sampled_277=None, n_states=N_STATES,
                       per_state=FRAMES_PER_STATE_277):
    """State index of every frame in sampled_frames.xtc.

    The picked-frame set is blocked by state -- `per_state` consecutive frames
    per state, in state order -- so the assignment is implied by the ordering
    and needs no sidecar file.  frame_metadata.txt is used when present, since
    it also carries source-trajectory provenance, but it is not indexed in
    directory_structure.txt and is not required.
    """
    sampled_277 = SAMPLED_277 if sampled_277 is None else Path(sampled_277)
    sidecar = sampled_277.parent / 'frame_metadata.txt'
    if sidecar.is_file():
        return np.loadtxt(sidecar, dtype=int)[:, 1]

    import mdtraj as md
    with md.open(str(sampled_277)) as fh:
        n_frames = len(fh)
    if n_frames != n_states * per_state:
        raise ValueError(
            f'{sampled_277.name} has {n_frames} frames; expected '
            f'{n_states} x {per_state}. The picked-frame set changed, so the '
            'implied state blocking is no longer valid -- supply '
            'frame_metadata.txt instead.')
    return np.repeat(np.arange(n_states), per_state)


def compute_observables(cooling_labels=COOLING_LABELS, cooled_dir=COOLED_DIR,
                        trex_dir=TREX_DIR, sampled_277=SAMPLED_277,
                        topology=TOPOLOGY, cache_npz=CACHE_NPZ):
    """Read every trajectory once and cache the per-state conditional stats."""
    payload, res_labels = {}, None

    for lab in cooling_labels:
        paths = {int(p.stem): p for p in sorted((cooled_dir / lab).iterdir())}
        obs, res_labels = _state_observables(paths, topology)
        payload.update({f'{lab}_{k}': v for k, v in obs.items()})
        print(f'  {lab}: {obs["counts"].sum()} frames over '
              f'{(obs["counts"] > 0).sum()} states')

    paths_230 = {int(p.stem): p for p in sorted(trex_dir.iterdir())}
    obs, res_labels = _state_observables(paths_230, topology)
    payload.update({f'230K_{k}': v for k, v in obs.items()})
    print(f'  230K: {obs["counts"].sum()} frames over '
          f'{(obs["counts"] > 0).sum()} states')

    obs, res_labels = _state_observables(sampled_277, topology,
                                         xtc_state_index=sampled_277_states(
                                             sampled_277))
    payload.update({f'277K_{k}': v for k, v in obs.items()})
    print(f'  277K: {obs["counts"].sum()} frames over '
          f'{(obs["counts"] > 0).sum()} states')

    payload['res_labels'] = np.array(res_labels)
    payload['chi_bins'] = CHI_BINS
    payload['rg_bins'] = RG_BINS
    payload['contact_radius'] = np.array(CONTACT_RADIUS)
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_npz, **payload)
    print(f'cached observables -> {cache_npz}')
    return payload


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 2 — bootstrap bands
# ═══════════════════════════════════════════════════════════════════════════

def load_population_replicates(boot_npz=BOOT_NPZ, cooling_labels=COOLING_LABELS):
    """Bootstrap replicates of the state populations, as used elsewhere."""
    z = np.load(boot_npz)
    reps = {lab: z[f'cooled_{lab}'] for lab in cooling_labels}
    # The 277 K population band rests on 50 posterior draws of pi.  The file
    # shipped in derived/ stores exactly those 50; the upstream
    # P-samples-derived file stores each of them tiled 100x (one row per TPM
    # draw at fixed pi), so de-tile when we are handed that shape instead.
    pi = z['pi_samples']
    reps['277K'] = pi if pi.shape[0] == N_PI_DRAWS else pi[::N_TPM_PER_PI]
    reps['230K'] = z['pi_230K_samples']
    return reps


def bootstrap_band(pi_replicates, conditional, counts, n_boot=N_BOOT,
                   ci=CI_PERCENTILES, seed=RNG_SEED, reduce_fn=None):
    """Two-level bootstrap of a population-weighted per-state observable.

    conditional : (n_states, n_bins) rows that each sum to 1 for occupied states
    counts      : (n_states,) frames behind each row
    reduce_fn   : optional callable mapping the weighted vector to a scalar
    """
    rng = np.random.default_rng(seed)
    occupied = np.flatnonzero(counts > 0)
    draws = []
    for _ in range(n_boot):
        pi = pi_replicates[rng.integers(len(pi_replicates))]
        resampled = np.zeros_like(conditional)
        for s in occupied:
            total = conditional[s].sum()
            if total <= 0:
                continue
            resampled[s] = rng.multinomial(counts[s], conditional[s] / total) \
                / counts[s] * total
        value = pi @ resampled
        draws.append(reduce_fn(value) if reduce_fn else value)
    draws = np.asarray(draws)
    return draws.mean(axis=0), np.percentile(draws, ci[0], axis=0), \
        np.percentile(draws, ci[1], axis=0), draws


def bootstrap_contacts(pi_replicates, contact_fractions, counts,
                       n_boot=N_BOOT, ci=CI_PERCENTILES, seed=RNG_SEED):
    """Same two-level scheme for per-residue contact fractions (binomial)."""
    rng = np.random.default_rng(seed)
    occupied = np.flatnonzero(counts > 0)
    draws = np.empty((n_boot, contact_fractions.shape[1]))
    for b in range(n_boot):
        pi = pi_replicates[rng.integers(len(pi_replicates))]
        resampled = np.zeros_like(contact_fractions)
        for s in occupied:
            p = np.clip(contact_fractions[s], 0.0, 1.0)
            resampled[s] = rng.binomial(counts[s], p) / counts[s]
        draws[b] = pi @ resampled
    return draws.mean(axis=0), np.percentile(draws, ci[0], axis=0), \
        np.percentile(draws, ci[1], axis=0), draws


def unpaired_difference(draws_a, draws_b, ci=CI_PERCENTILES):
    """Conservative difference band between two independently drawn ensembles."""
    mean = draws_a.mean(axis=0) - draws_b.mean(axis=0)
    sd = np.sqrt(draws_a.var(axis=0, ddof=1) + draws_b.var(axis=0, ddof=1))
    z = 1.959963984540054
    return mean, mean - z * sd, mean + z * sd


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 — drawing primitives (one encoding, used by every panel)
# ═══════════════════════════════════════════════════════════════════════════

def draw_reference_pair(ax, x, bands, reference_labels=REFERENCE_LABELS,
                        colors=COLORS, reference_tint=REFERENCE_TINT,
                        lw=OUTLINE_LINEWIDTH, linestyle=REFERENCE_LINESTYLE,
                        fill=True):
    """Both equilibrium ensembles, drawn so their fills never blend.

    Opaque tints layered in a FIXED series order (REFERENCE_DRAW_ORDER, back to
    front).  The occlusion order must not depend on x: picking the front series
    per-x by which profile is lower makes the base of the plot swap hue at every
    crossing, which reads as the data going discontinuous.  With a fixed order
    each ensemble keeps one colour everywhere, the front silhouette is whole,
    and the back ensemble shows wherever it rises above the front one.  Both
    saturated dashed outlines are drawn over everything, so the occluded
    profile is still traceable.
    """
    if fill:
        for depth, key in enumerate(REFERENCE_DRAW_ORDER):
            ax.fill_between(x, 0, bands[key][0], linewidth=0, zorder=1 + depth,
                            color=tint(colors[key], reference_tint))
    for key in reference_labels:
        mean, lo, hi, _ = bands[key]
        # CI ribbon as a DARKER opaque tint of the same hue rather than an alpha
        # band: over an already-tinted fill a translucent band is invisible.
        ax.fill_between(x, lo, hi, linewidth=0, zorder=3,
                        color=tint(colors[key], reference_tint - BAND_TINT_STEP))
        ax.plot(x, mean, color=colors[key], linewidth=lw, linestyle=linestyle,
                dash_capstyle='round', zorder=6)


def draw_cooling(ax, x, mean, lo, hi, key, colors=COLORS, lw=LINEWIDTH,
                 band_alpha=BAND_ALPHA, with_band=True,
                 linestyle=COOLING_LINESTYLE):
    """Cooling rate: solid saturated line over a same-hue CI ribbon.

    The ribbon carries hairline edges in the series colour.  A translucent fill
    alone disappears where it crosses the tinted reference fills, and these are
    not decorative -- at the histogram peaks the ribbon is 5-14% of the peak
    height, so it has to be readable.
    """
    color = colors[key]
    if with_band:
        ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, linewidth=0,
                        zorder=4)
        for edge in (lo, hi):
            ax.plot(x, edge, color=color, linewidth=BAND_EDGE_LINEWIDTH,
                    alpha=0.85, zorder=4.5)
    ax.plot(x, mean, color=color, linewidth=lw, linestyle=linestyle, zorder=5)


def legend_handles(cooling_labels=COOLING_LABELS,
                   reference_labels=REFERENCE_LABELS, colors=COLORS,
                   reference_tint=REFERENCE_TINT):
    """Swatches built from the same style values the panels draw with."""
    handles = [Line2D([], [], color=colors[k], linewidth=LINEWIDTH,
                      linestyle=COOLING_LINESTYLE) for k in cooling_labels]
    handles += [Patch(facecolor=tint(colors[k], reference_tint),
                      edgecolor=colors[k], linewidth=OUTLINE_LINEWIDTH,
                      linestyle='--') for k in reference_labels]
    labels = [SHORT[k] for k in cooling_labels + reference_labels]
    return handles, labels


# ═══════════════════════════════════════════════════════════════════════════
#  Stage 3 — panels
# ═══════════════════════════════════════════════════════════════════════════

CHI_TICKS = [np.pi / 3, np.pi, 5 * np.pi / 3]
CHI_TICKLABELS = [r'$\pi/3$', r'$\pi$', r'$5\pi/3$']


def _chi_bands(obs, reps, field, cooling_labels=COOLING_LABELS, n_boot=N_BOOT):
    out = {}
    for key in cooling_labels + REFERENCE_LABELS:
        out[key] = bootstrap_band(reps[key], obs[f'{key}_{field}'],
                                  obs[f'{key}_counts'], n_boot=n_boot)
    return out


def panel_chi1(ax, obs, reps, field, title, cooling_labels=COOLING_LABELS,
               chi_bins=CHI_BINS, show_bands=True, bands=None, n_boot=N_BOOT):
    centers = (chi_bins[:-1] + chi_bins[1:]) / 2
    bands = bands if bands is not None else _chi_bands(obs, reps, field,
                                                       cooling_labels, n_boot)
    draw_reference_pair(ax, centers, bands, fill=CHI_PANEL_FILL)
    for key in cooling_labels:
        mean, lo, hi, _ = bands[key]
        draw_cooling(ax, centers, mean, lo, hi, key, with_band=show_bands)
    ax.set_xticks(CHI_TICKS)
    ax.set_xticklabels(CHI_TICKLABELS)
    ax.set_ylabel('frac / bin')
    ax.set_title(title)
    ceiling = max(band[2].max() for band in bands.values())
    ax.set_ylim(0, ceiling * (1 + YLIM_HEADROOM))
    format_axes(ax)
    return bands


def share_residual_ylim(axes, extremes, headroom=YLIM_HEADROOM):
    """Put every residual strip on one y-scale.

    Autoscaling each strip separately invites a direct visual comparison
    between panels that is only true by coincidence.  Lock them together so
    the comparison is valid by construction.
    """
    limit = max(extremes) * (1 + headroom)
    for ax in axes:
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(3, symmetric=True))
    return limit


def panel_chi1_residual(ax, bands, cooling_labels=COOLING_LABELS,
                        chi_bins=CHI_BINS):
    """Delta from the 277 K profile, with conservative unpaired CI bands.

    Returns the largest |CI bound| so callers can share one y-scale across
    strips; see share_residual_ylim.
    """
    centers = (chi_bins[:-1] + chi_bins[1:]) / 2
    ref_draws = bands['277K'][3]
    ax.axhline(0.0, color=COLORS['277K'], linewidth=OUTLINE_LINEWIDTH,
               linestyle=REFERENCE_LINESTYLE, zorder=3)
    extreme = 0.0
    for key in cooling_labels + ['230K']:
        mean, lo, hi = unpaired_difference(bands[key][3], ref_draws)
        ax.fill_between(centers, lo, hi, color=COLORS[key], alpha=BAND_ALPHA,
                        linewidth=0, zorder=4)
        ax.plot(centers, mean, color=COLORS[key], linewidth=LINEWIDTH, zorder=5)
        extreme = max(extreme, np.abs(lo).max(), np.abs(hi).max())
    share_residual_ylim([ax], [extreme])
    ax.set_xticks(CHI_TICKS)
    ax.set_xticklabels(CHI_TICKLABELS)
    ax.set_ylabel(r'$\Delta$ vs' '\n277 K', fontsize=FONT_SIZE - 1)
    format_axes(ax)
    return extreme


def panel_rg(ax, obs, reps, cooling_labels=COOLING_LABELS, rg_bins=RG_BINS,
             compact_cutoff=COMPACT_CUTOFF, n_boot=N_BOOT, with_inset=True,
             rg_xlim=RG_XLIM, inset_rect=RG_INSET_RECT):
    centers = (rg_bins[:-1] + rg_bins[1:]) / 2
    bands = {}
    for key in cooling_labels + REFERENCE_LABELS:
        bands[key] = bootstrap_band(reps[key], obs[f'{key}_rg'],
                                    obs[f'{key}_counts'], n_boot=n_boot)
    draw_reference_pair(ax, centers, bands, fill=True)
    for key in cooling_labels:
        mean, lo, hi, _ = bands[key]
        draw_cooling(ax, centers, mean, lo, hi, key)
    ax.set_xlabel(r'$R_g$, $\mathrm{\AA}$')
    ax.set_ylabel('fraction per bin')
    if rg_xlim is not None:
        ax.set_xlim(*rg_xlim)
    shown = slice(None) if rg_xlim is None else (
        (centers >= rg_xlim[0]) & (centers <= rg_xlim[1]))
    ceiling = max(band[2][shown].max() for band in bands.values())
    ax.set_ylim(0, ceiling * (1 + YLIM_HEADROOM))
    format_axes(ax)

    if with_inset:
        compact = lambda v: v[centers < compact_cutoff].sum() / v.sum()
        stats = {}
        for key in cooling_labels + REFERENCE_LABELS:
            m, lo, hi, _ = bootstrap_band(reps[key], obs[f'{key}_rg'],
                                          obs[f'{key}_counts'], n_boot=n_boot,
                                          reduce_fn=compact)
            stats[key] = (m, lo, hi)
        _rg_inset(ax, stats, cooling_labels, compact_cutoff,
                  inset_rect=inset_rect)
    return bands


def _rg_inset(ax, stats, cooling_labels, compact_cutoff,
              inset_rect=RG_INSET_RECT):
    """Compactness vs cooling rate, with the two equilibria as reference bands.

    Replaces the submitted peak-height inset, which reported a bin-width
    dependent quantity and was never referred to in the text.
    """
    ins = ax.inset_axes(inset_rect)
    rates = np.array([RATE_VALUES[lab] for lab in cooling_labels])
    for key in REFERENCE_LABELS:
        color = COLORS[key]
        m, lo, hi = stats[key]
        ins.axhspan(lo, hi, color=tint(color, REFERENCE_TINT), linewidth=0,
                    zorder=1)
        ins.axhline(m, color=color, linewidth=OUTLINE_LINEWIDTH * 0.8,
                    linestyle=REFERENCE_LINESTYLE, zorder=2)
    for lab, rate in zip(cooling_labels, rates):
        m, lo, hi = stats[lab]
        ins.errorbar(rate, m, yerr=[[m - lo], [hi - m]], fmt='o', ms=3.2,
                     color=COLORS[lab], ecolor=COLORS[lab], elinewidth=0.8,
                     capsize=1.5, zorder=3, markeredgewidth=0.4,
                     markeredgecolor='black')
    ins.set_xscale('log')
    ins.set_xlim(rates.min() / 4, rates.max() * 4)
    ins.set_ylabel(rf'frac $R_g\!<\!{compact_cutoff:.0f}\,\mathrm{{\AA}}$',
                   fontsize=INSET_FONTSIZE, labelpad=1.0)
    ins.tick_params(axis='both', labelsize=INSET_FONTSIZE, direction='in',
                    length=2.0, pad=1.0)
    ins.set_xticks([1e7, 1e9])
    ins.set_xticklabels([r'$10^{7}$', r'$10^{9}$'])
    ins.xaxis.set_minor_locator(ticker.NullLocator())
    ins.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ins.spines['top'].set_visible(False)
    ins.spines['right'].set_visible(False)
    ins.set_xlabel('K/s', fontsize=INSET_FONTSIZE, labelpad=0.5)
    ins.patch.set_alpha(0.0)
    return ins


def panel_contacts(ax, obs, reps, res_labels, cooling_labels=COOLING_LABELS,
                   n_boot=N_BOOT):
    stats = {}
    for key in cooling_labels + REFERENCE_LABELS:
        stats[key] = bootstrap_contacts(reps[key], obs[f'{key}_contact'],
                                        obs[f'{key}_counts'], n_boot=n_boot)
    y = np.arange(len(res_labels), dtype=float)

    # Same fixed-order layering as the line panels, for the same reason: which
    # series is in front must not vary row to row, or the bar base changes hue
    # depending on which ensemble happens to be larger for that residue.
    for depth, key in enumerate(REFERENCE_DRAW_ORDER):
        ax.barh(y, stats[key][0], left=0, height=0.78, zorder=1 + depth,
                color=tint(COLORS[key], REFERENCE_TINT), edgecolor=COLORS[key],
                linewidth=OUTLINE_LINEWIDTH * 0.7, linestyle='--')
    for key in REFERENCE_LABELS:
        mean, lo, hi, _ = stats[key]
        ax.errorbar(mean, y, xerr=np.vstack((mean - lo, hi - mean)),
                    fmt='none', ecolor=COLORS[key], elinewidth=0.7, capsize=1.2,
                    zorder=6)

    offsets = np.linspace(-0.26, 0.26, len(cooling_labels))
    for key, dy in zip(cooling_labels, offsets):
        mean, lo, hi, _ = stats[key]
        ax.errorbar(mean, y + dy, xerr=np.vstack((mean - lo, hi - mean)),
                    fmt='o', ms=1.8, color=COLORS[key], ecolor=COLORS[key],
                    elinewidth=0.7, capsize=0, zorder=3,
                    markeredgewidth=0.25, markeredgecolor='black')

    ax.set_yticks(y)
    ax.set_yticklabels(res_labels)
    ax.set_xlabel('sidechain contact\nwith Trp 6')
    ax.set_xlim(left=0)
    format_axes(ax)
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  Composites
# ═══════════════════════════════════════════════════════════════════════════

def save(fig, name, out_dir=OUT_DIR, prefix=OUTPUT_PREFIX):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg'):
        fig.savefig(out_dir / f'{prefix}{name}.{ext}', transparent=True)
    print(f'  wrote {out_dir}/{prefix}{name}.{{pdf,svg}}')


def build_composite(obs, reps, res_labels, ab_variant='overlay',
                    cooling_labels=COOLING_LABELS, n_boot=N_BOOT,
                    colwidth_cm=SINGLE_COLWIDTH_CM,
                    height_scale=COMPOSITE_HEIGHT_SCALE):
    """Full 4-panel composite at exactly the submitted figure's dimensions."""
    figsize = (colwidth_cm * CM_TO_INCH,
               colwidth_cm * height_scale * CM_TO_INCH)
    if ab_variant == 'residual':
        mosaic = [['A', '.'], ['Ares', '.'], ['B', '.'], ['Bres', '.'],
                  ['C', 'D']]
        ratios = HEIGHT_RATIOS_RESIDUAL
    else:
        mosaic = [['A', '.'], ['B', '.'], ['C', 'D']]
        ratios = HEIGHT_RATIOS_OVERLAY

    fig, axd = plt.subplot_mosaic(mosaic, figsize=figsize, layout='constrained',
                                  height_ratios=ratios,
                                  gridspec_kw={'wspace': WSPACE})
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.97))

    bands_asn = panel_chi1(axd['A'], obs, reps, 'asn', r'Asn 1 $\chi_1$',
                           cooling_labels, n_boot=n_boot)
    bands_trp = panel_chi1(axd['B'], obs, reps, 'trp', r'Trp 6 $\chi_1$',
                           cooling_labels, n_boot=n_boot)
    if ab_variant == 'residual':
        extremes = [panel_chi1_residual(axd['Ares'], bands_asn, cooling_labels),
                    panel_chi1_residual(axd['Bres'], bands_trp, cooling_labels)]
        if RESIDUAL_SHARED_YLIM:
            share_residual_ylim([axd['Ares'], axd['Bres']], extremes)
        axd['A'].tick_params(labelbottom=False)
        axd['B'].tick_params(labelbottom=False)
        axd['Ares'].tick_params(labelbottom=False)
    else:
        axd['A'].sharex(axd['B'])
        axd['A'].tick_params(labelbottom=False)

    panel_rg(axd['C'], obs, reps, cooling_labels, n_boot=n_boot)
    panel_contacts(axd['D'], obs, reps, res_labels, cooling_labels,
                   n_boot=n_boot)

    handles, labels = legend_handles(cooling_labels)
    fig.legend(handles, labels, loc='upper center', ncol=len(handles),
               fontsize=LEGEND_FONTSIZE, frameon=False,
               bbox_to_anchor=(0.5, 0.995), handlelength=1.0,
               handletextpad=0.4, columnspacing=0.6)

    panel_keys = ['A', 'B', 'C', 'D']
    for key in panel_keys:
        axd[key].text(-0.15, 0.98, key, transform=axd[key].transAxes,
                      fontsize=PANEL_LABEL_FONTSIZE, va='bottom', ha='right')
    return fig


def build_single_panels(obs, reps, res_labels, cooling_labels=COOLING_LABELS,
                        n_boot=N_BOOT, colwidth_cm=SINGLE_COLWIDTH_CM):
    """Each panel on its own canvas, for hand assembly with the PyMOL renders."""
    half_w = colwidth_cm * 0.5 * CM_TO_INCH
    chi_h = colwidth_cm * COMPOSITE_HEIGHT_SCALE * 0.55 / 2.0 * CM_TO_INCH
    bottom_h = colwidth_cm * COMPOSITE_HEIGHT_SCALE * 0.90 / 2.0 * CM_TO_INCH

    panels = (('asn', r'Asn 1 $\chi_1$', 'A-asn1-chi1'),
              ('trp', r'Trp 6 $\chi_1$', 'B-trp6-chi1'))
    for field, title, name in panels:
        fig, ax = plt.subplots(figsize=(half_w, chi_h), layout='constrained')
        panel_chi1(ax, obs, reps, field, title, cooling_labels, n_boot=n_boot)
        save(fig, name)
        plt.close(fig)

    # The with-residual singles are held open until both exist, so their strips
    # can share one y-scale even though they are separate figures.
    pending = []
    for field, title, name in panels:
        fig, (ax_main, ax_res) = plt.subplots(
            nrows=2, figsize=(half_w, chi_h), layout='constrained',
            height_ratios=[0.36 / 0.55, 0.19 / 0.55], sharex=True)
        bands = panel_chi1(ax_main, obs, reps, field, title, cooling_labels,
                           n_boot=n_boot)
        extreme = panel_chi1_residual(ax_res, bands, cooling_labels)
        ax_main.tick_params(labelbottom=False)
        pending.append((fig, ax_res, extreme, name))
    if RESIDUAL_SHARED_YLIM:
        share_residual_ylim([p[1] for p in pending], [p[2] for p in pending])
    for fig, _, _, name in pending:
        save(fig, f'{name}-with-residual')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(half_w, bottom_h), layout='constrained')
    panel_rg(ax, obs, reps, cooling_labels, n_boot=n_boot)
    save(fig, 'C-rg')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(half_w, bottom_h), layout='constrained')
    panel_contacts(ax, obs, reps, res_labels, cooling_labels, n_boot=n_boot)
    save(fig, 'D-trp6-contacts')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(colwidth_cm * CM_TO_INCH, 0.35),
                           layout='constrained')
    ax.set_axis_off()
    handles, labels = legend_handles(cooling_labels)
    ax.legend(handles, labels, loc='center', ncol=len(handles),
              fontsize=LEGEND_FONTSIZE, frameon=False, handlelength=1.0,
              handletextpad=0.4, columnspacing=0.6)
    save(fig, 'legend')
    plt.close(fig)


def build_slide_panels(obs, reps, res_labels, cooling_labels=COOLING_LABELS,
                       n_boot=N_BOOT, width_cm=SLIDE_WIDTH_CM,
                       height_cm=SLIDE_HEIGHT_CM, font_size=SLIDE_FONT_SIZE,
                       chi_ticks=CHI_TICKS_FINE, chi_ticklabels=CHI_TICKLABELS_FINE,
                       rg_tick_step=RG_TICK_STEP_SLIDE,
                       contact_tick_step=CONTACT_TICK_STEP_SLIDE):
    """One panel per 16:9 canvas, for talks.

    Same numbers and same encoding as the paper panels; only the canvas shape,
    the type size and the x-tick density differ.  Panel C shows the full
    measured Rg range here rather than RG_XLIM, since there is room for the
    tail -- it runs out to 17.5 A.
    """
    size = (width_cm * CM_TO_INCH, height_cm * CM_TO_INCH)
    previous = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': font_size})

    def legend_on(ax):
        handles, labels = legend_handles(cooling_labels)
        ax.legend(handles, labels, ncol=len(handles), frameon=False,
                  fontsize=SLIDE_LEGEND_FONTSIZE, loc='upper right',
                  handlelength=1.4, handletextpad=0.5, columnspacing=1.0)

    for field, title, name in (('asn', r'Asn 1 $\chi_1$', 'A-asn1-chi1'),
                               ('trp', r'Trp 6 $\chi_1$', 'B-trp6-chi1')):
        fig, ax = plt.subplots(figsize=size, layout='constrained')
        panel_chi1(ax, obs, reps, field, title, cooling_labels, n_boot=n_boot)
        ax.set_xticks(chi_ticks)
        ax.set_xticklabels(chi_ticklabels)
        ax.set_xlabel(r'$\chi_1$, rad')
        legend_on(ax)
        save(fig, f'slide-{name}')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=size, layout='constrained')
    panel_rg(ax, obs, reps, cooling_labels, n_boot=n_boot, rg_xlim=None,
             inset_rect=RG_INSET_RECT_SLIDE)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(rg_tick_step))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    legend_on(ax)
    save(fig, 'slide-C-rg')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=size, layout='constrained')
    panel_contacts(ax, obs, reps, res_labels, cooling_labels, n_boot=n_boot)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(contact_tick_step))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.set_xlabel('sidechain contact with Trp 6')
    legend_on(ax)
    save(fig, 'slide-D-trp6-contacts')
    plt.close(fig)

    plt.rcParams.update({'font.size': previous})


def report_statistics(obs, reps, cooling_labels=COOLING_LABELS,
                      chi_bins=CHI_BINS, rg_bins=RG_BINS,
                      compact_cutoff=COMPACT_CUTOFF, n_boot=N_BOOT):
    """Numbers for the response letter: scalar summaries with 95% CIs."""
    chi_centers = (chi_bins[:-1] + chi_bins[1:]) / 2
    rg_centers = (rg_bins[:-1] + rg_bins[1:]) / 2
    gplus = (chi_centers >= 0) & (chi_centers < 2 * np.pi / 3)
    keys = cooling_labels + REFERENCE_LABELS

    print('\nAsn 1 chi1 gauche+ population, mean [95% CI]')
    asn_draws = {}
    for key in keys:
        m, lo, hi, d = bootstrap_band(reps[key], obs[f'{key}_asn'],
                                      obs[f'{key}_counts'], n_boot=n_boot,
                                      reduce_fn=lambda v: v[gplus].sum())
        asn_draws[key] = d
        print(f'  {PRETTY[key]:>10s}  {m:.3f} [{lo:.3f}, {hi:.3f}]')

    print(f'\nfraction with Rg < {compact_cutoff:.0f} A, mean [95% CI]')
    rg_draws = {}
    for key in keys:
        m, lo, hi, d = bootstrap_band(
            reps[key], obs[f'{key}_rg'], obs[f'{key}_counts'], n_boot=n_boot,
            reduce_fn=lambda v: v[rg_centers < compact_cutoff].sum() / v.sum())
        rg_draws[key] = d
        print(f'  {PRETTY[key]:>10s}  {m:.3f} [{lo:.3f}, {hi:.3f}]')

    print('\nmean Rg (A), mean [95% CI]')
    for key in keys:
        m, lo, hi, _ = bootstrap_band(
            reps[key], obs[f'{key}_rg'], obs[f'{key}_counts'], n_boot=n_boot,
            reduce_fn=lambda v: (v * rg_centers).sum() / v.sum())
        print(f'  {PRETTY[key]:>10s}  {m:.3f} [{lo:.3f}, {hi:.3f}]')

    for name, draws in (('Asn 1 chi1 gauche+', asn_draws),
                        (f'frac Rg < {compact_cutoff:.0f} A', rg_draws)):
        print(f'\npairwise differences, {name} (unpaired, conservative)')
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                mean, lo, hi = unpaired_difference(draws[a], draws[b])
                verdict = 'resolved' if lo * hi > 0 else 'not resolved'
                print(f'  {PRETTY[a]:>10s} - {PRETTY[b]:<10s} '
                      f'{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]  {verdict}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recompute', action='store_true',
                        help='re-read all trajectories instead of using the cache')
    parser.add_argument('--ab-variant', choices=('overlay', 'residual', 'both'),
                        default='overlay',
                        help="panel A/B presentation; 'overlay' is the figure, "
                             "'residual' is kept for the SI / response letter")
    parser.add_argument('--n-boot', type=int, default=N_BOOT)
    parser.add_argument('--no-singles', action='store_true',
                        help='skip the individual per-panel files')
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--slide', action='store_true',
                        help='write ONLY the 16:9 one-panel-per-canvas versions '
                             '(the paper composites and singles are skipped)')
    args = parser.parse_args()

    plt.rcParams.update({'font.size': FONT_SIZE})

    if args.recompute or not CACHE_NPZ.is_file():
        print(f'computing per-state observables (LOOS + mdtraj) ...')
        obs = compute_observables()
    else:
        obs = dict(np.load(CACHE_NPZ, allow_pickle=True))
        print(f'loaded cached observables from {CACHE_NPZ}')

    res_labels = [str(s) for s in obs['res_labels']]
    reps = load_population_replicates(cooling_labels=COOLING_LABELS)

    report_statistics(obs, reps, n_boot=args.n_boot)
    if args.stats_only:
        return

    if args.slide:
        build_slide_panels(obs, reps, res_labels, n_boot=args.n_boot)
        return

    variants = (['overlay', 'residual'] if args.ab_variant == 'both'
                else [args.ab_variant])
    for variant in variants:
        fig = build_composite(obs, reps, res_labels, ab_variant=variant,
                              n_boot=args.n_boot)
        save(fig, f'4panel-{variant}')
        plt.close(fig)

    if not args.no_singles:
        build_single_panels(obs, reps, res_labels, n_boot=args.n_boot)


if __name__ == '__main__':
    main()
