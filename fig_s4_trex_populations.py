#!/usr/bin/env python
"""T-REX state populations vs temperature, as a log10 heatmap.

Rebuilds the supplementary figure for the 50-state model: state index on x,
temperature on y, colour is log10 of the histogrammed state population at that
temperature.  Follows the recipe previously applied to the 100-state model,
with two changes:

  * the colour scale is log10(pi), not the natural log that np.log gives. Same
    picture, rescaled by 1/ln(10): the range runs about -5.1 .. -0.6 instead of
    -11.7 .. -1.3, and a unit step is now a decade.
  * the colorbar label actually appears.

Usage
-----
    python make_trex_population_heatmap.py
    python make_trex_population_heatmap.py --true-temperature-axis
    python make_trex_population_heatmap.py --natural-log     # reproduce the old scale
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

# ── Model / dataset selection ────────────────────────────────────────────────
K_CHOICE = 50
TICA_LAG = 250

# ── Paths: all anchored at the repository root, see paths.py ─────────────────
import paths

DTRAJ_DIR = paths.REPEX_DTRAJS
CACHE_NPZ = (paths.OBSERVABLES
             / f'fig-s4-trex-populations-k{K_CHOICE}-t{TICA_LAG}.npz')
OUT_DIR = paths.OUTPUT / 'fig-s4-trex-populations'
DATA_DIR = paths.OBSERVABLES
OUT_STEM = f'fig-s4-trex-log10-populations-k{K_CHOICE}-t{TICA_LAG}'

# ── Plot ─────────────────────────────────────────────────────────────────────
FIG_WIDTH_CM = 17.8           # double-column; this is a supplementary figure
FIG_HEIGHT_CM = 10.0
CM_TO_INCH = 0.393701
CMAP = 'viridis'
# States never visited at a given temperature.  'none' is matplotlib's default
# bad-colour, i.e. fully transparent, so those cells drop out of the image
# instead of being painted.  Set to a grey like '0.85' to make them explicit.
MASKED_COLOR = 'none'
STATE_TICK_STEP = 5
TEMP_TICK_STEP = 10
ONES_BASED_STATE_LABELS = True    # matches the submitted supplementary figure
COLORBAR_FRACTION = 0.030     # colorbar width as a fraction of the axes
COLORBAR_PAD = 0.015
COLORBAR_ASPECT = 28
FONT_SIZE = 9
STROKE_IN = 0.01


def format_axes(ax, stroke_in=STROKE_IN):
    """Spine/tick styling shared with the other revision figures."""
    lw = stroke_in * 72
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_color('black')
    ax.tick_params(which='major', width=lw, length=4.0, direction='out')
    return ax


def add_colorbar(ax, im, label, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD,
                 aspect=COLORBAR_ASPECT, stroke_in=STROKE_IN):
    """Attach a labelled colorbar to `ax` and return the Colorbar."""
    # fig.colorbar is layout-aware.  Axes from make_axes_locatable are not, so
    # under layout='constrained' their ticks and label get clipped off the edge.
    cbar = ax.get_figure().colorbar(im, ax=ax, fraction=fraction, pad=pad,
                                    aspect=aspect)
    # Label through the Colorbar, never Axes.set_label -- that is the
    # legend-label setter and fails silently.  cax.set_ylabel also works.
    cbar.set_label(label)
    cbar.outline.set_linewidth(stroke_in * 72)
    cbar.ax.tick_params(width=stroke_in * 72, length=3.0, direction='out')
    return cbar


def load_trex_populations(dtraj_dir=DTRAJ_DIR, k=K_CHOICE):
    """Histogram each temperature's discrete trajectory over the k states.

    Bincount over the reassigned dtraj, normalised by the number of frames at
    that temperature.
    """
    import h5py

    files = sorted(dtraj_dir.iterdir())
    temperatures = np.array([int(p.name[:3]) for p in files], dtype=int)
    populations = np.empty((len(files), k), dtype=float)
    n_frames = np.empty(len(files), dtype=int)
    for i, path in enumerate(files):
        # enspara's ra.load would also work, but each of these files holds a
        # single dataset 'arr_00' -- verified across all 90 -- so h5py reads
        # them byte-identically without the extra dependency.  The assertion
        # matters: a multi-row ragged array would make 'arr_00' a silent
        # truncation rather than the whole trajectory.
        with h5py.File(path, 'r') as fh:
            datasets = list(fh.keys())
            if datasets != ['arr_00']:
                raise ValueError(
                    f'{path.name} holds {datasets}, expected exactly '
                    "['arr_00']; use enspara.ra.load for ragged dtrajs.")
            assigns = fh['arr_00'][:]
        n_frames[i] = assigns.shape[0]
        populations[i] = np.bincount(assigns, minlength=k) / n_frames[i]
    return temperatures, populations, n_frames


def population_heatmap(temperatures, populations, natural_log=False,
                       true_temperature_axis=False, cmap=CMAP,
                       masked_color=MASKED_COLOR, k=K_CHOICE,
                       state_tick_step=STATE_TICK_STEP,
                       temp_tick_step=TEMP_TICK_STEP,
                       ones_based=ONES_BASED_STATE_LABELS,
                       fig_width_cm=FIG_WIDTH_CM, fig_height_cm=FIG_HEIGHT_CM):
    """Heatmap of log population: states on x, temperature on y."""
    log_fn = np.log if natural_log else np.log10
    label = r'$\ln \pi$' if natural_log else r'$\log_{10} \pi$'
    # Unvisited states are -inf under any log.  They MUST be masked rather than
    # left as -inf: matplotlib maps non-finite values to the colormap's "under"
    # colour, which is opaque and is the bottom of the scale, so an unvisited
    # state would be drawn identically to a genuinely very rare one.  Only NaN
    # and masked entries get the "bad" colour, whose default is transparent.
    with np.errstate(divide='ignore'):
        logp = np.ma.masked_invalid(log_fn(np.ma.masked_equal(populations, 0.0)))

    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(masked_color)

    fig, ax = plt.subplots(figsize=(fig_width_cm * CM_TO_INCH,
                                    fig_height_cm * CM_TO_INCH),
                           layout='constrained')
    offset = 1 if ones_based else 0

    if true_temperature_axis:
        # Real temperature on y.  The T-REX ladder is not evenly spaced, so this
        # and the ladder-index version below are genuinely different pictures.
        edges_y = np.concatenate([
            [temperatures[0] - (temperatures[1] - temperatures[0]) / 2],
            (temperatures[:-1] + temperatures[1:]) / 2,
            [temperatures[-1] + (temperatures[-1] - temperatures[-2]) / 2]])
        edges_x = np.arange(k + 1) + offset - 0.5
        im = ax.pcolormesh(edges_x, edges_y, logp, cmap=colormap,
                           shading='flat', rasterized=True)
        ax.set_ylabel('Temperature, K')
        ax.invert_yaxis()
    else:
        im = ax.imshow(logp, cmap=colormap, aspect='auto', interpolation='none',
                       extent=(offset - 0.5, k + offset - 0.5,
                               len(temperatures) - 0.5, -0.5))
        ax.set_yticks(np.arange(len(temperatures), step=temp_tick_step))
        ax.set_yticklabels(temperatures[::temp_tick_step])
        ax.set_ylabel('Temperature, K')

    ax.set_xticks(np.arange(offset, k + offset, step=state_tick_step))
    ax.set_xlabel(f'state index ({"ones" if ones_based else "zero"}-based)')
    add_colorbar(ax, im, label)
    format_axes(ax)
    return fig, im


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--recompute', action='store_true',
                        help='re-histogram the repex dtrajs instead of using '
                             'the cached population matrix')
    parser.add_argument('--natural-log', action='store_true',
                        help='use ln instead of log10 (the submitted scale)')
    parser.add_argument('--true-temperature-axis', action='store_true',
                        help='place rows at their actual temperature rather '
                             'than at their ladder index')
    args = parser.parse_args()

    plt.rcParams.update({'font.size': FONT_SIZE})

    if args.recompute or not CACHE_NPZ.is_file():
        temperatures, populations, n_frames = load_trex_populations()
        CACHE_NPZ.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE_NPZ, temperatures=temperatures,
                            populations=populations, n_frames=n_frames)
        print(f'histogrammed the dtrajs -> {CACHE_NPZ}')
    else:
        cached = np.load(CACHE_NPZ)
        temperatures = cached['temperatures']
        populations = cached['populations']
        n_frames = cached['n_frames']
        print(f'loaded cached populations from {CACHE_NPZ}')
    print(f'{len(temperatures)} temperatures, {temperatures.min()}-'
          f'{temperatures.max()} K, {K_CHOICE} states')
    print(f'frames per temperature: {n_frames.min()}-{n_frames.max()}')
    n_zero = int((populations == 0).sum())
    print(f'unvisited (state, temperature) cells: {n_zero} of {populations.size} '
          f'({100 * n_zero / populations.size:.1f}%), masked ({MASKED_COLOR})')
    nonzero = populations[populations > 0]
    print(f'log10 pi range: {np.log10(nonzero).min():.2f} to '
          f'{np.log10(nonzero).max():.2f}  '
          f'(ln: {np.log(nonzero).min():.2f} to {np.log(nonzero).max():.2f})')

    fig, _ = population_heatmap(temperatures, populations,
                                natural_log=args.natural_log,
                                true_temperature_axis=args.true_temperature_axis)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_STEM.replace('log10', 'ln') if args.natural_log else OUT_STEM
    if args.true_temperature_axis:
        stem += '-true-T'
    for ext in ('pdf', 'svg'):
        fig.savefig(OUT_DIR / f'{stem}.{ext}', transparent=True)
    print(f'wrote {OUT_DIR}/{stem}.{{pdf,svg}}')


if __name__ == '__main__':
    main()
