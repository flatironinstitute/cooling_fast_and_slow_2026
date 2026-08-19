#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import mdtraj as md


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(".").resolve()

TOPOLOGY = (
    PROJECT_ROOT
    / "input"
    / "md"
    / "topology"
    / "test.gro"
)

TRAJ_LIST_DIR = (
    PROJECT_ROOT
    / "input"
    / "cooling"
    / "endpoints"
    / "manifests"
)

ENDPOINT_XTC_ROOT = (
    PROJECT_ROOT
    / "input"
    / "cooling"
    / "endpoints"
    / "xtc"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "sasa"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# SETTINGS
# =============================================================================

COOLING_RATES = [
    "6e9",
    "6e8",
    "6e7",
    "6e6",
]

N_PROCS = min(
    8,
    cpu_count(),
)


# =============================================================================
# VALIDATE FIXED INPUTS
# =============================================================================

if not TOPOLOGY.is_file():
    raise FileNotFoundError(
        f"Topology not found:\n{TOPOLOGY}"
    )

if not TRAJ_LIST_DIR.is_dir():
    raise FileNotFoundError(
        f"Trajectory-list directory not found:\n{TRAJ_LIST_DIR}"
    )

if not ENDPOINT_XTC_ROOT.is_dir():
    raise FileNotFoundError(
        f"Endpoint-XTC root directory not found:\n{ENDPOINT_XTC_ROOT}"
    )


# =============================================================================
# RESIDUE METADATA
# =============================================================================

def discover_residues(topology_path):
    """
    Read residue sequence numbers and residue names from the topology.
    """

    traj = md.load(
        str(topology_path)
    )

    residues = list(
        traj.topology.residues
    )

    resseqs = np.array(
        [
            residue.resSeq
            for residue
            in residues
        ],
        dtype=int,
    )

    residue_names = np.array(
        [
            residue.name
            for residue
            in residues
        ],
        dtype=str,
    )

    return (
        resseqs,
        residue_names,
    )


# =============================================================================
# TRAJECTORY LIST HANDLING
# =============================================================================

def load_snapshot_paths(
    traj_list_path,
):
    """
    Read snapshot paths from a trajectory-list file.

    Absolute paths are used unchanged.

    Relative paths are interpreted relative to the project root.

    Rewritten manifests should therefore contain paths such as:

        input/cooling/endpoints/xtc/6e9/<snapshot>.xtc
    """

    paths = []

    with traj_list_path.open(
        "r"
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            path = Path(line)

            if not path.is_absolute():
                path = (
                    PROJECT_ROOT
                    / path
                )

            path = path.resolve()

            if not path.is_file():
                raise FileNotFoundError(
                    "Snapshot listed in "
                    f"{traj_list_path.name} "
                    "does not exist:\n"
                    f"{path}"
                )

            paths.append(
                str(path)
            )

    if not paths:
        raise RuntimeError(
            "No snapshot paths found in:\n"
            f"{traj_list_path}"
        )

    return paths


# =============================================================================
# SASA WORKER
# =============================================================================

def compute_sasa_for_snapshot(
    snapshot_path,
):
    """
    Calculate per-residue SASA for one snapshot file.

    If the file contains multiple frames, only the first frame is used,
    matching the original preprocessing behaviour.

    Returns
    -------
    sasa_res : ndarray, shape (n_residues,)
        Per-residue SASA in nm^2.
    """

    traj = md.load(
        snapshot_path,
        top=str(TOPOLOGY),
    )

    if traj.n_frames < 1:
        raise RuntimeError(
            "No frames found in:\n"
            f"{snapshot_path}"
        )

    # Preserve original behaviour:
    # one listed file = one snapshot, use frame 0.
    traj = traj[0]

    sasa = md.shrake_rupley(
        traj,
        mode="residue",
    )

    return np.asarray(
        sasa[0],
        dtype=float,
    )


# =============================================================================
# PROCESS ONE COOLING RATE
# =============================================================================

def process_cooling_rate(
    short,
    n_residues,
):
    traj_list_path = (
        TRAJ_LIST_DIR
        / f"end-{short}-traj-list.txt"
    )

    if not traj_list_path.is_file():
        raise FileNotFoundError(
            "Missing trajectory list:\n"
            f"{traj_list_path}"
        )

    snapshot_paths = (
        load_snapshot_paths(
            traj_list_path
        )
    )

    print()
    print("=" * 72)
    print(
        f"Cooling rate: {short}"
    )
    print(
        f"Trajectory list: {traj_list_path}"
    )
    print(
        f"Snapshots: {len(snapshot_paths)}"
    )
    print(
        f"Worker processes: {N_PROCS}"
    )
    print("=" * 72)

    with Pool(
        processes=N_PROCS
    ) as pool:

        sasa_rows = list(
            pool.imap(
                compute_sasa_for_snapshot,
                snapshot_paths,
                chunksize=50,
            )
        )

    sasa_all = np.vstack(
        sasa_rows
    )

    expected_shape = (
        len(snapshot_paths),
        n_residues,
    )

    if sasa_all.shape != expected_shape:
        raise RuntimeError(
            "Unexpected SASA array shape.\n"
            f"Expected: {expected_shape}\n"
            f"Observed: {sasa_all.shape}"
        )

    output_path = (
        OUTPUT_DIR
        / f"end-{short}-sasa-all.npy"
    )

    np.save(
        output_path,
        sasa_all,
    )

    print(
        f"Saved: {output_path}"
    )

    print(
        f"Shape: {sasa_all.shape}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        f"Project root:\n  {PROJECT_ROOT}"
    )

    print(
        f"\nTopology:\n  {TOPOLOGY}"
    )

    print(
        f"\nSASA output:\n  {OUTPUT_DIR}"
    )

    # -------------------------------------------------------------------------
    # Residue information
    # -------------------------------------------------------------------------

    resseqs_all, residue_names_all = (
        discover_residues(
            TOPOLOGY
        )
    )

    print(
        f"\nNumber of residues: "
        f"{len(resseqs_all)}"
    )

    print(
        "\nResidues:"
    )

    for resseq, name in zip(
        resseqs_all,
        residue_names_all,
    ):
        print(
            f"  {int(resseq):3d} {name}"
        )

    residue_info_path = (
        OUTPUT_DIR
        / "sasa_residue_info.npz"
    )

    np.savez(
        residue_info_path,
        resseqs_all=resseqs_all,
        residue_names_all=residue_names_all,
    )

    print(
        "\nSaved residue metadata:"
    )

    print(
        f"  {residue_info_path}"
    )

    # -------------------------------------------------------------------------
    # Cooling ensembles
    # -------------------------------------------------------------------------

    for short in COOLING_RATES:

        process_cooling_rate(
            short=short,
            n_residues=len(
                resseqs_all
            ),
        )

    print()
    print("=" * 72)
    print("SASA preprocessing complete.")
    print("=" * 72)

    print(
        "\nGenerated:"
    )

    print(
        f"  {residue_info_path.name}"
    )

    for short in COOLING_RATES:
        print(
            f"  end-{short}-sasa-all.npy"
        )


if __name__ == "__main__":
    main()