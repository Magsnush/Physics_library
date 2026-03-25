# Script that calls the Integration_functions_4D script and computes the difference between the structure functions with z_override at None and the structure functions at z_override at 0.0.
# This is to test whether the dipole distribution is the correct distribution at large Q^2 values. 

import os
import multiprocessing
import sys
import numpy as np

# Try importing the compute function using both relative and absolute paths so the
# script can be run from different working directories.
try:
    from Integration_functions_4D import compute_cross_section_4D
except Exception:
    from physicslib.numerics.totalDIS.LO.Integration_functions_4D import compute_cross_section_4D


def compute_F2(Q, xB, m, Zf, largeNc, r_max, mcpoints, n_cores=None, zlimit=False):
    """Compute F2 and its error for given kinematics and a z->0 override flag.

    Returns (F2, F2_err)
    """
    # Integration ranges (same defaults as LO_4D_integration_script)
    umin, umax = 1e-6, r_max
    upmin, upmax = 1e-6, r_max
    zmin, zmax = 1e-6, 1.0 - 1e-6
    thetamin, thetamax = 0.0, 2.0 * np.pi

    # Prefer SLURM_CPUS_PER_TASK in batch environments (Puhti/Slurm).
    if n_cores is None:
        try:
            n_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        except Exception:
            n_cores = multiprocessing.cpu_count()

    z_target_override = 0.0 if zlimit else None

    sigmaL, sigmaL_err = compute_cross_section_4D(
        Q=Q,
        xB=xB,
        m=m,
        Zf=Zf,
        polarization="L",
        largeNc=largeNc,
        umin=umin,
        umax=umax,
        upmin=upmin,
        upmax=upmax,
        zmin=zmin,
        zmax=zmax,
        thetamin=thetamin,
        thetamax=thetamax,
        mcpoints=mcpoints,
        n_cores=n_cores,
        z_target_override=z_target_override,
    )

    sigmaT, sigmaT_err = compute_cross_section_4D(
        Q=Q,
        xB=xB,
        m=m,
        Zf=Zf,
        polarization="T",
        largeNc=largeNc,
        umin=umin,
        umax=umax,
        upmin=upmin,
        upmax=upmax,
        zmin=zmin,
        zmax=zmax,
        thetamin=thetamin,
        thetamax=thetamax,
        mcpoints=mcpoints,
        n_cores=n_cores,
        z_target_override=z_target_override,
    )

    # DIS structure function prefactor
    alphaEM = 1.0 / 137.0
    prefactor = Q**2 / ((4.0 * np.pi**2) * alphaEM)

    FL = prefactor * sigmaL
    FL_err = prefactor * sigmaL_err
    FT = prefactor * sigmaT
    FT_err = prefactor * sigmaT_err

    F2 = FL + FT
    F2_err = np.sqrt(FL_err**2 + FT_err**2)
    return F2, F2_err


def main(argv=None):
    """Simple sys.argv-based CLI for a single Q value.

    Usage:
        python3 total_dp_dist_diff.py Q xB mcpoints r_max m Zf largeNc

    Where largeNc is 0 (off) or 1 (on). The script will auto-detect the number of cores
    (preferring SLURM_CPUS_PER_TASK on batch systems such as Puhti).
    """
    if argv is None:
        argv = sys.argv

    if len(argv) < 8:
        raise SystemExit(
            "Usage: total_dp_dist_diff.py Q xB mcpoints r_max m Zf largeNc [ncores]"
        )

    Q = float(argv[1])
    xB = float(argv[2])
    mcpoints = int(float(argv[3]))
    r_max = float(argv[4])
    m = float(argv[5])
    Zf = float(argv[6])
    largeNc = bool(int(argv[7]))

    # Example command line usage
    # python3 total_dp_dist_diff.py 1.0 0.1 1000 5.0 0.14 0.8165 0

    # We intentionally do not accept ncores from the command line. The
    # script will automatically choose the number of cores (SLURM_CPUS_PER_TASK
    # when available, otherwise all local CPUs).
    n_cores = None

    # Print CSV header
    print(
        "Q, xB, m, largeNc, mcpoints, r_max, F2_none, F2_none_err, F2_zero, F2_zero_err, F2_diff, F2_diff_err"
    )

    F2_none, F2_none_err = compute_F2(
        Q, xB, m, Zf, largeNc, r_max, mcpoints, n_cores=n_cores, zlimit=False
    )

    F2_zero, F2_zero_err = compute_F2(
        Q, xB, m, Zf, largeNc, r_max, mcpoints, n_cores=n_cores, zlimit=True
    )

    F2_zero *= 2
    F2_zero_err *= 2

    F2_diff = F2_none - F2_zero
    F2_diff_err = np.sqrt(F2_none_err**2 + F2_zero_err**2)

    print(
        f"{Q}, {xB}, {m}, {int(largeNc)}, {mcpoints}, {r_max}, {F2_none}, {F2_none_err}, {F2_zero}, {F2_zero_err}, {F2_diff}, {F2_diff_err}"
    )


if __name__ == "__main__":
    main()

