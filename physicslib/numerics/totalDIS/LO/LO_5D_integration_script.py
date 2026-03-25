#!/usr/bin/env python3

"""
Command-line driver for integrating the 5D finite-energy DIS integrand.

Usage (Analytic dipole):
    LO_5D_integration_script.py Q mcpoints r_max m Zf largeNc zlimit

Usage (BK-evolved dipole):
    LO_5D_integration_script.py Q mcpoints r_max m Zf largeNc zlimit --bk Y

where
    Q        : photon virtuality (GeV)
    mcpoints : Monte Carlo points per VEGAS iteration
    r_max    : upper bound for u and up
    m        : quark mass (GeV)
    Zf       : quark charge factor
    largeNc  : 0 (finite Nc) or 1 (large Nc)
    zlimit   : 0 (no z->0 limit) or 1 (z->0 limit)
    --bk Y   : optional, use BK-evolved dipole with rapidity Y

Note: Polarizations (T and L) are computed automatically for both dipole types.
"""

import sys
import os
import multiprocessing

import numpy as np

from Integration_functions_5D import compute_cross_section_5D
from physicslib.multipole_models.MV_models.rcbk_adapter import RCBKData

import time



def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 9:
        raise SystemExit(
            "Usage: LO_5D_integration_script.py Q xB mcpoints r_max m Zf largeNc zlimit [--bk Y]"
        )

    Q = float(argv[1])
    xB = float(argv[2])  # This is used to compute the Msq_qqtilde integration limits
    mcpoints = int(float(argv[3]))
    r_max = float(argv[4])
    m = float(argv[5])
    Zf = float(argv[6])
    largeNc = bool(int(argv[7]))
    
    # Optional z→0 limit flag
    if len(argv) > 8:
        zlimit = bool(int(argv[8]))
    else:
        zlimit = False

    # Parse optional --bk flag for BK-evolved dipole
    bk_provider = None
    bk_Y = None

    if len(argv) > 9 and argv[8] == "--bk":
        if len(argv) < 10:
            raise SystemExit("--bk flag requires rapidity value: --bk Y")
        bk_Y = float(argv[9])
        
        # Load BK data from the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bk_file = os.path.join(script_dir, 'rcbk_mv_proton.dat')
        
        if not os.path.exists(bk_file):
            raise SystemExit(f"BK data file not found at {bk_file}")
        
        print(f"Loading BK data from: {bk_file}")
        bk_provider = RCBKData(bk_file, interp_on_logr=True, fill_value=0.5)
        print(f"✓ BK data loaded successfully (Y={bk_Y})")

    #start = time.perf_counter()

    # An example of the command line arguments would be:
    # python3 LO_5D_integration_script.py 10 0.01 1e5 10 0.14 0.8165 0 0
    # python3 LO_5D_integration_script.py 10 0.01 1e5 10 0.14 0.8165 0 0 --bk 2.0
    # where the last argument 0 is the flag for no z->0 limit and 1 is the flag for z->0 limit

    # Integration ranges
    umin, umax = 1e-6, r_max
    upmin, upmax = 1e-6, r_max
    zmin, zmax = 1e-6, 1.0 - 1e-6
    thetamin, thetamax = 0.0, 2.0 * np.pi
    Msq_qqtilde_min, Msq_qqtilde_max = m**2, (Q**2 / 4) *(1/xB - 1)

    n_cores = multiprocessing.cpu_count()

    z_target_override = 0.0 if zlimit else None

    sigmaL, sigmaL_err = compute_cross_section_5D(
        Q=Q,
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
        Msq_qqtilde_min=Msq_qqtilde_min,
        Msq_qqtilde_max=Msq_qqtilde_max,
        mcpoints=mcpoints,
        n_cores=n_cores,
        z_target_override=z_target_override,
        bk_provider=bk_provider,
        bk_Y=bk_Y,
    )

    # DIS structure function prefactor
    alphaEM = 1.0 / 137.0
    prefactor = Q**2 / ((4.0 * np.pi**2) * alphaEM)

    FL = prefactor * sigmaL
    FL_err = prefactor * sigmaL_err


    sigmaT, sigmaT_err = compute_cross_section_5D(
        Q=Q,
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
        Msq_qqtilde_min=Msq_qqtilde_min,
        Msq_qqtilde_max=Msq_qqtilde_max,
        mcpoints=mcpoints,
        n_cores=n_cores,
        z_target_override=z_target_override,
        bk_provider=bk_provider,
        bk_Y=bk_Y,
    )

    FT = prefactor * sigmaT
    FT_err = prefactor * sigmaT_err

    F2 = FL + FT
    F2_err = np.sqrt(FL_err**2 + FT_err**2) # Error propagation for F2

    # Print header indicating dipole type used
    dipole_type = f"BK (Y={bk_Y})" if bk_provider is not None else "Analytic MV"
    #print(f"Dipole: {dipole_type}")
    print("Q2, xB, m, largeNc, zlimit, FL, FL_err, FT, FT_err, F2, F2_err")
    print(f"{Q**2}, {xB}, {m}, {int(largeNc)}, {int(zlimit)}, {FL}, {FL_err}, {FT}, {FT_err}, {F2}, {F2_err}")


    #end = time.perf_counter()

    #print(f"Integration took {end - start:.2f} seconds")



if __name__ == "__main__":
    main()

