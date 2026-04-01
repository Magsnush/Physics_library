#!/usr/bin/env python3

"""
Command-line driver for integrating the 4D finite-energy DIS integrand.

Usage:
    LO_4D_integration_script.py Q xB mcpoints r_max m Zf pol largeNc

where
    Q        : photon virtuality (GeV)
    xB       : Bjorken-x
    mcpoints : Monte Carlo points per VEGAS iteration
    r_max    : upper bound for u and up
    m        : quark mass (GeV)
    Zf       : quark charge factor
    pol      : "T" or "L" or "TL"
    largeNc  : 0 (finite Nc) or 1 (large Nc)
"""

import sys
import multiprocessing

import numpy as np

from Integration_functions_4D import compute_cross_section_4D

import time



def main(argv=None):
    if argv is None:
        argv = sys.argv

    if len(argv) < 8:
        raise SystemExit(
            "Usage: LO_4D_integration_script.py Q xB mcpoints r_max m Zf largeNc zlimit"
        )

    Q = float(argv[1])
    xB = float(argv[2])
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


    #start = time.perf_counter()

    # An example of the command line arguments would be:
    # python3 LO_4D_integration_script.py 10 0.01 1e5 10 0.14 0.8165 0 0
    # where the last argument 0 is the flag for no z->0 limit and 1 is the flag for z->0 limit


    # Integration ranges
    umin, umax = 1e-6, r_max
    upmin, upmax = 1e-6, r_max
    zmin, zmax = 1e-6, 1.0 - 1e-6
    thetamin, thetamax = 0.0, 2.0 * np.pi

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

    # DIS structure function prefactor
    alphaEM = 1.0 / 137.0
    prefactor = Q**2 / ((4.0 * np.pi**2) * alphaEM)

    FL = prefactor * sigmaL
    FL_err = prefactor * sigmaL_err


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

    FT = prefactor * sigmaT
    FT_err = prefactor * sigmaT_err

    F2 = FL + FT
    F2_err = np.sqrt(FL_err**2 + FT_err**2) # Error propagation for F2

    print("Q2, xB, m, largeNc, zlimit, FL, FL_err, FT, FT_err, F2, F2_err")
    print(f"{Q**2}, {xB}, {m}, {int(largeNc)}, {int(zlimit)}, {FL}, {FL_err}, {FT}, {FT_err}, {F2}, {F2_err}")


    #end = time.perf_counter()

    #print(f"Integration took {end - start:.2f} seconds")



if __name__ == "__main__":
    main()

