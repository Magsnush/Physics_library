#!/usr/bin/env python3

"""
Speed comparison test: 4D vs 5D finite-energy LO DIS integration.

This script compares the integration time for the 4D case (fixed Msq)
vs the 5D case (varying Msq_qq) to benchmark the performance difference.

Usage:
    python3 LO_totalDIS_4D_vs_5D_speedtest.py [Q] [mcpoints] [r_max] [m] [Zf]

Default parameters:
    Q = 10 GeV
    mcpoints = 1e5
    r_max = 10
    m = 0.14 GeV
    Zf = 0.8165 (√(6/9) for light quark charge)
"""

import sys
import os
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physicslib.numerics.totalDIS.LO.Integration_functions_4D import compute_cross_section_4D
from physicslib.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D


def run_speedtest(Q=10.0, mcpoints=int(1e6), r_max=10.0, m=0.14, Zf=0.8165):
    """
    Run 4D and 5D integration and compare speeds.
    
    Parameters
    ----------
    Q : float
        Photon virtuality (GeV)
    mcpoints : int
        Monte Carlo points per VEGAS iteration
    r_max : float
        Upper bound for u and up
    m : float
        Quark mass (GeV)
    Zf : float
        Quark charge factor
    """
    
    print("=" * 70)
    print("Speed Test: 4D vs 5D Finite-Energy DIS Integration")
    print("=" * 70)
    print(f"\nParameters:")
    print(f"  Q        = {Q} GeV")
    print(f"  mcpoints = {mcpoints}")
    print(f"  r_max    = {r_max}")
    print(f"  m        = {m} GeV")
    print(f"  Zf       = {Zf}")
    print()
    
    # Integration bounds
    xB = 0.01  # Fixed xB for 4D case to compute Msq

    umin, umax = 1e-6, r_max
    upmin, upmax = 1e-6, r_max
    zmin, zmax = 1e-6, 1.0 - 1e-6
    thetamin, thetamax = 0.0, 2.0 * np.pi
    Msq_qqtilde_min, Msq_qqtilde_max = m**2,  (Q**2 /4) *(1/xB -1) 
    
    # For 4D case, use a fixed xB to compute Msq
    
    
    # ========== 4D Integration ==========
    print("-" * 70)
    print("Running 4D Integration (fixed Msq)...")
    print("-" * 70)
    
    start_4d = time.perf_counter()
    
    try:
        sigma_T_4d, sigma_T_err_4d = compute_cross_section_4D(
            Q=Q,
            xB=xB,
            m=m,
            Zf=Zf,
            polarization="T",
            largeNc=False,
            umin=umin,
            umax=umax,
            upmin=upmin,
            upmax=upmax,
            zmin=zmin,
            zmax=zmax,
            thetamin=thetamin,
            thetamax=thetamax,
            mcpoints=mcpoints,
            n_cores=None,
            z_target_override=None,
        )
        
        end_4d = time.perf_counter()
        time_4d = end_4d - start_4d
        
        # DIS structure function prefactor
        alphaEM = 1.0 / 137.0
        prefactor = Q**2 / ((4.0 * np.pi**2) * alphaEM)
        FT_4d = prefactor * sigma_T_4d
        FT_err_4d = prefactor * sigma_T_err_4d
        
        print(f"✓ 4D Integration completed in {time_4d:.2f} seconds")
        print(f"  FT = {FT_4d:.6e} ± {FT_err_4d:.6e}")
        
    except Exception as e:
        print(f"✗ 4D Integration failed: {e}")
        time_4d = None
        FT_4d = None
        FT_err_4d = None
    
    # ========== 5D Integration ==========
    print("\n" + "-" * 70)
    print("Running 5D Integration (varying Msq_qq)...")
    print("-" * 70)
    
    start_5d = time.perf_counter()
    
    try:
        sigma_T_5d, sigma_T_err_5d = compute_cross_section_5D(
            Q=Q,
            m=m,
            Zf=Zf,
            polarization="T",
            largeNc=False,
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
            n_cores=None,
            z_target_override=None,
        )
        
        end_5d = time.perf_counter()
        time_5d = end_5d - start_5d
        
        # DIS structure function prefactor
        alphaEM = 1.0 / 137.0
        prefactor = Q**2 / ((4.0 * np.pi**2) * alphaEM)
        FT_5d = prefactor * sigma_T_5d
        FT_err_5d = prefactor * sigma_T_err_5d
        
        print(f"✓ 5D Integration completed in {time_5d:.2f} seconds")
        print(f"  FT = {FT_5d:.6e} ± {FT_err_5d:.6e}")
        
    except Exception as e:
        print(f"✗ 5D Integration failed: {e}")
        time_5d = None
        FT_5d = None
        FT_err_5d = None
    
    # ========== Comparison ==========
    print("\n" + "=" * 70)
    print("Speed Comparison Results")
    print("=" * 70)
    
    if time_4d is not None and time_5d is not None:
        ratio = time_5d / time_4d
        print(f"\n4D Integration time: {time_4d:.2f} s")
        print(f"5D Integration time: {time_5d:.2f} s")
        print(f"Speed ratio (5D/4D): {ratio:.2f}x")
        print(f"\nIntegration dimension overhead: {(ratio - 1.0) * 100:.1f}% slower for 5D")
        
        if FT_4d is not None and FT_5d is not None:
            print(f"\n4D FT result: {FT_4d:.6e} ± {FT_err_4d:.6e}")
            print(f"5D FT result: {FT_5d:.6e} ± {FT_err_5d:.6e}")
            
            # Check relative difference
            if FT_4d != 0:
                rel_diff = abs(FT_5d - FT_4d) / abs(FT_4d) * 100
                print(f"Relative difference: {rel_diff:.2f}%")
    else:
        print("\nCould not complete speed comparison (one or both integrations failed)")
    
    print("\n" + "=" * 70)


def main(argv=None):
    if argv is None:
        argv = sys.argv
    
    # Parse command-line arguments
    Q = float(argv[1]) if len(argv) > 1 else 10.0
    mcpoints = int(float(argv[2])) if len(argv) > 2 else int(1e5)
    r_max = float(argv[3]) if len(argv) > 3 else 10.0
    m = float(argv[4]) if len(argv) > 4 else 0.14
    Zf = float(argv[5]) if len(argv) > 5 else 0.8165
    
    run_speedtest(Q=Q, mcpoints=mcpoints, r_max=r_max, m=m, Zf=Zf)


if __name__ == "__main__":
    main()
