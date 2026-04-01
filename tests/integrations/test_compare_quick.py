#!/usr/bin/env python3
"""
Quick test of 4D vs 5D integration with kinematic bounds.

This is a faster version with fewer xB points for quick testing.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from small_x_physics.numerics.totalDIS.LO.Integration_functions_4D import compute_cross_section_4D
from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D


def quick_compare_4d_vs_5d_FL(Q, xB_values, m=0.14, Zf=np.sqrt(2/3), mcpoints=5e5, n_cores=None, use_kinematic_bounds=True):
    """
    Quick comparison of 4D vs 5D for FL structure function.
    
    Parameters
    ----------
    Q : float
        Photon virtuality (GeV)
    xB_values : array-like
        Array of xB values to compute
    m : float
        Quark mass (GeV)
    Zf : float
        Quark charge factor
    mcpoints : int
        Monte Carlo points per integration
    n_cores : int, optional
        Number of cores for VEGAS
    use_kinematic_bounds : bool
        If True, use z-dependent kinematic bounds in 5D
    """
    
    # Set VEGAS environment variables
    os.environ.setdefault("VEGAS_WARM_NITN", "15")
    os.environ.setdefault("VEGAS_FULL_NITN", "25")
    os.environ.setdefault("VEGAS_MIN_NEVAL_BATCH", "50000")
    
    # Common bounds
    r_max = 20.0
    zmin, zmax = 1e-6, 1.0 - 1e-6
    thetamin, thetamax = 0.0, 2.0 * np.pi
    
    bounds_label = "ENABLED (xB provided)" if use_kinematic_bounds else "DISABLED (xB=None)"
    
    print("="*90)
    print(f"4D vs 5D Comparison: FL Structure Function (FAST TEST)")
    print(f"Q = {Q} GeV, m = {m} GeV, MC points = {mcpoints:.0e}")
    print(f"Kinematic bounds: {bounds_label}")
    print(f"Computing for {len(xB_values)} xB values")
    print("="*90)
    
    results_4d = []
    results_5d = []
    
    for i, xB in enumerate(xB_values):
        print(f"\n[{i+1}/{len(xB_values)}] xB = {xB:.4e} (1/xB = {1/xB:6.1f})", flush=True)
        print("-" * 90)
        
        # Kinematic bounds for 5D
        Msq_min = m**2
        Msq_max = Q**2 * (1.0/xB - 1.0)
        
        # 4D Integration
        print("  4D: ", end="", flush=True)
        sig4D_L, err4D_L = compute_cross_section_4D(
            Q=Q, xB=xB, m=m, Zf=Zf, polarization="L",
            largeNc=False,
            umin=1e-6, umax=r_max, upmin=1e-6, upmax=r_max,
            zmin=zmin, zmax=zmax, thetamin=thetamin, thetamax=thetamax,
            mcpoints=int(mcpoints), n_cores=n_cores
        )
        
        # Convert to FL structure function
        alphaEM = 1.0 / 137.0
        prefactor = Q**2 / ((4.0 * np.pi**2) * alphaEM)
        FL_4D = prefactor * sig4D_L
        FL_4D_err = prefactor * err4D_L
        
        print(f"FL = {FL_4D:.6e} ± {FL_4D_err:.6e}  (rel.err: {FL_4D_err/FL_4D*100:.1f}%)")
        
        results_4d.append({'xB': xB, '1/xB': 1/xB, 'FL': FL_4D, 'FL_err': FL_4D_err})
        
        # 5D Integration
        print("  5D: ", end="", flush=True)
        sig5D_L, err5D_L = compute_cross_section_5D(
            Q=Q, xB=xB, m=m, Zf=Zf, polarization="L",
            largeNc=False,
            umin=1e-6, umax=r_max, upmin=1e-6, upmax=r_max,
            zmin=zmin, zmax=zmax, thetamin=thetamin, thetamax=thetamax,
            Msq_qqtilde_min=Msq_min, Msq_qqtilde_max=Msq_max,
            mcpoints=int(mcpoints), n_cores=n_cores,
        )
        
        FL_5D = prefactor * sig5D_L
        FL_5D_err = prefactor * err5D_L
        
        print(f"FL = {FL_5D:.6e} ± {FL_5D_err:.6e}  (rel.err: {FL_5D_err/FL_5D*100:.1f}%)")
        
        results_5d.append({'xB': xB, '1/xB': 1/xB, 'FL': FL_5D, 'FL_err': FL_5D_err})
        
        # Discrepancy
        disc = (FL_5D - FL_4D) / FL_4D * 100
        print(f"  Discrepancy: {disc:+.2f}%")
    
    # Summary table
    print("\n" + "="*90)
    print(f"SUMMARY TABLE ({bounds_label})")
    print("="*90)
    print(f"{'1/xB':<10} {'FL_4D':<18} {'FL_5D':<18} {'Discrepancy':<12} {'4D σ':<12}")
    print("-"*90)
    
    for i in range(len(results_4d)):
        disc = (results_5d[i]['FL'] - results_4d[i]['FL']) / results_4d[i]['FL'] * 100
        rel_err_4d = results_4d[i]['FL_err'] / results_4d[i]['FL'] * 100
        print(f"{results_4d[i]['1/xB']:<10.1f} "
              f"{results_4d[i]['FL']:<18.6e} "
              f"{results_5d[i]['FL']:<18.6e} "
              f"{disc:+<11.2f}% "
              f"{rel_err_4d:<11.2f}%")
    
    return results_4d, results_5d


if __name__ == "__main__":
    # Fast test with 3 xB values
    Q = 1.0
    xB_values = np.array([1e-3, 3e-3, 1e-2])
    mcpoints = 5e5
    n_cores = None
    
    print("\n" + "="*90)
    print("TEST 1: With z-dependent kinematic bounds (NEW FEATURE)")
    print("="*90)
    
    results_4d_with, results_5d_with = quick_compare_4d_vs_5d_FL(
        Q=Q, xB_values=xB_values, mcpoints=mcpoints, n_cores=n_cores, 
        use_kinematic_bounds=True
    )
    
    print("\n" + "="*90)
    print("TEST 2: Without kinematic bounds (ORIGINAL BEHAVIOR)")
    print("="*90)
    
    results_4d_without, results_5d_without = quick_compare_4d_vs_5d_FL(
        Q=Q, xB_values=xB_values, mcpoints=mcpoints, n_cores=n_cores, 
        use_kinematic_bounds=False
    )
    
    print("\n" + "="*90)
    print("COMPARISON: With vs Without Kinematic Bounds")
    print("="*90)
    print("Kinematic bounds in 5D integration provide analytically-exact constraint on Msq_qq.")
    print()
    for i in range(len(xB_values)):
        xB = xB_values[i]
        fl_with = results_5d_with[i]['FL']
        fl_without = results_5d_without[i]['FL']
        diff = (fl_with - fl_without) / fl_without * 100
        print(f"xB = {xB:.2e}: 5D with bounds = {fl_with:.6e}, "
              f"5D without bounds = {fl_without:.6e}, "
              f"Difference = {diff:+.2f}%")
