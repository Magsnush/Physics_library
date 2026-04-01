"""
Comprehensive 4D vs 5D integration comparison with mandatory kinematic bounds.

This script compares 4D and 5D integration for FL structure function.
The 5D integration ALWAYS uses z-dependent kinematic bounds to enforce
physical consistency: Msq_qq <= W²·z·(1-z) where W² = Q²(1/xB - 1).
"""

import sys
import os

# Add parent directory to path so we can import small_x_physics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from small_x_physics.numerics.totalDIS.LO.Integration_functions_4D import compute_cross_section_4D
from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D


def compare_4d_vs_5d_FL(Q, xB_values, m=0.14, Zf=np.sqrt(2/3), mcpoints=1e6, n_cores=None):
    """
    Compare 4D and 5D integration for FL structure function across multiple xB values.
    
    The 5D integration ALWAYS enforces z-dependent kinematic bounds to ensure
    physical consistency of the finite-energy DIS formulation.
    
    Parameters
    ----------
    Q : float
        Photon virtuality (GeV)
    xB_values : array-like
        Array of Bjorken-x values to compute
    m : float
        Quark mass (GeV)
    Zf : float
        Quark charge factor
    mcpoints : int
        Monte Carlo points per integration
    n_cores : int, optional
        Number of cores for VEGAS
    
    Returns
    -------
    dict with keys:
        - 'xB_values': input xB values
        - '1/xB': computed 1/xB values
        - 'FL_4D': 4D longitudinal structure function values
        - 'FL_4D_err': 4D uncertainties
        - 'FL_5D': 5D longitudinal structure function values (with kinematic bounds)
        - 'FL_5D_err': 5D uncertainties
    """
    
    # Set VEGAS environment variables for better convergence in 5D
    # More iterations help stabilize the 5D integral
    os.environ.setdefault("VEGAS_WARM_NITN", "15")   # More warm-up iterations
    os.environ.setdefault("VEGAS_FULL_NITN", "25")   # More full iterations
    os.environ.setdefault("VEGAS_MIN_NEVAL_BATCH", "50000")  # Reasonable batch size
    
    # Common bounds
    r_max = 20.0
    zmin, zmax = 1e-6, 1.0 - 1e-6
    thetamin, thetamax = 0.0, 2.0 * np.pi
    
    print("="*90)
    print(f"4D vs 5D Comparison: FL Structure Function")
    print(f"Q = {Q} GeV, m = {m} GeV, MC points = {mcpoints:.0e}")
    print(f"5D Kinematic bounds: ALWAYS ENABLED (mandatory for physical consistency)")
    print(f"Computing for {len(xB_values)} values of xB between {xB_values.min():.1e} and {xB_values.max():.1e}")
    print(f"VEGAS tuning: WARM_NITN=15, FULL_NITN=25 (for better 5D stability)")
    print("="*90)
    
    results_4d = []
    results_5d = []
    
    for i, xB in enumerate(xB_values):
        print(f"\n[{i+1}/{len(xB_values)}] xB = {xB:.6e} (1/xB = {1/xB:.2f})")
        print("-" * 90)
        
        # Kinematic bounds for 5D
        Msq_min = m**2
        Msq_max = (Q**2) * (1.0/xB - 1.0)
        
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
        
        print(f"FL = {FL_4D:.6e} ± {FL_4D_err:.6e}  (rel. error: {FL_4D_err/FL_4D*100:.2f}%)")
        
        results_4d.append({
            'xB': xB,
            '1/xB': 1/xB,
            'FL': FL_4D,
            'FL_err': FL_4D_err,
        })
        
        # 5D Integration with mandatory kinematic bounds
        print("  5D: ", end="", flush=True)
        sig5D_L, err5D_L = compute_cross_section_5D(
            Q=Q, xB=xB, m=m, Zf=Zf, polarization="L",
            largeNc=False,
            umin=1e-6, umax=r_max, upmin=1e-6, upmax=r_max,
            zmin=zmin, zmax=zmax, thetamin=thetamin, thetamax=thetamax,
            Msq_qqtilde_min=Msq_min, Msq_qqtilde_max=Msq_max,
            mcpoints=int(mcpoints), n_cores=n_cores,
        )
        
        # Convert to FL structure function
        FL_5D = prefactor * sig5D_L
        FL_5D_err = prefactor * err5D_L
        
        print(f"FL = {FL_5D:.6e} ± {FL_5D_err:.6e}  (rel. error: {FL_5D_err/FL_5D*100:.2f}%)")
        
        results_5d.append({
            'xB': xB,
            '1/xB': 1/xB,
            'FL': FL_5D,
            'FL_err': FL_5D_err,
        })
        
        # Discrepancy
        disc = (FL_5D - FL_4D) / FL_4D * 100
        print(f"  Discrepancy: {disc:+.2f}%")
    
    # Prepare output
    results = {
        'xB_values': np.array(xB_values),
        '1/xB': np.array([r['1/xB'] for r in results_4d]),
        'FL_4D': np.array([r['FL'] for r in results_4d]),
        'FL_4D_err': np.array([r['FL_err'] for r in results_4d]),
        'FL_5D': np.array([r['FL'] for r in results_5d]),
        'FL_5D_err': np.array([r['FL_err'] for r in results_5d]),
    }
    
    return results


if __name__ == "__main__":
    # Example usage: compare 4D and 5D for FL structure function
    # across 5 xB values between 1e-3 and 1e-2
    
    Q = 1.0
    xB_values = np.logspace(-4, -2, 10)  # Reduced from 10 to 5 for faster testing
    mcpoints = 1e5  # Reduced from 1e6 for faster testing
    n_cores = None  # Use all available cores
    
    # Test WITH kinematic bounds enabled
    print("\n" + "="*90)
    print("4D vs 5D Comparison with Mandatory Kinematic Bounds")
    print("="*90 + "\n")
    
    results = compare_4d_vs_5d_FL(
        Q=Q, xB_values=xB_values, mcpoints=mcpoints, n_cores=n_cores
    )
    
    print("\n" + "="*90)
    print("SUMMARY TABLE (5D WITH MANDATORY KINEMATIC BOUNDS)")
    print("="*90)
    print(f"{'1/xB':<12} {'FL_4D':<20} {'FL_5D':<20} {'Discrepancy (%)':<15}")
    print("-"*90)
    for i in range(len(results['1/xB'])):
        disc = (results['FL_5D'][i] - results['FL_4D'][i]) / results['FL_4D'][i] * 100
        print(f"{results['1/xB'][i]:<12.1f} {results['FL_4D'][i]:<20.6e} {results['FL_5D'][i]:<20.6e} {disc:+<14.2f}")
    
    # Create single plot with three lines (but now only 4D and 5D with bounds)
    print("\n" + "="*90)
    print("GENERATING PLOT")
    print("="*90)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot 4D results
    ax.errorbar(
        results['1/xB'],
        results['FL_4D'],
        yerr=results['FL_4D_err'],
        fmt='o-',
        linewidth=2.5,
        markersize=9,
        capsize=6,
        capthick=2,
        label='4D Integration',
        color='steelblue',
        ecolor='steelblue',
        alpha=0.9,
    )
    
    # Plot 5D with kinematic bounds (mandatory)
    ax.errorbar(
        results['1/xB'],
        results['FL_5D'],
        yerr=results['FL_5D_err'],
        fmt='s-',
        linewidth=2.5,
        markersize=9,
        capsize=6,
        capthick=2,
        label='5D Integration (with kinematic bounds)',
        color='coral',
        ecolor='coral',
        alpha=0.9,
    )
    
    ax.set_xlabel('$1/x_B$', fontsize=14, fontweight='bold')
    ax.set_ylabel('$F_L$', fontsize=14, fontweight='bold')
    ax.set_title(f'FL Structure Function: 4D vs 5D with Mandatory Kinematic Bounds (Q={Q} GeV)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('figures/4d_vs_5d_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: figures/4d_vs_5d_comparison.png")
    plt.show()




