#!/usr/bin/env python3
"""
Test script demonstrating the z-dependent kinematic bound feature.

Shows how providing xB parameter enforces Msq_qq <= W²·z·(1-z).
"""

import sys
from pathlib import Path

# Add Physics_code_library to path for imports
physics_lib_path = Path(__file__).parent.parent
sys.path.insert(0, str(physics_lib_path))

import numpy as np
from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D

def test_kinematic_bounds():
    """Test that kinematic bounds are enforced correctly."""
    
    # Setup parameters
    Q = 1.0  # Photon virtuality (GeV)
    m = 0.14  # Quark mass (GeV)
    Zf = np.sqrt(2.0/3.0)  # Quark charge
    xB = 1e-2  # Bjorken-x
    
    # Compute W² from kinematic relation
    W_squared = Q**2 * (1.0/xB - 1.0)
    print(f"Q² = {Q**2:.4f} GeV²")
    print(f"xB = {xB:.4e}")
    print(f"W² = Q²(1/xB - 1) = {W_squared:.4f} GeV²")
    print()
    
    # Integration bounds
    umin, umax = 1e-6, 1.0
    upmin, upmax = 1e-6, 1.0
    zmin, zmax = 0.01, 0.99
    thetamin, thetamax = 0.0, np.pi
    
    # The key difference: with xB provided, the integrand will enforce
    # Msq_qq <= W²·z·(1-z) during integration
    # For z=0.5 (maximum of z(1-z)), max Msq_qq = W²/4
    max_Msq_at_z_half = W_squared * 0.5 * 0.5
    print(f"Maximum allowed Msq_qq at z=0.5: {max_Msq_at_z_half:.4f} GeV²")
    print()
    
    # Test 1: WITHOUT kinematic bounds - Use a very small xB to approximate no bounds
    print("=" * 60)
    print("Test 1: WITH minimal kinematic bounds (xB=1e-6)")
    print("=" * 60)
    
    result_minimal_bounds = compute_cross_section_5D(
        Q=Q,
        xB=1e-6,  # Very small xB gives very large W², minimal bound effect
        m=m,
        Zf=Zf,
        polarization="T",
        largeNc=False,
        umin=umin, umax=umax,
        upmin=upmin, upmax=upmax,
        zmin=zmin, zmax=zmax,
        thetamin=thetamin, thetamax=thetamax,
        Msq_qqtilde_min=1e-3,
        Msq_qqtilde_max=1.0,  # Large upper bound
        mcpoints=1000,
        n_cores=1,
    )
    
    print(f"σ_T = {result_minimal_bounds[0]:.6e} ± {result_minimal_bounds[1]:.6e} GeV⁻²")
    print()
    
    # Test 2: WITH standard kinematic bounds (xB provided)
    print("=" * 60)
    print("Test 2: WITH kinematic bounds (xB={:.4e})".format(xB))
    print("=" * 60)
    print("Constraint: Msq_qq <= W²·z·(1-z)")
    print()
    
    result_with_bounds = compute_cross_section_5D(
        Q=Q,
        xB=xB,
        m=m,
        Zf=Zf,
        polarization="T",
        largeNc=False,
        umin=umin, umax=umax,
        upmin=upmin, upmax=upmax,
        zmin=zmin, zmax=zmax,
        thetamin=thetamin, thetamax=thetamax,
        Msq_qqtilde_min=1e-3,
        Msq_qqtilde_max=1.0,  # Same large upper bound
        mcpoints=1000,
        n_cores=1,
    )
    
    print(f"σ_T = {result_with_bounds[0]:.6e} ± {result_with_bounds[1]:.6e} GeV⁻²")
    print()
    
    # Compare results
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    
    if result_minimal_bounds[0] != 0:
        ratio = result_with_bounds[0] / result_minimal_bounds[0]
        print(f"Ratio (tighter bounds) / (minimal bounds) = {ratio:.4f}")
        print()
        print("With tighter kinematic bounds (larger xB), the integration domain")
        print("becomes more restricted, so the result should be smaller.")

    
    print()
    print("Summary:")
    print(f"  - Minimal bounds (xB=1e-6): {result_minimal_bounds[0]:.6e} GeV⁻²")
    print(f"  - Standard bounds (xB={xB}): {result_with_bounds[0]:.6e} GeV⁻²")
    print(f"  - Difference:               {abs(result_with_bounds[0] - result_minimal_bounds[0]):.6e} GeV⁻²")



if __name__ == "__main__":
    test_kinematic_bounds()
