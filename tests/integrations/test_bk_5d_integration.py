#!/usr/bin/env python3
"""
Test: BK-evolved dipole in 5D VEGAS integration
Uses the existing rcbk_mv_proton.dat data file
"""

import os
import sys
from pathlib import Path

# Ensure repo root is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np

def test_analytic_dipole():
    """Test that analytic dipole works in 5D integration."""
    from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D
    
    print("\n" + "="*70)
    print("TEST 1: Analytic Dipole in 5D Integration")
    print("="*70)
    
    # Parameters
    Q = 10.0
    m = 0.14
    Zf = np.sqrt(6.0 / 9.0)  # quark charge
    
    try:
        sigma, sigma_err = compute_cross_section_5D(
            Q=Q,
            m=m,
            Zf=Zf,
            polarization="T",
            largeNc=False,
            umin=1e-6,
            umax=10.0,
            upmin=1e-6,
            upmax=10.0,
            zmin=0.01,
            zmax=0.99,
            thetamin=0.0,
            thetamax=2.0*np.pi,
            Msq_qq_min=0.1,
            Msq_qq_max=100.0,
            mcpoints=int(1e4),
            n_cores=None,
            z_target_override=None,
        )
        print(f"✓ SUCCESS: σ_T = {sigma:.6e} ± {sigma_err:.6e}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bk_dipole():
    """Test that BK-evolved dipole works in 5D integration."""
    from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D
    from small_x_physics.multipole_models.MV_models.rcbk_adapter import RCBKData
    
    print("\n" + "="*70)
    print("TEST 2: BK-Evolved Dipole in 5D Integration (Direct Parameter)")
    print("="*70)
    
    # Load BK data
    bk_file = os.path.join(script_dir, 'rcbk_mv_proton.dat')
    if not os.path.exists(bk_file):
        print(f"✗ SKIPPED: BK data file not found at {bk_file}")
        return None
    
    print(f"Loading BK data from: {bk_file}")
    try:
        bk_data = RCBKData(bk_file, interp_on_logr=True, fill_value=0.5)
        print(f"✓ BK data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load BK data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Parameters
    Q = 10.0
    m = 0.14
    Zf = np.sqrt(6.0 / 9.0)  # quark charge
    
    try:
        sigma, sigma_err = compute_cross_section_5D(
            Q=Q,
            m=m,
            Zf=Zf,
            polarization="T",
            largeNc=False,
            umin=1e-6,
            umax=10.0,
            upmin=1e-6,
            upmax=10.0,
            zmin=0.01,
            zmax=0.99,
            thetamin=0.0,
            thetamax=2.0*np.pi,
            Msq_qq_min=0.1,
            Msq_qq_max=100.0,
            mcpoints=int(1e4),
            n_cores=None,
            z_target_override=None,
            bk_provider=bk_data,
            bk_Y=2.0,  # Rapidity for BK evolution
        )
        print(f"✓ SUCCESS: σ_T = {sigma:.6e} ± {sigma_err:.6e}")
        return True
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bk_dipole_manual_wrapping():
    """Test BK-evolved dipole using manual wrapping (like gaussian_quadrupole_a_test.py)."""
    from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import compute_cross_section_5D
    from small_x_physics.multipole_models.MV_models.rcbk_adapter import RCBKData
    from small_x_physics.multipole_models.MV_models.dipole import Dipole
    
    print("\n" + "="*70)
    print("TEST 3: BK-Evolved Dipole in 5D Integration (Manual Wrapping)")
    print("="*70)
    
    # Load BK data
    bk_file = os.path.join(script_dir, 'rcbk_mv_proton.dat')
    if not os.path.exists(bk_file):
        print(f"✗ SKIPPED: BK data file not found at {bk_file}")
        return None
    
    print(f"Loading BK data from: {bk_file}")
    try:
        bk_data = RCBKData(bk_file, interp_on_logr=True, fill_value=0.5)
        print(f"✓ BK data loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load BK data: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Note: This test demonstrates the manual wrapping pattern used in gaussian_quadrupole_a_test.py
    # However, for 5D integration, the direct parameter passing (TEST 2) is cleaner and preferred
    print("ℹ Note: This test uses the same wrapping pattern as gaussian_quadrupole_a_test.py")
    print("  (The direct parameter passing in TEST 2 is the recommended approach for integration)")
    
    # Parameters
    Q = 10.0
    m = 0.14
    Zf = np.sqrt(6.0 / 9.0)  # quark charge
    bk_Y = 2.0
    
    try:
        # Create a dipole instance
        dip = Dipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0)
        
        # Store original S_xy method
        original_S_xy = dip.S_xy
        
        # Create wrapper that injects BK provider
        def bk_S_xy_wrapper(x, y):
            return original_S_xy(x, y, bk=bk_data, Y=bk_Y)
        
        # Replace S_xy with wrapper
        dip.S_xy = bk_S_xy_wrapper
        
        print(f"✓ Dipole wrapped with BK provider (Y={bk_Y})")
        
        # The integration framework will use the wrapped S_xy method
        # Note: compute_cross_section_5D creates its own Dipole instance internally,
        # so this manual wrapping approach requires modifying the internal dipole
        # which is why the direct bk_provider parameter (TEST 2) is cleaner
        
        print("ℹ Manual wrapping requires deeper integration changes; using TEST 2 approach instead")
        
        # Restore original
        dip.S_xy = original_S_xy
        
        print(f"✗ SKIPPED: Manual wrapping not applicable to compute_cross_section_5D")
        return None
        
    except Exception as e:
        print(f"✗ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "#"*70)
    print("# BK-Evolved Dipole in 5D Integration Tests")
    print("#"*70)
    
    results = {}
    
    # Test 1: Analytic
    results['analytic'] = test_analytic_dipole()
    
    # Test 2: BK (direct parameter)
    results['bk_direct'] = test_bk_dipole()
    
    # Test 3: BK (manual wrapping - demonstrates pattern from gaussian_quadrupole_a_test.py)
    results['bk_manual'] = test_bk_dipole_manual_wrapping()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASSED"
        else:
            status = "✗ FAILED"
        print(f"  {name:20s} : {status}")
    
    print()
    return all(v for v in results.values() if v is not None)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
