"""
Diagnostic script to analyze 5D integrand behavior and identify optimization opportunities.

Run this to understand:
1. Where the integrand peaks
2. Relative magnitudes across parameter space
3. Numerical stability issues
4. Suggested improved bounds
"""

import sys
import os

# Add parent directory to path so we can import small_x_physics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from small_x_physics.integrands.totalDIS.LO.integrand5D import LODISIntegrand5D
from small_x_physics.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.multipole_models.MV_models.dipole import Dipole
from small_x_physics.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
from small_x_physics.numerics.totalDIS.LO.Integration_functions_5D import QuadrupolePolarWrapper


def setup_integrand(Q, m, Zf, largeNc=False):
    """Setup a 5D integrand for analysis."""
    quark_masses = np.array([m])
    quark_charges = np.array([Zf])
    
    photon_wf = LO_FE_PhotonWF_squared(
        quark_masses=quark_masses,
        quark_charges=quark_charges,
    )
    
    Qs0 = np.sqrt(0.104)
    dipole_model = Dipole(Qs0=Qs0, gamma=1.0, ec=1.0)
    quad_model = GaussianQuadrupole(dipole_model)
    sigma0 = 2.57 * 2 * 18.81
    
    quadrupole_polar = QuadrupolePolarWrapper(
        quad_model=quad_model,
        largeNc=largeNc,
        z_override=None,
    )
    
    return LODISIntegrand5D(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_model,
        quadrupole_model=quadrupole_polar,
        polarization="T",
        largeNc=largeNc,
    )


def sample_integrand_grid(integrand, Q, m, xB, r_max, n_samples=10):
    """Sample the integrand on a grid to understand its behavior."""
    
    Msq_min = m**2
    Msq_max = (Q**2 / 4) * (1/xB - 1)
    
    # Sample points (log-spaced for M²)
    u_vals = np.logspace(np.log10(1e-6), np.log10(r_max), n_samples)
    up_vals = np.logspace(np.log10(1e-6), np.log10(r_max), n_samples)
    z_vals = np.linspace(1e-6, 1.0-1e-6, n_samples)
    theta_vals = np.linspace(0, 2*np.pi, n_samples)
    Msq_vals = np.logspace(np.log10(Msq_min), np.log10(Msq_max), n_samples)
    
    results = []
    
    print(f"Sampling integrand for Q={Q}, m={m}, xB={xB}")
    print(f"M² range: [{Msq_min:.4e}, {Msq_max:.4e}] GeV²")
    print()
    
    # Sample along different 1D slices
    print("1D slice along u (fixed at mid-points for other vars):")
    u_mid, up_mid, z_mid, theta_mid = r_max/2, r_max/2, 0.5, np.pi
    Msq_mid = np.sqrt(Msq_min * Msq_max)
    
    u_samples = []
    u_values = []
    for u in u_vals:
        try:
            val = integrand.FE_integrand(Q, Msq_mid, u, up_mid, z_mid, theta_mid, flavor=0)
            u_samples.append(val)
            u_values.append(u)
        except Exception as e:
            print(f"  Error at u={u}: {e}")
    
    if u_samples:
        valid_u_vals = np.array(u_values)
        valid_u_samples = np.array(u_samples)
        peak_idx = np.nanargmax(np.abs(valid_u_samples))
        print(f"  Peak at u={valid_u_vals[peak_idx]:.4e}, value={valid_u_samples[peak_idx]:.4e}")
        print(f"  Min/max: {np.nanmin(valid_u_samples):.4e} / {np.nanmax(valid_u_samples):.4e}")
    
    print()
    print("1D slice along M² (fixed at mid-points for other vars):")
    Msq_samples = []
    Msq_values = []
    for Msq in Msq_vals:
        try:
            val = integrand.FE_integrand(Q, Msq, u_mid, up_mid, z_mid, theta_mid, flavor=0)
            Msq_samples.append(val)
            Msq_values.append(Msq)
        except Exception as e:
            print(f"  Error at M²={Msq}: {e}")
    
    if Msq_samples:
        valid_Msq_vals = np.array(Msq_values)
        valid_Msq_samples = np.array(Msq_samples)
        peak_idx = np.nanargmax(np.abs(valid_Msq_samples))
        print(f"  Peak at M²={valid_Msq_vals[peak_idx]:.4e}, value={valid_Msq_samples[peak_idx]:.4e}")
        print(f"  Min/max: {np.nanmin(valid_Msq_samples):.4e} / {np.nanmax(valid_Msq_samples):.4e}")
    
    print()
    print("1D slice along z (fixed at mid-points for other vars):")
    z_samples = []
    z_values = []
    for z in z_vals:
        try:
            val = integrand.FE_integrand(Q, Msq_mid, u_mid, up_mid, z, theta_mid, flavor=0)
            z_samples.append(val)
            z_values.append(z)
        except Exception as e:
            print(f"  Error at z={z}: {e}")
    
    if z_samples:
        valid_z_vals = np.array(z_values)
        valid_z_samples = np.array(z_samples)
        peak_idx = np.nanargmax(np.abs(valid_z_samples))
        print(f"  Peak at z={valid_z_vals[peak_idx]:.4e}, value={valid_z_samples[peak_idx]:.4e}")
        print(f"  Min/max: {np.nanmin(valid_z_samples):.4e} / {np.nanmax(valid_z_samples):.4e}")


if __name__ == "__main__":
    # Test with a typical kinematics point
    Q = 1.0  # GeV
    m = 0.14  # GeV (light quark mass)
    Zf = np.sqrt(2.0/3.0)  # u-quark charge
    xB = 0.01
    r_max = 20.0
    
    integrand = setup_integrand(Q, m, Zf, largeNc=False)
    sample_integrand_grid(integrand, Q, m, xB, r_max, n_samples=15)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. If M² shows strong peak: Consider restricting bounds")
    print("2. If u/up show extended behavior: Log-transform may help")
    print("3. If z shows edge behavior: Check z-integration split")
    print("4. Numerical issues? Check photon wavefunction and dipole for stability")
