#!/usr/bin/env python3
"""Demo: compute Optical-Theorem longitudinal cross section with
an rcBK-evolved dipole and analytic MV dipole using OTIntegration."""

import os
import sys
import argparse
from pathlib import Path

# Ensure repo root (Physics_code_library) is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Demo: rcBK vs analytic dipole using OTIntegration")
    p.add_argument('--rcbk', required=True, help='Path to rcBK output datafile')
    p.add_argument('--Q', type=float, default=10.0, help='Photon virtuality Q (GeV)')
    p.add_argument('--xB', type=float, default=0.01, help='Bjorken x')
    p.add_argument('--m', type=float, default=0.14, help='Quark mass (GeV)')
    p.add_argument('--rmax', type=float, default=10.0, help='r_max for radial integral (GeV^-1)')
    p.add_argument('--sigma0', type=float, default=2.57 * 2 * 18.81, help='sigma0 parameter')
    p.add_argument('--Y', type=float, default=None, help='Rapidity Y to evaluate BK (if omitted, uses the BK instance default)')
    p.add_argument('--Ys', nargs='+', type=float, help='List of Y values to evaluate (overrides automatic selection)')
    p.add_argument('--polarization', choices=['T', 'L'], default='L')
    return p.parse_args()


def main():
    args = parse_args()

    rcbk_file = Path(args.rcbk)
    if not rcbk_file.exists():
        print(f"rcBK file not found: {rcbk_file}")
        raise SystemExit(1)

    # Local imports from package
    try:
        from small_x_physics.numerics.totalDIS.LO.OTIntegration import OT_integral
        from small_x_physics.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared
        from small_x_physics.multipole_models.MV_models.dipole import Dipole, BKEvolvedDipole
    except Exception as e:
        print("Failed to import required modules from physicslib:", e)
        raise

    # Prepare photon wavefunction object (single-flavor demo)
    quark_masses = [args.m]
    quark_charges = [np.sqrt(6.0 / 9.0)]  # example charge
    photon_wf_obj = LO_OT_PhotonWF_squared(quark_masses, quark_charges)

    # Create a callable that matches OTIntegrand expectation: wf(Q, r, z, flavor)
    def photon_wf(Q, r, z, flavor):
        if args.polarization == 'L':
            return photon_wf_obj.psi_L_squared(Q, r, z, flavor)
        return photon_wf_obj.psi_T_squared(Q, r, z, flavor)

    # Analytic MV dipole instance and callable
    Qs0 = np.sqrt(0.104)
    gamma = 1.0
    ec = 1.0
    analytic_dip = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)
    analytic_callable = analytic_dip.S_xy  # signature (x,y) -> S

    # BK-evolved dipole
    bk = BKEvolvedDipole.from_file(str(rcbk_file), Y=args.Y if args.Y is not None else 0.0)

    # Single-flavor run using OT_integral
    Q = args.Q
    m = args.m
    Zf = quark_charges[0]
    r_max = args.rmax

    # Determine Y values to sample
    if args.Ys is not None and len(args.Ys) > 0:
        Ys = np.array(args.Ys, dtype=float)
    else:
        # try to read available y grid from the rcBK data
        Ys = None
        try:
            y_vals = getattr(bk._rcbk_dipole._rcbk, 'y_vals', None)
            if y_vals is not None:
                Ys = np.linspace(float(y_vals.min()), float(y_vals.max()), 9)
        except Exception:
            Ys = None
        if Ys is None:
            Ys = np.linspace(0.0, 4.0, 9)

    print(f"Evaluating at Y values: {Ys}")
    sigma0 = args.sigma0

    # Now compute differential contribution dSigma/dr = ∫ dz [OT_integrand(r, Q, z)]
    from small_x_physics.integrands.totalDIS.LO.OTintegrand import OTIntegrand
    from scipy.integrate import quad
    import matplotlib.pyplot as plt

    # r grid (log-spaced to capture small-r behaviour)
    r_min = 1e-6
    r_vals = np.concatenate((np.logspace(np.log10(1e-6), np.log10(1e-2), 20),
                             np.linspace(1e-2, r_max, 120)))

    # build analytic integrand object
    analytic_integrand = OTIntegrand(quark_masses=quark_masses, photon_wf=photon_wf,
                                     sigma0=sigma0, dipole_model=analytic_callable,
                                     polarization=args.polarization)

    # compute analytic dSigma/dr by integrating over z for each r
    print('Computing dSigma/dr for analytic MV dipole...')
    dsig_dr_anal = []
    for r in r_vals:
        val, err = quad(lambda z: analytic_integrand.OT_integrand(r, Q, z, 0), 1e-6, 1.0 - 1e-6)
        dsig_dr_anal.append(val)
    dsig_dr_anal = np.array(dsig_dr_anal)

    # compute BK-evolved dSigma/dr for each Y
    dsig_dr_bk = np.zeros((len(Ys), len(r_vals)))
    for i, y in enumerate(Ys):
        print(f'Computing dSigma/dr for BK at Y={y:.4g} ...')
        dip_callable = bk.as_callable_S_xy(Y=float(y))
        integrand_i = OTIntegrand(quark_masses=quark_masses, photon_wf=photon_wf,
                                 sigma0=sigma0, dipole_model=dip_callable,
                                 polarization=args.polarization)
        for j, r in enumerate(r_vals):
            val, err = quad(lambda z: integrand_i.OT_integrand(r, Q, z, 0), 1e-6, 1.0 - 1e-6)
            dsig_dr_bk[i, j] = val

    # Plot: analytic + 4 Y-values (starting from Y=0 choose first 4 Ys)
    fig_dir = Path.cwd() / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale('log')
    ax.plot(r_vals, dsig_dr_anal, color='k', linestyle='--', label='analytic MV')

    # pick up to 4 Ys starting from the smallest (assumed Y=0 included)
    Ys_sorted = np.array(sorted(Ys))
    plot_Ys = Ys_sorted[:4]
    colors = plt.get_cmap('tab10').colors
    for idx, y in enumerate(plot_Ys):
        i = list(Ys_sorted).index(y)
        ax.plot(r_vals, dsig_dr_bk[i, :], label=f'rcBK Y={y:.3g}', color=colors[idx % len(colors)])

    ax.set_xlabel('r [GeV^-1]')
    ax.set_ylabel('dSigma/dr (arb. units)')
    ax.set_title(f'dSigma/dr vs r (Q={Q} GeV)')
    ax.legend()
    ax.grid(True, which='both', ls='--', alpha=0.5)

    outpng = fig_dir / f'rcbk_vs_analytic_Q{Q:g}_dsigdr_vs_r.png'
    fig.savefig(outpng, dpi=200, bbox_inches='tight')
    print(f'Saved plot to {outpng}')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
