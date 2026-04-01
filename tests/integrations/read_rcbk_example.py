#!/usr/bin/env python3
"""Example: read an rcbk output file and plot N(Y,r) and S(Y,r) for a few Y values.

Example usage:
    python3 read_rcbk_example.py --file rcbk_mv_proton.dat --plot-quad --Ys 0 4 8 12 --theta 0.0 --zvals 0.5 --out figures/temp_rcbk.png
    

The script expects the repo root (Physics_code_library) to be on PYTHONPATH; if not,
it will add the parent directory automatically.
"""
import os
import sys
import argparse

# Ensure repo root is importable
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import matplotlib.pyplot as plt

try:
    from small_x_physics.multipole_models.MV_models.rcbk_adapter import RCBKData
    from small_x_physics.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
    from small_x_physics.multipole_models.MV_models.dipole import Dipole
except ModuleNotFoundError:
    # Try to add the local physicslib path (relative to this file)
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    physics_path = os.path.join(repo_root, "physicslib")
    if os.path.isdir(physics_path):
        sys.path.insert(0, physics_path)
        from multipole_models.MV_models.rcbk_adapter import RCBKData
        from multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
        from multipole_models.MV_models.dipole import Dipole
    else:
        raise
except Exception as e:
    print("Failed to import rcbk/quadrupole helpers from small_x_physics:", e)
    print("Make sure you run this from the repository or that Physics_code_library is on PYTHONPATH.")
    raise


def parse_args():
    p = argparse.ArgumentParser(description="Read rcbk datafile and plot N(Y,r) and S(Y,r)")
    p.add_argument('--file', '-f', required=True, help='Path to rcbk output datafile')
    p.add_argument('--Ys', nargs='+', type=float, default=[0.0, 1.0, 2.0],
                   help='List of rapidity (Y) values to plot')
    p.add_argument('--out', '-o', default='rcbk_example.png', help='Output image filename')
    p.add_argument('--logr', action='store_true', help='Plot versus log(r) instead of r')
    p.add_argument('--plot-quad', action='store_true', help='Also plot Gaussian quadrupole using BK dipole')
    p.add_argument('--theta', nargs='+', type=float, default=[0.0],
                   help='List of theta angles in radians to plot for quadrupole (default: 0.0)')
    p.add_argument('--zvals', nargs='+', type=float, default=[0.5],
                   help='List of z values to evaluate quadrupole at (default: 0.5)')
    p.add_argument('--plot-vs-Msq', action='store_true', 
                   help='Plot BK dipole as function of M² using Y = log(W²+Q²) - log(M²+Q²)')
    p.add_argument('--Q', type=float, default=1.0, 
                   help='Photon virtuality Q² (GeV) for Y calculation (default: 1.0)')
    p.add_argument('--W-squared', type=float, default=10.0,
                   help='W² value (GeV²) for Y calculation (default: 10.0)')
    p.add_argument('--Msq-range', nargs=2, type=float, default=[0.01, 100.0],
                   help='Range of M² values to plot (default: 0.01 100.0)')
    return p.parse_args()


def plot_dipole_vs_Msq(rcbk_data, Q, W_squared, Msq_range, r_vals, output_dir='figures'):
    """
    Plot BK-evolved dipole S(r) as a function of M² using the analytical Y relation.
    
    Y = log(W² + Q²) - log(M² + Q²)
    
    Parameters
    ----------
    rcbk_data : RCBKData
        The BK-evolved dipole data object
    Q : float
        Photon virtuality (GeV)
    W_squared : float
        W² value (GeV²)
    Msq_range : tuple
        (Msq_min, Msq_max) for plotting
    r_vals : array
        Transverse separation values to evaluate dipole at
    output_dir : str
        Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    Msq_min, Msq_max = Msq_range
    Q_sq = Q**2
    
    # Create logarithmically spaced M² values
    Msq_vals = np.logspace(np.log10(Msq_min), np.log10(Msq_max), 50)
    
    # Compute Y for each M²
    Y_vals = np.log(W_squared + Q_sq) - np.log(Msq_vals + Q_sq)
    
    print(f"\nPlotting dipole vs M²:")
    print(f"  Q² = {Q_sq:.4f} GeV²")
    print(f"  W² = {W_squared:.4f} GeV²")
    print(f"  M² range: {Msq_min:.4e} to {Msq_max:.4e} GeV²")
    print(f"  Corresponding Y range: {Y_vals.min():.4f} to {Y_vals.max():.4f}")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: S(r) vs M² for different r values
    r_plot_vals = np.array([1.0, 2.0, 5.0, 10.0])  # GeV^-1
    
    for r in r_plot_vals:
        if r < r_vals.min() or r > r_vals.max():
            continue
        
        S_vals = []
        for Y in Y_vals:
            try:
                N = rcbk_data.N(Y, r)
                S = 1.0 - float(N)
                S_vals.append(S)
            except Exception as e:
                print(f"Warning: failed to evaluate at Y={Y:.4f}, r={r}: {e}")
                S_vals.append(np.nan)
        
        ax1.plot(Msq_vals, S_vals, 'o-', label=f'r={r:.3f} GeV⁻¹', markersize=4, linewidth=1.5)
    
    ax1.set_xlabel('$M^2$ (GeV²)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$S(r)$ = 1 - $N(Y,r)$', fontsize=12, fontweight='bold')
    ax1.set_title('BK Dipole S-matrix vs $M^2$', fontsize=12, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Y vs M²
    ax2.plot(Msq_vals, Y_vals, 'k-', linewidth=2)
    ax2.fill_between(Msq_vals, Y_vals - 0.1, Y_vals + 0.1, alpha=0.2)
    ax2.set_xlabel('$M^2$ (GeV²)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('$Y = \ln(W^2 + Q^2) - \ln(M^2 + Q^2)$', fontsize=12, fontweight='bold')
    ax2.set_title(f'Rapidity vs $M^2$ (W²={W_squared:.2f}, Q²={Q_sq:.2f})', 
                  fontsize=12, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'rcbk_vs_Msq.png')
    plt.savefig(outfile, dpi=150)
    print(f"Saved figure to {outfile}")
    plt.show()
    plt.close()


def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        print(f"Datafile not found: {args.file}")
        sys.exit(1)

    print(f"Loading rcbk data from: {args.file}")
    rcbk_data = RCBKData(args.file, interp_on_logr=True, fill_value=0.5)
    r_vals = rcbk_data.r_vals
    y_vals = rcbk_data.y_vals
    print(f"Data grid: {len(y_vals)} Y points from {y_vals.min()} to {y_vals.max()}")
    print(f"          {len(r_vals)} r points from {r_vals.min():.3e} to {r_vals.max():.3e} (GeV^-1)")

    # Prepare plot
    plt.figure(figsize=(8, 6))

    for Y in args.Ys:
        # Evaluate N and S on the native r grid
        try:
            Nvals = rcbk_data.N(Y, r_vals)
            # Ensure shape is (N_r,) for plotting (squeeze any leading singleton dim)
            Nvals = np.asarray(Nvals).squeeze()
        except Exception as e:
            print(f"Failed to interpolate at Y={Y}: {e}")
            continue
        Svals = 1.0 - Nvals
        if args.logr:
            x = np.log(r_vals)
            xlabel = 'log(r) [GeV^-1]'
        else:
            x = r_vals
            xlabel = 'r [GeV^-1]'

        plt.plot(x, Nvals, label=f'N(Y={Y:g})')
        plt.plot(x, Svals, '--', label=f'S(Y={Y:g})')

    plt.xscale('linear' if args.logr else 'log')
    plt.xlabel(xlabel)
    plt.ylabel('Amplitude')
    plt.title(f'rcBK data: {os.path.basename(args.file)}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved figure to {args.out}")

    # Optional: plot dipole as function of M² using analytical Y relation
    if args.plot_vs_Msq:
        fig_dir = os.path.dirname(args.out) or 'figures'
        plot_dipole_vs_Msq(
            rcbk_data, 
            Q=args.Q, 
            W_squared=args.W_squared,
            Msq_range=args.Msq_range,
            r_vals=r_vals,
            output_dir=fig_dir
        )

        # Optional: plot Gaussian quadrupole built from the BK-evolved dipole
    if args.plot_quad:
        print("Building BK-evolved dipole for quadrupole plotting...")
        print("Note: Quadrupole evaluation requires dipole.S_xy() interface with BK provider")
        print("This functionality requires additional setup with BKEvolvedDipole adapter")
        print("For now, use --plot-vs-Msq for BK dipole analysis")

if __name__ == '__main__':
    main()
