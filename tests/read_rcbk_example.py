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
    from physicslib.multipole_models.MV_models.dipole import RCBDipole, BKEvolvedDipole
    from physicslib.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
except ModuleNotFoundError:
    # Try to add the local physicslib path (relative to this file)
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    physics_path = os.path.join(repo_root, "physicslib")
    if os.path.isdir(physics_path):
        sys.path.insert(0, physics_path)
        from multipole_models.MV_models.dipole import RCBDipole, BKEvolvedDipole
        from multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
    else:
        raise
except Exception as e:
    print("Failed to import rcbk/quadrupole helpers from physicslib:", e)
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
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        print(f"Datafile not found: {args.file}")
        sys.exit(1)

    print(f"Loading rcbk data from: {args.file}")
    dip = RCBDipole.from_file(args.file)
    r_vals = dip._rcbk.r_vals
    y_vals = dip._rcbk.y_vals
    print(f"Data grid: {len(y_vals)} Y points from {y_vals.min()} to {y_vals.max()}")
    print(f"          {len(r_vals)} r points from {r_vals.min():.3e} to {r_vals.max():.3e} (GeV^-1)")

    # Prepare plot
    plt.figure(figsize=(8, 6))

    for Y in args.Ys:
        # Evaluate N and S on the native r grid
        try:
            Nvals = dip._rcbk.N(Y, r_vals)
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

    # Optional: plot Gaussian quadrupole built from the BK-evolved dipole
    if args.plot_quad:
        print("Building BK-evolved dipole for quadrupole plotting...")
        # Use the BKEvolvedDipole adapter which exposes S_xy(x,y) using its internal Y
        bk = BKEvolvedDipole.from_file(args.file, Y=0.0)
        quad = GaussianQuadrupole(bk)

        # Use r-grid inside the available rcBK grid to avoid OOB interpolation
        r_min_grid = max(1e-12, float(r_vals.min()))
        r_max_grid = float(r_vals.max())
        rvals_q = np.linspace(r_min_grid, r_max_grid, 300)

        # Ensure output directory exists (use the same directory as --out)
        fig_dir = os.path.dirname(args.out) or '.'
        os.makedirs(fig_dir, exist_ok=True)

        # For each requested z (one figure per z), plot multiple curves corresponding to different Y values.
        # For each theta provided, create a separate figure where each curve is a different Y (this matches the
        # dipole plotting style where Ys are separate lines).
        for z in args.zvals:
            for theta in args.theta:
                plt.figure(figsize=(7, 5))
                for iY, Y in enumerate(args.Ys):
                    # set internal rapidity for BK-evolved dipole
                    bk.set_Y(float(Y))

                    # u == up == r (collinear configuration)
                    u = rvals_q
                    up = rvals_q

                    try:
                        Q_FNc = quad.quadrupole_polar(u, up, z, theta, largeNc=False)
                    except Exception as e:
                        # rcbk interpolator can fail for exact zero separations or edge values.
                        # Retry with a tiny offset to avoid log(0) issues.
                        eps = 1e-9
                        u_safe = u + eps
                        up_safe = up + eps
                        try:
                            Q_FNc = quad.quadrupole_polar(u_safe, up_safe, z, theta, largeNc=False)
                            print(f"Warning: quadrupole eval at z={z}, theta={theta}, Y={Y} required small eps fallback: {e}")
                        except Exception:
                            print(f"Failed to evaluate quadrupole at z={z}, theta={theta}, Y={Y}: {e}")
                            Q_FNc = np.full_like(rvals_q, np.nan)

                    # ensure a 1D array for plotting
                    Q_FNc = np.asarray(Q_FNc).squeeze()
                    if Q_FNc.ndim != 1:
                        try:
                            Q_FNc = Q_FNc.reshape(-1)
                        except Exception:
                            Q_FNc = np.full_like(rvals_q, np.nan)

                    # Label curves by Y to match dipole plotting convention
                    plt.plot(rvals_q, Q_FNc, label=f'Y={Y:g}')

                th_deg = np.degrees(theta)
                plt.title(f'rcBK-based quadrupole — z={z}, θ={th_deg:.1f}°')
                plt.xlabel('transverse separation r (GeV^-1)')
                plt.ylabel('Quadrupole S^{(4)}')
                plt.legend(ncol=2, fontsize='small')
                plt.grid(True)
                plt.tight_layout()

                outname = os.path.join(fig_dir, f'rcbk_quadrupole_z{z:g}_theta{th_deg:.0f}.png')
                plt.savefig(outname)
                print(f'Saved quadrupole figure to {outname}')
                plt.show()
                plt.close()


if __name__ == '__main__':
    main()
