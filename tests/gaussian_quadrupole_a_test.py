"""
gaussian_quadrupole_a_test.py

Quick interactive test that builds GaussianQuadrupole models with different
dipole parameter configurations and plots the quadrupole (finite-Nc and
large-Nc) as a function of transverse separation.

Usage Examples
--------------

1. Run directly to plot both analytic and BK-evolved quadrupoles:
   
   python3 tests/gaussian_quadrupole_a_test.py
   
   This will:
   - Load BK data from tests/rcbk_mv_proton.dat (if available)
   - Plot analytic (solid lines) and BK-evolved (dashed lines) quadrupoles
   - Create separate figures for z=0.0 and z=0.5
   - Show multiple theta angles on each figure

2. Programmatic usage in your own code:
   
   from physicslib.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
   from physicslib.multipole_models.MV_models.dipole import Dipole
   from physicslib.multipole_models.MV_models.rcbk_adapter import RCBKData
   
   # Create analytic dipole
   dip = Dipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0)
   quad = GaussianQuadrupole(dip)
   
   # Evaluate analytic quadrupole
   rvals = np.linspace(0.01, 20.0, 300)
   Q_analytic = quad.quadrupole_polar(rvals, rvals, z=0.5, theta=0.0)
   
   # Load BK data and evaluate BK-evolved quadrupole
   bk = RCBKData('tests/rcbk_mv_proton.dat', interp_on_logr=True)
   
   # Call S_xy with bk provider
   S_bk = dip.S_xy(x, y, bk=bk, Y=2.0)  # BK-evolved
   S_analytic = dip.S_xy(x, y)            # Analytic (no bk parameter)

Run from the repository root or from the tests directory. The script will
attempt to import the local `physicslib` package; if not found it will add
the relative path used elsewhere in the repo.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from physicslib.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
    from physicslib.multipole_models.MV_models.dipole import Dipole
    from physicslib.multipole_models.MV_models.rcbk_adapter import RCBKData
except ModuleNotFoundError:
    # Try to add the local physicslib path (relative to this file)
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    physics_path = os.path.join(repo_root, "physicslib")
    if os.path.isdir(physics_path):
        sys.path.insert(0, physics_path)
        from multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole
        from multipole_models.MV_models.dipole import Dipole
        from multipole_models.MV_models.rcbk_adapter import RCBKData
    else:
        raise


def plot_quadrupole_for_configs(configs, rvals=None, zvals=(0.5, 0.1), theta_vals=(0.0,), bk_provider=None, bk_Y=2.0):
    """Evaluate and plot quadrupole_polar for a list of dipole configs.

    For each config and for each z in `zvals` this function creates one
    figure and plots multiple curves corresponding to different theta
    values supplied in `theta_vals`. Both analytic and BK-evolved dipoles
    are compared (if BK provider is supplied).

    Parameters
    ----------
    configs : list of dicts
        Each dict has keys (Qs0, gamma, ec, label).
    rvals : array-like of transverse separation values (used for u/up)
    zvals : iterable of z values to evaluate (one figure per z)
    theta_vals : iterable of theta angles (radians) to plot inside each figure
    bk_provider : RCBKData instance, optional
        If provided, BK-evolved quadrupole will be overlaid for comparison.
    bk_Y : float
        Rapidity to use for BK evaluation (default: 2.0).
    """

    if rvals is None:
        rvals = np.linspace(0.01, 10.0, 150)

    for cfg in configs:
        Qs0 = cfg.get("Qs0", np.sqrt(0.104))
        gamma = cfg.get("gamma", 1.0)
        ec = cfg.get("ec", 1.0)
        label = cfg.get("label", f"Qs0={Qs0:.3f}, g={gamma:.2f}")

        # Create the analytic (non-BK) dipole
        dip_analytic = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)
        quad_analytic = GaussianQuadrupole(dip_analytic)

        # Create BK-evolved dipole if provider is supplied
        quad_bk = None
        if bk_provider is not None:
            dip_bk = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)
            quad_bk = GaussianQuadrupole(dip_bk)

        # For each z produce a separate figure where theta lines are compared
        for z in zvals:
            plt.figure(figsize=(10, 6))
            for theta in theta_vals:
                # u == up == r (collinear configuration)
                u = rvals
                up = rvals

                # Plot analytic (non-BK) quadrupole
                Q_analytic = quad_analytic.quadrupole_polar(u, up, z, theta, largeNc=False)
                th_deg = np.degrees(theta)
                plt.plot(rvals, Q_analytic, linewidth=2, label=f"Analytic, θ={th_deg:.1f}°")

                # Plot BK-evolved quadrupole if provider is supplied
                if quad_bk is not None and bk_provider is not None:
                    try:
                        # Evaluate quadrupole with BK-evolved dipole by calling S_xy with bk provider
                        original_S_xy = dip_bk.S_xy
                        
                        def bk_S_xy_wrapper(x, y):
                            return original_S_xy(x, y, bk=bk_provider, Y=bk_Y)
                        
                        dip_bk.S_xy = bk_S_xy_wrapper
                        Q_bk = quad_bk.quadrupole_polar(u, up, z, theta, largeNc=False)
                        dip_bk.S_xy = original_S_xy  # restore
                        
                        # Flatten output if needed (shape issues from interpolation)
                        Q_bk = np.asarray(Q_bk).flatten()
                        
                        plt.plot(rvals, Q_bk, linewidth=2, linestyle="--", label=f"BK (Y={bk_Y}), θ={th_deg:.1f}°")
                    except Exception as e:
                        print(f"Warning: Could not evaluate BK quadrupole: {e}")

            plt.title(f"Gaussian quadrupole — {label} — z={z}")
            plt.xlabel(r" r (GeV^-1)")
            plt.ylabel(r"Quadrupole S^{(4)}")
            plt.legend(ncol=2, fontsize="small")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()


def main():
    # Some representative dipole configurations to explore
    configs = [
        {"Qs0": np.sqrt(0.104), "gamma": 1.0, "ec": 1.0, "label": "MV-like Qs0=sqrt(0.104), gamma=1"},
    ]

    rvals = np.linspace(0.0001, 20.0, 300)
    # choose a small set of z-values to create one figure per z
    zvals = [0.0001, 0.5]
    # thetas to plot inside each z-figure (in radians)
    theta_vals = np.linspace(0.0, 1.0 * np.pi, 5)

    # Load BK data if available
    bk_test_file = os.path.join(os.path.dirname(__file__), 'rcbk_mv_proton.dat')
    bk_provider = None
    if os.path.isfile(bk_test_file):
        try:
            bk_provider = RCBKData(bk_test_file, interp_on_logr=True, fill_value=np.nan)
            print(f"Loaded BK data from {bk_test_file}")
        except Exception as e:
            print(f"Could not load BK data: {e}")
    else:
        print(f"BK data file not found at {bk_test_file}. Plotting analytic dipole only.")

    plot_quadrupole_for_configs(configs, rvals=rvals, zvals=zvals, theta_vals=theta_vals, 
                                bk_provider=bk_provider, bk_Y=0.0)

    print("Finished plotting Gaussian quadrupole configurations. Close figures to exit.")
    plt.show()


if __name__ == "__main__":
    main()
