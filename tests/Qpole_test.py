"""
Quick comparison plot of finite-Nc vs large-Nc Gaussian quadrupole,
similar in spirit to the scans done in `physicslib/gaussianapprox.ipynb`.

We consider a simple "line" configuration:

  x1  fixed
  x2  moves along a ray with radius r
  x1' and x2' fixed

and plot Q_FNc(r) and Q_LNc(r) as functions of r.
"""

import numpy as np
import matplotlib.pyplot as plt

from small_x_physics.multipole_models.MV_models.dipole import Dipole
from small_x_physics.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole


def build_quadrupole():
    """
    Construct MV dipole + GaussianQuadrupole with parameters
    matching your typical DIS choices.
    """
    Qs0 = np.sqrt(0.104)  # saturation scale (GeV)
    gamma = 1.0
    ec = 1.0
    dip = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)
    quad = GaussianQuadrupole(dipole_model=dip)
    return quad


def line_configuration_scan():
    """
    Reproduce a simple line scan:

    - Fix x1 and (x1', x2') geometry
    - Vary x2 along a ray x2 = x1 + r * n_hat
    - Compare finite-Nc and large-Nc quadrupoles vs r
    """
    quad = build_quadrupole()

    # Base point and direction (similar to patterns in the notebook)
    x1 = np.array([1.0, 0.0])
    n_hat = np.array([1.0, 0.0])  # direction along +x

    # Fixed primed points
    x1p = np.array([0.0, 1.0])
    x2p = np.array([1.0, 1.0])

    rvals=np.linspace(1e-4,10)
    eps=np.array([1e-5,1e-5])

    x=np.array([1,0])


    unitv = -np.array([1,0])

    # Radii to scan
    rvals = np.linspace(0.01, 10.0, 100)

    Q_FNc = []
    Q_LNc = []

    for r in rvals:
        x2 = x1 + r * n_hat
        Q_FNc.append(quad.FNc_quadrupole(x, x+unitv*r,  x,x+unitv*r))
        Q_LNc.append(quad.LNc_quadrupole(x, x+unitv*r-eps, x+eps, x+unitv*r))

    Q_FNc = np.array(Q_FNc)
    Q_LNc = np.array(Q_LNc)

    plt.figure(figsize=(7, 5))
    plt.plot(rvals, Q_FNc, label="Quadrupole finite $N_c$", linewidth=2)
    plt.plot(rvals, Q_LNc, label="Quadrupole large $N_c$", linewidth=2, linestyle="--")

    plt.xlabel(r"$r$")
    plt.ylabel(r"$Q(r)$")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show on screen; you can save instead if you prefer
    plt.show()


if __name__ == "__main__":
    line_configuration_scan()
