#!/usr/bin/env python3

# This code integrates to obtain the longitudinal and transverse LO DIS total cross section in the dipole picture  
# and then computes the structure functions. It runs from the command line as 
# LO_SF_integration_script.py Q_value xB_value mcpoints_value r_max_value m_value

import sys
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
from Integration_functions import ( 
    compute_cross_section
)
from time import perf_counter

start = perf_counter()
# ======================================================
# Parse arguments
# ======================================================

Q = float(sys.argv[1])
xB = float(sys.argv[2])
mcpoints = int(float(sys.argv[3]))
r_max = float(sys.argv[4])
m = float(sys.argv[5])
Zf = float(sys.argv[6])


# ======================================================
# Configuration
# ======================================================
alphaEM = 1 / 137

r_min, r_max = 1e-6, r_max
z_min, z_max = 1e-6, 1 - 1e-6
alpha_min, alpha_max = 0.0, 2*np.pi

# ======================================================
# Run integrations
# ======================================================

n_cores = multiprocessing.cpu_count()


def compute_F(mode, pol):
    """
    mode: "KC" or "LargeNc"
    pol:  "L" or "T"
    returns: (F, F_err)
    """
    sigma, sigma_err = compute_cross_section(
        mode, pol,
        Q, xB, m, Zf,
        r_min, r_max, r_min, r_max,
        z_min, z_max,
        alpha_min, alpha_max,
        mcpoints,
        n_cores
    )

    prefactor = Q**2 / ((4*np.pi**2) * alphaEM)
    return prefactor * sigma, prefactor * sigma_err

    
# ======================================================
# Compute all four structure functions
# ======================================================

KC_FL,  KC_FL_err  = compute_F("KC",      "L")
LNc_FL, LNc_FL_err = compute_F("LargeNc", "L")

KC_FT,  KC_FT_err  = compute_F("KC",      "T")
LNc_FT, LNc_FT_err = compute_F("LargeNc", "T")


# ======================================================
# Print results
# ======================================================

print("Q2, xB, m, KCFL, KCFLE, LNCKCFL, LNCKCFLE, KCFT, KCFTE, LNCKCFT, LNCKCFTE")
print(f"{Q**2}, {xB}, {m}, {KC_FL}, {KC_FL_err}, {LNc_FL}, {LNc_FL_err}, {KC_FT}, {KC_FT_err}, {LNc_FT}, {LNc_FT_err}")

end = perf_counter()

#print(end - start)
