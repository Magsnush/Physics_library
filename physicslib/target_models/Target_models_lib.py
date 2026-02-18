import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv
from scipy.integrate import quad
from scipy.interpolate import interp1d
from pathlib import Path

## BELOW WE DEFINE WHAT WE WILL USE FOR THE QUADRUPOLE AND DIPOLE ##

# parameters 
Qs0 = np.sqrt(0.104) #GeV
gamma = 1.0
LambdaQCD = 0.241 #GeV
ec = 1.0
Nc = 3


def radius_func(x, y):
    diff = x - y
    diff = np.asarray(diff)
    if diff.ndim == 0:
        return abs(diff)
    return np.linalg.norm(diff, axis=-1)


# Dipole exponent
def dip_exp(x,y ,Qs0, gamma, LambdaQCD, ec):
    eps = 1e-14
    r = radius_func(x,y) + eps
    return -((r**2 * Qs0**2)**gamma) / 4 * np.log(1/(r * LambdaQCD) + ec * np.exp(1))


# ============================================
#  Dipole
# ============================================
def dipole(x, y):
    return np.exp(dip_exp(x,y, Qs0, gamma, LambdaQCD, ec))

def dipole_1d(r):
    eps = 1e-14
    r = np.abs(r) + eps
    return np.exp(-((r**2 * Qs0**2)**gamma)/4 * np.log(1/(r * LambdaQCD) + ec*np.exp(1)))


def quadrupole(x1,x2,x2p,x1p, Nc, largeNc=False):

    # Casimir factor
    CF = (Nc**2 - 1) / (2 * Nc) # <---- THIS IS OK
    
    #Always ensure broadcasting works
    x1  = np.array(x1)
    x2  = np.array(x2)
    x1p = np.array(x1p)               # <---- THIS IS OK
    x2p = np.array(x2p)

    def f(x,y):
        return dip_exp(x,y, Qs0, gamma, LambdaQCD, ec)
    
    # Functions that appear in quadrupole in terms of dipole exponential
    def F(x1,x2,x2p,x1p):
        return (1/CF)*(f(x1,x2p) 
              + f(x2,x1p) 
              - f(x1,x1p) 
              - f(x2,x2p))              # <----- THIS IS OK
    
    F1 = F(x1,x2p,x2,x1p)
    F2 = F(x1,x2,x2p,x1p)               # <----- THIS IS OK

    # Shared dipole S(u) factors
    SuSup = np.exp(f(x1, x2) + f(x2p, x1p)) # <---- THIS IS OK
    
    # ============================================================
    #  Large-Nc approximation (simple)
    # ============================================================
    if largeNc:
        Su_mixed = np.exp(f(x1, x1p) + f(x2p, x2))
        return SuSup - (F2 / (F1 + 1e-12)) * (SuSup - Su_mixed)
    
    # ============================================================
    #  Full finite-Nc quadrupole (Gaussian)
    # ============================================================

    F3 = F(x1,x1p,x2p,x2)                  # <----- THIS IS OK

    # Discriminant (avoid tiny negative numerical noise)
    Delta = F1**2 + (4 / Nc**2) * F2 * F3
    sqrt_Delta = np.sqrt(Delta)             # <----- THIS IS OK


    # Avoid 0/0
    good = sqrt_Delta >0
    term1 = np.zeros_like(sqrt_Delta)
    term2 = np.zeros_like(sqrt_Delta)

    # Compute the two terms only where valid. A 1/Nc**2 factor can be introduced here if one is thinking of a dipole-dipole correlator as 
    term1[good] = ((sqrt_Delta[good] + F1[good]) / (2 * sqrt_Delta[good]) - F2[good] / sqrt_Delta[good])* np.exp( Nc * sqrt_Delta[good] / 4) 

    term2[good] = ((sqrt_Delta[good] - F1[good]) / (2 * sqrt_Delta[good]) + F2[good] / sqrt_Delta[good])* np.exp(-Nc * sqrt_Delta[good] / 4)    

    BigFactor = term1 + term2       # <----- THIS IS OK

    # Final finite-Nc quadrupole expression
    return SuSup * BigFactor * np.exp((-Nc/4)*F1 + (1/(2*Nc))*F2)

def quadrupole_uu(u, up, z, alpha, largeNc=False):
    x1  = np.stack([(1 - z) * u, np.zeros_like(u)], axis=-1)
    x2  = np.stack([-z * u, np.zeros_like(u)], axis=-1)
    x1p = np.stack([(1 - z) * up * np.cos(alpha), (1 - z) * up * np.sin(alpha)], axis=-1)
    x2p = np.stack([-z * up * np.cos(alpha), -z * up * np.sin(alpha)], axis=-1)

    if largeNc:
        return quadrupole(x1, x2, x2p, x1p, Nc, largeNc=True)
    
    return quadrupole(x1, x2, x2p, x1p, Nc, largeNc=False)

# uvals = np.linspace(0.01,20,100)
# angles = np.linspace(0.5,1,5)*np.pi
# z=0.0001

# plt.figure()
# for alpha in angles:
#     q = [quadrupole_uu(u, u, z, alpha, largeNc=True) for u in uvals]
#     plt.plot(uvals, q, label=f"{alpha}")
# plt.plot(uvals, [dipole(u,0) for u in uvals], label = "dipole")
# plt.legend()
# plt.show()



# Schenke, Lappi et al functions
def Square(x,y):
    dip = dipole(x,y)  
    diproot2 = dipole(np.sqrt(2)*x,np.sqrt(2)*y)
    return dip**2*((Nc+1.)/2*(dip/diproot2)**(2/(Nc+1)) - (Nc-1)/2*(diproot2/dip)**(2/(Nc-1)))

def LNc_Square(x,y):
    dip = dipole(x,y)  
    diproot2 = dipole(np.sqrt(2)*x,np.sqrt(2)*y)
    return dip**2 * (1 + 2*np.log(dip/diproot2))

def Line(x,y):
    dip  = dipole(x,y) 
    return (Nc+1)/2.*(dip)**(2*(Nc+2.)/(Nc+1.)) - (Nc-1.)/2.*(dip)**(2*(Nc-2.)/(Nc-1.))















def Dipole_distribution(r_max, K, sigma0, alphaS):
    S_perp = sigma0/2

    prefactor = (Nc * S_perp/(2 * alphaS * np.pi**2))

    def integrand(r):
        jacobian = (1/(2*np.pi)**2) * r
        angle_integrated_phase_factor = 2*np.pi * jv(0, K*r)
        return jacobian * angle_integrated_phase_factor * dipole_1d(r)

    integral, error = quad(integrand(0, r_max), limit=200)

    return prefactor * K**2 * integral