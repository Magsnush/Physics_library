import numpy as np
from scipy.special import kv

# Optical theorem definition of the square of the lightcone wavefunctions for photon interacting with proton as found in Kovchegov and Levin
def OT_photon_proton_wave_function_sq(r, z, polarization, Q, m, Zf, Nc):
    """
    Squared lightcone wavefunction for photon polarization ('L' or 'T').

    Variables:
    r : dipole size (float or np.array)
    z : momentum fraction of quark (float or np.array)
    polarization : 'L' for longitudinal, 'T' for transverse

    Returns:
    float or np.array: the squared wavefunction
    """
    Q2 = Q**2   # Momentum transfer squared
    m2 = m**2   # Atm keep mass as a single parameter. Can make it flavor dep. in future
    Zf2 = Zf**2 # Sum of the quark charges square
    alphaEM = 1/137
    epsilon_sq = Q2 * z * (1 - z) + m2
    bessel_arg = r * np.sqrt(epsilon_sq)
    LongCoefficient = (8*Nc*alphaEM*Zf2/np.pi)
    TransCoefficient = LongCoefficient/4

    if polarization == "L":
        K0_sq = kv(0, bessel_arg)**2
        return LongCoefficient * Q2 * z**3 * (1 - z)**3 * K0_sq

    elif polarization == "T":
        K0_sq = kv(0, bessel_arg)**2
        K1_sq = kv(1, bessel_arg)**2
        term1 = (z**2 + (1 - z)**2) * epsilon_sq * K1_sq
        term2 = m2 * K0_sq
        return TransCoefficient * (term1 + term2)* z*(1-z)

    else:
        raise ValueError("Polarization must be 'L' (longitudinal) or 'T' (transverse)")


# Kinematically constrained version of the lightcone wavefunction. Here u and up are dipole radii for the amplitude and its conjugate respectively
def KC_photon_proton_wave_function_sq(u, up, z, alpha, polarization, Q, m, Zf, Nc):
    """
    Squared lightcone wavefunction for photon polarization ('L' or 'T').

    Variables:
    u / up : dipole size / conjugate dipole size
    z : momentum fraction of quark (float or np.array)
    polarization : 'L' for longitudinal, 'T' for transverse

    Returns:
    float or np.array: the squared wavefunction
    """
    Q2 = Q**2   # Momentum transfer squared 
    m2 = m**2   # Atm keep mass as a single parameter. Can make it flavor dep. in future
    Zf2 = Zf**2
    alphaEM = 1/137
    epsilon_sq = Q2 * z * (1 - z) + m2
    LongCoefficient = (8*Nc*alphaEM*Zf2/np.pi)
    TransCoefficient = LongCoefficient/4

    # Bessel functions that represent coordinate space version of the LCWFs
    bessel_arg = u * np.sqrt(epsilon_sq)
    bessel_arg_conj = up * np.sqrt(epsilon_sq)
    K0 = kv(0,bessel_arg)
    K0_conj = np.conjugate(kv(0, bessel_arg_conj))
    K1 = kv(1,bessel_arg)
    K1_conj = np.conjugate(kv(1, bessel_arg_conj))

    K0_sq = K0 * K0_conj
    K1_sq = K1 * K1_conj

    if polarization == "L":
        return LongCoefficient * Q2 * z**3 * (1 - z)**3 * K0_sq

    elif polarization == "T":
        term1 = (z**2 + (1 - z)**2) * epsilon_sq * np.cos(alpha) * K1_sq
        term2 = m2 * K0_sq
        return TransCoefficient * z* (1-z) * (term1 + term2)

    else:
        raise ValueError("Polarization must be 'L' (longitudinal) or 'T' (transverse)")

# print(OT_photon_proton_wave_function_sq(5, 0.5, 'T', 1,1,1,3), KC_photon_proton_wave_function_sq(5,5,0.5,0,'T',1,1,1,3))

### BOTH OT AND KC TAKE ON SAME VALUE FOR AS LONG AS ALPHA = 0 ###