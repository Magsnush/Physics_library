from scipy.integrate import dblquad

from small_x_physics.integrands.totalDIS.LO.OTintegrand import OTIntegrand
from small_x_physics.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared
from small_x_physics.multipole_models.MV_models.dipole import Dipole

def OT_integral(Q, m, Zf, r_max, quark_masses, photon_wf, sigma0, dipole_model, polarization="T", flavor=0, z_min=1e-6, z_max=1.0-1e-6):
    """
    Compute the optical theorem integral for the total DIS cross section.

    Parameters
    ----------
    Q : float
        Photon virtuality (GeV).
    m : float
        Quark mass (GeV). Currently treated as flavor-independent.
    Zf : float
        Quark charge factor for this flavor.
    r_max : float
        Upper limit of the r integration (GeV^-1).
    quark_masses : array-like
        Array of quark masses for each flavor.
    photon_wf : LO_FE_PhotonWF_squared instance
        Object that can compute the photon wavefunction squared.
    sigma0 : float
        Parameter for the transverse area of the target.
    dipole_model : Dipole instance
        Object that can compute the dipole amplitude.
    polarization : {"T", "L", "TL"}
        Photon polarization.
    flavor : int
        Quark flavor index.

    Returns
    -------
    float
        The value of the optical theorem integral.
    """
    integrand = OTIntegrand(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_model,
        polarization=polarization
    )

    # Integrate over r and z (use dblquad). The integrand expects signature
    # integrand.OT_integrand(r, Q, z, flavor).
    result, _ = dblquad(
        lambda z, r: integrand.OT_integrand(r, Q, z, flavor),
        0,
        r_max,
        z_min,
        z_max,
    )

    return result