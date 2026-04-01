"""
Basic tests for cross-section integrands.

Currently:
    - Tests the LODISIntegrand in physicslib.integrands.totalDIS.LO.integrand
      by evaluating it at a sample kinematic point using realistic
      photon wavefunctions and MV dipole/quadrupole models.
"""

import numpy as np
import sys
import importlib

from small_x_physics.integrands.totalDIS.LO.integrand import LODISIntegrand
from small_x_physics.wavefunctions.FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from small_x_physics.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared
from small_x_physics.multipole_models.MV_models.dipole import Dipole
from small_x_physics.multipole_models.MV_models.gaussian_quadrupole import GaussianQuadrupole


def build_lodis_integrand(
    polarization: str = "T",
    largeNc: bool = False,
) -> LODISIntegrand:
    """
    Construct a LODISIntegrand instance with reasonable default parameters.
    """
    # Simple 3-flavor setup (u, d, c) as an example
    quark_masses = np.array([0.14, 0.14, 1.27])  # GeV
    quark_charges = np.array([2 / 3, -1 / 3, 2 / 3])

    # Photon wavefunction (finite-energy, LO)
    photon_wf = LO_FE_PhotonWF_squared(
        quark_masses=quark_masses,
        quark_charges=quark_charges,
    )

    # MV dipole parameters (you can adjust these)
    Qs0 = np.sqrt(0.104)  # saturation scale
    gamma = 1.0
    ec = 1.0
    dipole_model = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)

    # Gaussian quadrupole built from the same dipole model
    quad_model = GaussianQuadrupole(dipole_model)

    # Transverse area parameter sigma0 (same as used elsewhere in the project)
    sigma0 = 2.57 * 2 * 18.81

    # Wrap the quadrupole in a callable that matches LODISIntegrand expectations:
    # quadrupole(u, up, z, theta) -> scalar/array
    def quadrupole_polar(u, up, z, theta):
        return quad_model.quadrupole_polar(u, up, z, theta, largeNc=largeNc)

    return LODISIntegrand(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_model,
        quadrupole_model=quadrupole_polar,
        polarization=polarization,
        largeNc=largeNc,
    )


def build_lodis_ot_integrand(
    polarization: str = "T",
) -> LODISIntegrand:
    """
    Construct a LODISIntegrand instance configured for the OT_integrand.

    Uses the LO optical-theorem photon wavefunction and a radial MV dipole.
    """
    quark_masses = np.array([0.14, 0.14, 1.27])  # GeV
    quark_charges = np.array([2 / 3, -1 / 3, 2 / 3])

    photon_wf = LO_OT_PhotonWF_squared(
        quark_masses=quark_masses,
        quark_charges=quark_charges,
    )

    # MV dipole (same parameters as before)
    Qs0 = np.sqrt(0.104)
    gamma = 1.0
    ec = 1.0
    dipole_model = Dipole(Qs0=Qs0, gamma=gamma, ec=ec)

    sigma0 = 2.57 * 2 * 18.81

    # For OT_integrand we only need a radial dipole S(r); LODISIntegrand
    # expects self.dipole(r, 0) in the OT case, so we pass a small
    # wrapper with that call signature.
    def dipole_radial(r, _y):
        return dipole_model.S(r)

    # Quadrupole is not used in OT_integrand; provide a dummy.
    def dummy_quadrupole(*_args, **_kwargs):
        return 0.0

    return LODISIntegrand(
        quark_masses=quark_masses,
        photon_wf=photon_wf,
        sigma0=sigma0,
        dipole_model=dipole_radial,
        quadrupole_model=dummy_quadrupole,
        polarization=polarization,
        largeNc=False,
    )


def test_lodis_integrand_single_point():
    """
    Evaluate LODISIntegrand at a single kinematic point and print the result.

    This is not a unit test with assertions yet, but a numerical
    sanity check that the pipeline runs without errors and returns
    a finite value.
    """
    integrand = build_lodis_integrand(polarization="T", largeNc=False)

    # Sample kinematic point
    Q = 10.0        # GeV
    Msq_max = 100.0 # GeV^2 (example upper bound)
    u = 0.25         # transverse size (GeV^-1)
    up = 0.25
    z = 0.4         # momentum fraction
    theta = 0.01     # angle between u and up
    flavor = 0      # first flavor (u quark)

    # Backward/forward compatibility: older versions used `integrand(...)`
    # while newer versions use `FE_integrand(...)`.
    fe_method = getattr(integrand, "FE_integrand", None)
    if fe_method is None:
        fe_method = getattr(integrand, "integrand")

    value = fe_method(Q, Msq_max, u, up, z, theta, flavor=flavor)

    print("LODISIntegrand value at sample point:")
    print(f"  value = {value}")

    if not np.isfinite(value):
        raise RuntimeError(f"LODISIntegrand FE integrand returned non-finite value: {value}")


def test_lodis_ot_integrand_single_point():
    """
    Evaluate LODISIntegrand.OT_integrand at a single kinematic point.
    """
    integrand = build_lodis_ot_integrand(polarization="T")

    Q = 10.0   # GeV
    r = 0.25    # dipole size (GeV^-1)
    z = 0.4
    flavor = 0

    ot_method = getattr(integrand, "OT_integrand", None)
    if ot_method is None:
        # Helpful diagnostics if Python imported a different module version.
        mod = importlib.import_module(LODISIntegrand.__module__)
        raise AttributeError(
            "LODISIntegrand has no OT_integrand(). "
            f"Imported from: {getattr(mod, '__file__', 'unknown')} ; "
            f"sys.path[0]={sys.path[0]!r}"
        )

    value = ot_method(Q, r, z, flavor=flavor)

    print("LODISIntegrand OT_integrand value at sample point:")
    print(f"  value = {value}")

    if not np.isfinite(value):
        raise RuntimeError(f"LODISIntegrand OT_integrand returned non-finite value: {value}")


if __name__ == "__main__":
    # Run the simple tests when executed as a script
    test_lodis_integrand_single_point()
    test_lodis_ot_integrand_single_point()
