"""
Wavefunction subpackage for photon light-cone wavefunctions.

Provides finite-energy (FE) and optical-theorem (OT) photon
wavefunctions at leading order.
"""

from .FE_photon_wavefunctions.LO import LO_FE_PhotonWF_squared
from .OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared

__all__ = [
    "LO_FE_PhotonWF_squared",
    "LO_OT_PhotonWF_squared",
]

