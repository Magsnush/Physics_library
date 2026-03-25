# physicslib

A Python library for modeling photon–proton Deep Inelastic Scattering (DIS) cross sections using the dipole picture and lightcone wavefunctions.

## Overview

**physicslib** provides reusable, modular building blocks for calculating structure functions and cross sections in DIS, with support for both finite-energy constrained and optical theorem approaches. The library implements leading-order (LO) calculations using color dipole models and quantum chromodynamics (QCD) dynamics.

### Key Features

- **Multiple Kinematic Schemes**: Support for both 4D and 5D kinematic formulations with configurable energy constraints
- **Lightcone Wavefunctions**: Implementations for finite-energy (FE) constrained and optical theorem (OT) photon wavefunctions
- **Dipole Models**: Saturation-based dipole cross section models including MV (McLerran-Venugopalan) and evolved dipole amplitudes via BK equations
- **Structure Functions**: Calculations of $F_L$ (longitudinal), $F_T$ (transverse), and $F_2$ (total) structure functions
- **Numerical Integration**: Optimized 2D and 4D integration routines for computing cross sections and structure functions
- **Flexible Architecture**: Modular design allowing customization of wavefunctions, multipole models, and integration parameters

## Repository Structure

```
physicslib/
├── wavefunctions/                    # Photon wavefunctions
│   ├── FE_photon_wavefunctions/     # Finite-energy constrained wavefunctions
│   ├── OT_photon_wavefunctions/     # Optical theorem wavefunctions
│   └── proton_wavefunctions/        # Proton target structure
├── multipole_models/
│   └── MV_models/                   # MV saturation model implementations
│       ├── dipole.py                # Core dipole cross section model
│       ├── gaussian_quadrupole.py   # Quadrupole correlation (NLO)
│       └── rcbk_adapter.py          # Interface to BK-evolved saturation
├── integrands/
│   └── totalDIS/
│       └── LO/                      # Leading-order integrands
│           ├── integrand4D.py       # 4D kinematic integrand
│           └── integrand5D.py       # 5D kinematic integrand
├── numerics/
│   ├── totalDIS/LO/                # Integration routines for structure functions
│   │   ├── Integration_functions_4D.py
│   │   ├── Integration_functions_5D.py
│   │   ├── OTIntegration.py        # Optical theorem integration
│   │   └── LO_*_integration_script.py  # Workflow scripts
│   └── collinear_dipole_matching/  # Collinear matching calculations
├── target_models/                   # Nuclear/nucleon target parameterizations
├── constants.py                     # Physical constants and parameters
└── PhotonProtonCrossSectionLib.py  # High-level API
```

## Core Modules

### Wavefunctions (`wavefunctions/`)
- **Finite-Energy (FE) Wavefunctions**: Constrained photon wavefunctions implementing kinematic constraints
- **Optical Theorem (OT) Wavefunctions**: Wavefunctions derived from the optical theorem in the BFKL limit
- **Proton Wavefunctions**: Incoming and outgoing proton wave function contributions

### Multipole Models (`multipole_models/MV_models/`)
- **Dipole Model**: Implements the color dipole cross section with saturation
  - Parameters: $Q_s$ (saturation scale), $\gamma$ (anomalous dimension), $e_c$ (small-x cutoff)
  - Methods: `S(r)` (amplitude), `S_xy(x, y)` (anisotropic version)
- **Gaussian Quadrupole**: NLO multipole correlators with Gaussian approximation
- **BK Adapter**: Interface to numerically evolved saturation via the Balitsky-Kovchegov equation

### Integration Routines (`numerics/`)
- **4D Integration**: Energy-conserving kinematic scheme
- **5D Integration**: Unconstrained integration over all 5 variables
- **Optical Theorem Integration**: Direct computation of OT cross sections
- **Performance**: Vectorized NumPy operations for efficiency

## Installation

### Prerequisites
- Python 3.9+
- NumPy, SciPy, pandas (for analysis)
- Matplotlib (for visualization)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Magsnush/Physics_library.git
cd Physics_library
```

2. Create and activate a Python environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e .
```

## Usage

### Basic Example: Computing Structure Functions

```python
import numpy as np
from physicslib.multipole_models.MV_models.dipole import Dipole
from physicslib.wavefunctions.OT_photon_wavefunctions.LO import LO_OT_PhotonWF_squared
from scipy.integrate import dblquad

# Initialize models
dipole = Dipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0)
wf = LO_OT_PhotonWF_squared(quark_masses=np.array([0.14]), quark_charges=np.array([np.sqrt(2/3)]))

# Define integrand for longitudinal structure function
def integrand_FL(r, z, Q):
    wf_sq = wf.psi_L_squared(Q, r, z, flavor=0)
    dipole_amp = 2.0 * (1.0 - dipole.S(r))
    jac = 2.0 * np.pi * r / (z * (1.0 - z))
    return jac * wf_sq * dipole_amp

# Integrate
Q = 5.0  # GeV
result, error = dblquad(
    lambda z, r: integrand_FL(r, z, Q),
    0.0, 20.0,  # r limits
    lambda r: 1e-6, lambda r: 1.0 - 1e-6  # z limits
)
```

### Example: Comparing 4D vs 5D Schemes

See `tests/FE_OT_matching_test.ipynb` and `tests/LO_totalDIS_4D_vs_5D_speedtest.py` for comprehensive examples comparing kinematic schemes and performance.

## Testing

Run the test suite:
```bash
cd tests
python -m pytest .  # Or run individual test files
```

### Key Tests
- `FE_OT_matching_test.ipynb`: Validates finite-energy vs optical theorem matching
- `LO_totalDIS_4D_vs_5D_speedtest.py`: Benchmarks 4D and 5D integration
- `gaussian_quadrupole_a_test.py`: Tests NLO quadrupole calculations
- `test_bk_5d_integration.py`: Validates BK-evolved dipole integration

## References

This library implements theoretical frameworks from:
- Dipole picture and color glass condensate
- BFKL dynamics (Balitsky-Kovchegov equation)
- Finite-energy constraints in DIS
- Optical theorem in the BFKL limit

## Contributing

Contributions are welcome. Please ensure:
1. Code follows PEP 8 style guidelines
2. New features include tests in the `tests/` directory
3. Commit messages are descriptive and reference related work

## License

See LICENSE file for details.

