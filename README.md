# small_x_physics

A Python library for computing different cross sections at small x in the dipole picture. Currently it can compute inclusive deep inelastic scattering (DIS) cross sections at LO and with an rcBK evolved dipole, both with the optical theorem (OT) and constrained at finite W. 
It provides the ingredients (photon light-cone wavefunctions, dipole and quadrupole correlators) and cross sections
assembled from them:

- **Optical theorem (OT)** cross sections at leading order (LO).
- **Finite-energy constrained (FE)** cross sections at LO, eq. 25 of
  [arXiv:2601.07302](https://arxiv.org/abs/2601.07302), where the target
  amplitude is $1 - S(u) - S(u') + S_4$ and the quadrupole $S_4$ is built from
  dipoles in the Gaussian approximation.
- Both with either an analytic **MV or GBW** dipole or an **rcBK-evolved
  dipole** read from the output of the
  [rcBK solver](https://github.com/hejajama/rcbkdipole).
- Cross sections for **matching the dipole picture onto collinear
  factorisation** (the FE cross section with the quadrupole evaluated at z = 0,
  and its difference from the full one).

The multidimensional integrals are done with [VEGAS](https://vegas.readthedocs.io)
Monte Carlo, vectorised over batches of points and parallelised over CPU cores.

## Installation

Requires Python 3.9 or newer.

```bash
git clone https://github.com/Magsnush/Physics_library.git
cd Physics_library
pip install -e .
```

`pip install -e .` installs the package `small_x_physics` and its dependencies
(numpy, scipy and the Monte Carlo integrator vegas, which brings gvar with it)
in editable mode, so changes to the source take effect without reinstalling.
matplotlib is only needed for the plotting check scripts.

## Quick start

The finite-energy constrained structure functions with an rcBK-evolved dipole,
using the example BK solution in `data/mv.dat` (run from the repository root):

```python
import numpy as np
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import FE_CrossSection_BK_4D
from small_x_physics.building_blocks.constants import alpha_em

Q, xB = np.sqrt(10.0), 1e-3                # Q in GeV, Bjorken x
params = dict(
    Q=Q, xB=xB,
    mf=0.14, Zf=np.sqrt(2/3),              # u, d, s together, with a common mass
    sigma0=2 * 2.57 * 18.81,               # GeV^-2 (sigma0/2 = 18.81 mb)
    Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0,
    bkfile="data/mv.dat", x0=0.01,
    mcpoints=100_000,
)
# r_min, r_max (GeV^-1), z_min, z_max, theta_min, theta_max
limits = (1e-6, 30.0, 1e-8, 1 - 1e-8, 0.0, 2 * np.pi)

if __name__ == "__main__":
    sigma = {}
    for pol in ("L", "T"):
        cs = FE_CrossSection_BK_4D(polarization=pol, **params)
        sigma[pol] = cs.BK_FE_cross_section_4D(*limits)    # (mean, error) in GeV^-2

    to_F = Q**2 / (4 * np.pi**2 * alpha_em)
    FL, FT = to_F * sigma["L"][0], to_F * sigma["T"][0]
    print(f"F_L = {FL:.4f}, F_T = {FT:.4f}, F_2 = {FL + FT:.4f}")
```

A complete example script computes the OT, FE 4D and FE 5D structure functions
with a BK dipole and prints them as CSV:

```bash
python3 src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py \
    --bkfile data/mv.dat --xB 1e-3 --mcpoints 100000
```

Its other options (`--Q`, `--mf`, `--Zf`, `--sigma0`, `--x0`, `--largeNc`, ...)
are listed by `--help`.

## What is in the library

```
src/small_x_physics/
├── building_blocks/
│   ├── constants.py                      alpha_em, Nc, CF, LambdaQCD
│   ├── wavefunctions/
│   │   ├── OT_photon_wavefunctions/LO.py     LO_OT_PhotonWF_squared
│   │   └── FE_photon_wavefunctions/LO.py     LO_FE_PhotonWF_squared
│   └── correlators/
│       ├── Dipoles/IC_dipole.py              ICDipole: MV and GBW dipoles
│       ├── Dipoles/BK_dipole.py              BKDipole: rcBK solution from a file
│       └── Quadrupoles/QuadrupoleCorrelator.py   QuadrupoleCorrelatorModel
└── Cross_Sections/
    ├── InclusiveDIS/LO/                  OT and FE cross sections, MV/GBW dipole
    ├── InclusiveDIS/withBK/              OT and FE cross sections, BK dipole
    └── collinear_dipole_matching/        FE cross sections for collinear matching
data/mv.dat                               example rcBK solution (see data/README.md)
```

The `NLO.py` files next to the LO wavefunctions are empty placeholders.

### Building blocks

| Class | Description |
|---|---|
| `LO_OT_PhotonWF_squared(Q, mf, Zf)` | Squared LO photon wavefunctions of the OT cross section: `psi_L_squared(Q, r, z)`, `psi_T_squared(Q, r, z)`. |
| `LO_FE_PhotonWF_squared(mf, Zf)` | The FE counterparts, a product of amplitude and conjugate amplitude with dipole sizes `u`, `u'` at relative angle `theta`: `psi_L_squared(Q, u, up, z, theta)`, `psi_T_squared(...)`. |
| `ICDipole(Qs0, gamma, ec, LambdaQCD)` | Analytic dipole S-matrices of transverse positions `x`, `y` (2-vectors on the last axis): `MV_model_S2(x, y)` $= \exp[-(r^2 Q_{s0}^2)^\gamma/4 \, \ln(1/(r\Lambda_\mathrm{QCD}) + e_c e)]$, `GBW_model_S2(x, y)` $= \exp(-r^2 Q_{s0}^2/4)$. |
| `BKDipole(bkfile, Y=None)` | rcBK solution N(r, Y) read from a file and interpolated with a bicubic spline: `N(r, Y)`, `S_r(r, Y)` = 1 − N, `S_xy(x, y, Y)`, `rapidity_from_x(x)` = ln(x0/x). The initial-condition parameters in the file header are available as `Qs0_sq`, `gamma`, `ec`, `x0`, `LambdaQCD`. |
| `QuadrupoleCorrelatorModel(dipole_model, Nc, LambdaQCD)` | Quadrupole in the Gaussian approximation (eq. B21 of Dominguez, Marquet, Xiao and Yuan, Phys. Rev. D 83, 105005 (2011)), built from any dipole function: `quadrupole_polar(u, up, z, theta, dipole_args=None, largeNc=False)`. `polar_correlators(...)` returns S(u), S(u′) and S₄ together. |

### Cross sections

Each cross section is a class: construct it with the kinematics and model
parameters, then call its integration method with the integration limits.

| Class | Module (under `small_x_physics.Cross_Sections`) | Computes | Integral |
|---|---|---|---|
| `OT_CrossSection_LO` | `InclusiveDIS.LO.LO_OT_CS` | OT, MV dipole | 2D, scipy `dblquad` |
| `FE_CrossSection_LO` | `InclusiveDIS.LO.LO_FE_CS` | FE, MV or GBW dipole | 4D, VEGAS |
| `OT_CrossSection_BK` | `InclusiveDIS.withBK.BK_OT_CS` | OT, BK dipole at Y = ln(x0/xB) | 2D, VEGAS |
| `FE_CrossSection_BK_4D` | `InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB` | FE, BK dipole and quadrupole at Y = ln(x0/xB) | 4D, VEGAS |
| `FE_CrossSection_BK_5D` | `InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP` | FE with the invariant mass M² of the quark pair integrated explicitly, the dipoles evaluated at Y = ln(x0/x_P) | 5D, VEGAS |
| `FE_CrossSection_LO_z_is_zero_correlator` | `collinear_dipole_matching.LO_FE_CS_Z0_in_correlator` | FE with the quadrupole evaluated at z = 0 | 4D, VEGAS |
| `FE_CrossSection_LO_diff` | `collinear_dipole_matching.LO_FE_CS_diff` | full FE minus the z = 0 one; integrand S₄(z) − S₄(0) | 4D, VEGAS |

In the 5D cross section, $x_P = (M^2 + Q^2)/(W^2 + Q^2) \geq x_B$ with
$W^2 = Q^2(1/x_B - 1)$. Where $x_P > x_0$ the BK solution has no data, and the
dipole is held at its initial condition (Y = 0).

Methods and return values:

| Class | Method | Returns |
|---|---|---|
| `OT_CrossSection_LO` | `OT_cross_section(r_min, r_max, z_min, z_max)` | `(sigma_L, err_L, sigma_T, err_T)` |
| `OT_CrossSection_BK` | `BK_OT_cross_section(r_min, r_max, z_min, z_max)` | `(sigma_L, err_L, sigma_T, err_T)` |
| `FE_CrossSection_LO` and the two collinear-matching classes | `cross_section(r_min, r_max, z_min, z_max, theta_min, theta_max)` | `(sigma, err)` |
| | `dsigma_dz(z, r_min, r_max, theta_min, theta_max)` | `(dsigma/dz, err)` at one z |
| `FE_CrossSection_BK_4D` | `BK_FE_cross_section_4D(r_min, r_max, z_min, z_max, theta_min, theta_max)` | `(sigma, err)` |
| `FE_CrossSection_BK_5D` | `BK_FE_cross_section_5D(r_min, r_max, z_min, z_max, theta_min, theta_max, Msq_min, Msq_max)` | `(sigma, err)` |

The OT classes return both polarizations. The FE classes compute one, chosen by
the `polarization` argument, so L and T are two objects (as in the quick start).

## How to use it

### Conventions

- **Units.** Q and masses in GeV; dipole sizes in GeV⁻¹; `sigma0` and the
  returned cross sections in GeV⁻² (1 mb = 2.568 GeV⁻²).
- **Structure functions.** $F_{L,T} = \frac{Q^2}{4\pi^2\alpha_\mathrm{em}}\,\sigma_{L,T}$,
  $F_2 = F_L + F_T$, and the reduced cross section
  $\sigma_r = F_2 - \frac{y^2}{1 + (1-y)^2} F_L$ with $y = Q^2/(s\,x_B)$.
  When L and T come from separate integrations their errors are independent,
  so the error of $\sigma_r = F_T + (1 - f) F_L$ is
  $\sqrt{\delta F_T^2 + (1 - f)^2\, \delta F_L^2}$, with $f = y^2/(1+(1-y)^2)$.
- **Flavours.** One object computes one group of quarks with a common mass:
  `mf` is the mass and `Zf` the charge factor, with `Zf**2` the sum of the
  squared quark charges in the group: `np.sqrt(2/3)` for u, d, s, and `2/3`
  for charm. Add the groups up afterwards.
- **Errors.** The VEGAS classes return the Monte Carlo standard deviation.
  `OT_CrossSection_LO` returns `dblquad`'s error estimate.

### Constructor arguments

| Argument | Meaning |
|---|---|
| `Q` | photon virtuality, as $\sqrt{Q^2}$ in GeV |
| `xB` | Bjorken x |
| `mf`, `Zf` | quark mass (GeV) and charge factor, see above |
| `sigma0` | σ₀ in GeV⁻²; σ₀/2 is the proton's transverse area, and every cross section is proportional to it |
| `Qs0`, `gamma`, `ec` | MV initial condition: saturation scale (GeV), anomalous dimension, and the constant in the logarithm |
| `bkfile`, `x0` | rcBK solution file, and the x at which its evolution starts |
| `mcpoints` | VEGAS integrand evaluations per iteration |
| `polarization` | `"L"` or `"T"` (FE classes) |
| `dipole_model` | `"MV"` or `"GBW"` (`FE_CrossSection_LO` and the collinear-matching classes) |
| `largeNc` | `True` for the large-Nc quadrupole; finite Nc by default (`FE_CrossSection_BK_4D`, `FE_CrossSection_BK_5D`) |

Constructor signatures:

```python
OT_CrossSection_LO(Q, mf, Zf, sigma0, Qs0, gamma, ec)
FE_CrossSection_LO(Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, mcpoints, polarization, dipole_model)
OT_CrossSection_BK(Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints)
FE_CrossSection_BK_4D(Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints, polarization, largeNc=False)
FE_CrossSection_BK_5D(Q, xB, mf, Zf, sigma0, Qs0, gamma, ec, bkfile, x0, mcpoints, polarization, largeNc=False)
```

The BK classes accept `Qs0`, `gamma` and `ec` but do not use them: the initial
condition is the one the BK file was computed with. `BKDipole.check_parameters`
compares the values you pass with the file header (see below). The default
parameters used in these examples (`Qs0**2 = 0.104` GeV², γ = 1, e_c = 1,
σ₀/2 = 18.81 mb) are the MV fit of
[arXiv:1309.6963](https://arxiv.org/abs/1309.6963), which `data/mv.dat` was
computed with.

### Integration limits

Typical choices: dipole sizes from `1e-6` to `30` GeV⁻¹, `z` from `1e-8` to
`1 - 1e-8`, `theta` from 0 to 2π and, for the 5D cross section, the invariant
mass from `4*mf**2` to `W**2 = Q**2 * (1/xB - 1)`. For a BK dipole, keep the
dipole sizes inside the r range of the file.

### Examples

LO cross sections with an analytic dipole, no BK file needed:

```python
import numpy as np
from small_x_physics.Cross_Sections.InclusiveDIS.LO.LO_OT_CS import OT_CrossSection_LO
from small_x_physics.Cross_Sections.InclusiveDIS.LO.LO_FE_CS import FE_CrossSection_LO

if __name__ == "__main__":
    model = dict(mf=0.14, Zf=np.sqrt(2/3), sigma0=2 * 2.57 * 18.81,
                 Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0)
    Q, xB = np.sqrt(10.0), 1e-3

    # Optical theorem: both polarizations at once.
    ot = OT_CrossSection_LO(Q=Q, **model)
    sigma_L, err_L, sigma_T, err_T = ot.OT_cross_section(1e-6, 30.0, 1e-8, 1 - 1e-8)

    # Finite-energy constrained, one polarization per object.
    fe_T = FE_CrossSection_LO(Q=Q, xB=xB, mcpoints=100_000, polarization="T",
                              dipole_model="MV", **model)
    sigma, err = fe_T.cross_section(r_min=1e-6, r_max=30.0, z_min=1e-8, z_max=1 - 1e-8,
                                    theta_min=0.0, theta_max=2 * np.pi)
    # The same integrand, differential in z at z = 0.3.
    dsigma_dz, err_dz = fe_T.dsigma_dz(0.3, r_min=1e-6, r_max=30.0,
                                       theta_min=0.0, theta_max=2 * np.pi)
```

The three BK cross sections:

```python
import numpy as np
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_OT_CS import OT_CrossSection_BK
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_4D_evolve_in_xB import FE_CrossSection_BK_4D
from small_x_physics.Cross_Sections.InclusiveDIS.withBK.BK_FE_5D_evolve_in_xP import FE_CrossSection_BK_5D

if __name__ == "__main__":
    Q, xB, mf = np.sqrt(10.0), 1e-3, 0.14
    common = dict(Q=Q, xB=xB, mf=mf, Zf=np.sqrt(2/3), sigma0=2 * 2.57 * 18.81,
                  Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0,
                  bkfile="data/mv.dat", x0=0.01, mcpoints=100_000)

    sigma_L, err_L, sigma_T, err_T = OT_CrossSection_BK(**common).BK_OT_cross_section(
        1e-6, 30.0, 1e-8, 1 - 1e-8)

    fe4 = FE_CrossSection_BK_4D(polarization="L", largeNc=False, **common)
    sigma_L4, err_L4 = fe4.BK_FE_cross_section_4D(1e-6, 30.0, 1e-8, 1 - 1e-8, 0.0, 2 * np.pi)

    W2 = Q**2 * (1 / xB - 1)
    fe5 = FE_CrossSection_BK_5D(polarization="L", **common)
    sigma_L5, err_L5 = fe5.BK_FE_cross_section_5D(1e-6, 30.0, 1e-8, 1 - 1e-8, 0.0, 2 * np.pi,
                                                  4 * mf**2, W2)
```

The building blocks on their own, for example to plot a dipole or a quadrupole:

```python
import numpy as np
from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole
from small_x_physics.building_blocks.correlators.Dipoles.IC_dipole import ICDipole
from small_x_physics.building_blocks.correlators.Quadrupoles.QuadrupoleCorrelator import QuadrupoleCorrelatorModel
from small_x_physics.building_blocks.wavefunctions import LO_FE_PhotonWF_squared
from small_x_physics.building_blocks.constants import Nc, LambdaQCD

dipole = BKDipole("data/mv.dat")
print(dipole)                                    # grid ranges and header parameters
print(dipole.check_parameters(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0, x0=0.01))  # [] if all agree

r = np.geomspace(1e-3, 10.0, 50)                 # GeV^-1
Y = dipole.rapidity_from_x(1e-4)                 # ln(x0 / x)
N = dipole.N(r, Y)
S = dipole.S_r(r, Y)                             # 1 - N

# Analytic dipoles take transverse positions (2-vectors on the last axis).
mv = ICDipole(Qs0=np.sqrt(0.104), gamma=1.0, ec=1.0, LambdaQCD=LambdaQCD)
S_MV = mv.MV_model_S2(np.stack([r, np.zeros_like(r)], axis=-1), np.zeros(2))

# Quadrupole of the BK dipole; extra dipole arguments go in dipole_args.
quad = QuadrupoleCorrelatorModel(dipole_model=dipole.BK_evolved_MV_model_S2_Y, Nc=Nc, LambdaQCD=LambdaQCD)
u, z, theta = 1.0, 0.3, 0.5
S4 = quad.quadrupole_polar(np.full_like(r, u), r, np.full_like(r, z), np.full_like(r, theta),
                           dipole_args={"Y": Y})

psi_T_sq = LO_FE_PhotonWF_squared(0.14, np.sqrt(2/3)).psi_T_squared(np.sqrt(10.0), u, r, z, theta)
```

### Parallel runs and number of points

The VEGAS cross sections evaluate the integrand on several processes. The
number of processes is read from the environment variable
`SLURM_CPUS_PER_TASK`, which SLURM sets inside a job; without it every CPU of
the machine is used. To run on one process:

```bash
SLURM_CPUS_PER_TASK=1 python3 my_script.py
```

The worker processes are started with Python's `multiprocessing`, so put the
code that runs integrals under `if __name__ == "__main__":` as in the examples.
On macOS and Windows this is required.

`mcpoints` sets the cost and the precision. Each integral runs 10 adaptation
iterations of `mcpoints/10` points, which are discarded, followed by 20
iterations of `mcpoints` points that make up the result, so the error falls
like `1/sqrt(mcpoints)`. 10⁵ points is enough to try things out (seconds per
integral on a multi-core machine); production runs typically use 10⁶ to 2×10⁶.
The 5D integral converges more slowly than the 4D one and needs more points
for the same error.

### BK solution files

`BKDipole` reads the text output of the
[rcBK solver](https://github.com/hejajama/rcbkdipole): a comment header
(lines starting with `#`) recording the initial condition, then four `###`
lines giving the r grid (`MinR`, `RMultiplier`, `RPoints`) and `x0`, then one
block per rapidity: a `### Y` line followed by `RPoints` values of N on the
grid r_i = MinR · RMultiplier^i. `data/mv.dat` is an example; `data/README.md`
lists its parameters.

Queries outside the tabulated grid are evaluated at the nearest edge, with a
warning the first time it happens. Pass `on_out_of_range="raise"` to refuse
them instead, or `warn_on_clamp=False` to silence the warning (for the 5D cross
section, where x_P > x0 is expected, set `cs.BKdipole.warn_on_clamp = False`).
`BKDipole.clamp_report()` counts how many queries were clamped.

## Example and check scripts

- `Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py`: OT, FE 4D and
  FE 5D structure functions with a BK dipole, printed as CSV (see Quick start).
- `building_blocks/correlators/Dipoles/testDipoles.py` and
  `building_blocks/correlators/Quadrupoles/testQuadrupoles.py`: plot BK, MV and
  GBW dipoles and quadrupoles (need matplotlib).

## References

- Finite-energy constrained dipole picture: [arXiv:2601.07302](https://arxiv.org/abs/2601.07302).
- Gaussian approximation of the quadrupole: F. Dominguez, C. Marquet, B.-W. Xiao
  and F. Yuan, Phys. Rev. D 83, 105005 (2011), eq. B21.
- rcBK solver and file format: <https://github.com/hejajama/rcbkdipole>.
- MV initial-condition parameters: [arXiv:1309.6963](https://arxiv.org/abs/1309.6963).

## License

MIT, see [LICENSE](LICENSE).
