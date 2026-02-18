## Purpose
Short, actionable instructions for an AI coding assistant to be productive in this repository.

## Quick start (install / run / smoke tests)
- Install minimal runtime: this project expects numpy and scipy (see `pyproject.toml`).
- From the repo root you can run quick checks without installing the package:
  - PYTHONPATH=. python tests/integrand_test.py  # basic integrand smoke tests
  - python physicslib/PhotonProtonCrossSectionLib.py  # small usage example / smoke run
- For development installs: pip install -e .  (Setuptools backend in `pyproject.toml`.)

## Big-picture architecture
- Top-level package: `physicslib`.
- Major subpackages:
  - `wavefunctions/` — photon wavefunctions (FE/OT, LO/NLO). Example entrypoints: `FE_photon_wavefunctions.LO.LO_FE_PhotonWF_squared`
  - `multipole_models/` — dipole & quadrupole MV-model implementations (radial and Cartesian APIs).
    - MV models live in `multipole_models/MV_models/` (see `dipole.py` and `gaussian_quadrupole.py`).
  - `integrands/` — physics integrands wired to use wavefunctions & target models (tests exercise `integrands/totalDIS/LO/integrand`).
  - `numerics/` — numerical integration helpers and higher-level scripts.
  - `target_models/` and `utils/` — supporting helpers and target model wrappers.

## Key conventions & APIs (practical examples)
- Units: GeV (Q, masses, LambdaQCD) and GeV^-1 for transverse coordinates. Look at `constants.py` for physical constants.
- Dipole API (example from `multipole_models/MV_models/dipole.py`):
  - `Dipole(Qs0, gamma, ec)` exposes `S(r)` (radial), `S_r(r)` and `S_xy(x,y)` and `exponent(x,y)`.
  - Prefer `S(r)` when integrands use radial variables (u, up).
- Quadrupole API (example from `multipole_models/MV_models/gaussian_quadrupole.py`):
  - `GaussianQuadrupole(dipole_model)` provides `quadrupole(x1,x2,x2p,x1p)` and `quadrupole_polar(u, up, z, theta, largeNc=False)`.
  - Broadcasting is expected: many functions accept NumPy arrays; coordinates are often packed as last-axis 2-vectors via `np.stack`.
  - `largeNc` boolean toggles large-Nc approximation vs finite-Nc formula.
- Wavefunctions:
  - Two families: FE (finite-energy) and OT (optical theorem). Use the `LO_*` classes under `wavefunctions/*/LO.py` for examples of the expected method names (e.g. `psi_T_squared`, `psi_L_squared`, or `psi_T/psi_L` wrappers used by integrands).

## Developer workflows & debugging
- Running tests / smoke checks:
  - Quick: PYTHONPATH=. python tests/integrand_test.py
  - Prefer reproducible runs by installing editable package: pip install -e . then run `python -m pytest tests` or individual test scripts.
- Interactive debugging:
  - Open `gaussianapprox.ipynb` for exploratory work and quick plots.
  - Use small sample kinematics from `tests/integrand_test.py` (Q=10 GeV, u~0.25, z~0.4) to reproduce and debug numerical issues.
- Numerical stability notes:
  - Many functions rely on vectorized NumPy broadcasting; watch shapes (radial vs Cartesian inputs).
  - Quadrupole uses small cutoffs (e.g. 1e-12) and explicit epsilon handling in the dipole exponent — preserve these patterns when refactoring.

## Integration points & external dependencies
- External packages: numpy, scipy (declared in `pyproject.toml`).
- Typical integration: wavefunction -> integrand -> target model (dipole/quadrupole) -> numerics/Integration_functions.
  - Example flow: LO FE photon WF --> LODISIntegrand (in `integrands/.../LO/integrand.py`) --> calls `dipole.S(r)` and `quadrupole_polar(...)` --> integrated by functions under `numerics/totalDIS/LO/`.

## Files to read first when editing or adding features
- `multipole_models/MV_models/dipole.py` — dipole primitive and exponent.
- `multipole_models/MV_models/gaussian_quadrupole.py` — quadrupole approximation; watch broadcasting.
- `wavefunctions/*/LO.py` — photon wavefunction implementations and signatures used by integrands.
- `integrands/totalDIS/LO/integrand.py` and `tests/integrand_test.py` — show how components are wired together and provide runnable examples.

## Common pitfalls for automated edits
- Do not change numeric epsilons or small-argument safeguards without tests — they control numerical stability.
- Keep vectorized implementations (avoid forcing scalar-only code paths) and maintain consistent coordinate conventions (radial vs 2D arrays).
- When renaming public model methods, update all call sites in `integrands/` and tests — imports rely on stable names like `S`, `S_r`, `quadrupole_polar`.

## If you need to extend or refactor
- Add a focused unit test (script-style tests are acceptable) that reproduces a numerical value at a sample kinematic point from `tests/integrand_test.py`.
- Run the smoke script before/after edits: `PYTHONPATH=. python tests/integrand_test.py`.

---
If anything above is unclear or you want more examples (e.g. typical numeric tolerances, plotting utilities, or a recommended pytest conversion), tell me which area to expand and I will iterate.
