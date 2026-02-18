# physicslib

Reusable cross-section physics building blocks used for photon–proton calculations.

This repository provides implementations of photon wavefunctions, MV-model dipole and quadrupole correlators, and integrands intended for numerical integration and analysis.

Quick start
1. Create and activate a Python environment (recommended).
2. Install minimal runtime dependencies:

```bash
pip install -e .
```

3. Run a quick smoke test without installing (from repo root):

```bash
PYTHONPATH=. python tests/integrand_test.py
```

Project layout (high level)
- `physicslib/` — main package
	- `wavefunctions/` — photon wavefunctions (FE/OT, LO/NLO)
	- `multipole_models/` — dipole & quadrupole MV-models
	- `integrands/` — physics integrands wired to wavefunctions & target models
	- `numerics/` — numerical integration helpers and scripts

Notes for publishing to GitHub
- This project already includes a `LICENSE` (MIT) and a `.gitignore` suitable for Python projects.
- To push to GitHub, add a remote and push the local branch (example commands shown after local commit).

License
This project is available under the MIT License — see `LICENSE`.

If you want, I can also:
- Add a GitHub Actions workflow for CI (run tests with pytest).
- Create a small `CONTRIBUTING.md` and PR checklist.
- Add a `pyproject`/packaging badge and repository metadata for GitHub.
