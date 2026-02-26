"""
quadrupole_override_test.py

Verify that when building the LODIS integrand with
`z_target_override=0.0` the quadrupole callable used inside the integrand
receives z=0 for its evaluation even if the sampled z is non-zero. This
test intentionally checks only that the quadrupole sees z=0; it does not
modify photon wavefunctions or Jacobian behaviour.

Run from the repository root::

    python3 tests/quadrupole_override_test.py

"""

import os
import sys
import numpy as np

try:
    from physicslib.numerics.totalDIS.LO.Integration_functions_4D import QuadrupolePolarWrapper
except Exception:
    # Fallback to local numerics LO path
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    numerics_lo = os.path.join(repo_root, "physicslib", "numerics", "totalDIS", "LO")
    if os.path.isdir(numerics_lo):
        sys.path.insert(0, numerics_lo)
        from Integration_functions_4D import QuadrupolePolarWrapper
    else:
        raise


class DummyQuad:
    """A minimal quadrupole model that echoes the z argument it receives.

    This avoids calling into the GaussianQuadrupole implementation (which may
    contain unrelated bugs) and directly tests the wrapper behaviour.
    """
    def quadrupole_polar(self, u, up, z, theta, largeNc=False):
        # Return the passed z so callers can inspect whether the wrapper
        # replaced it or left it unchanged.
        return np.array([z])


def main():
    dummy = DummyQuad()

    # wrapper with no override
    wrap_none = QuadrupolePolarWrapper(dummy, largeNc=False, z_override=None)
    # wrapper that forces z->0
    wrap_z0 = QuadrupolePolarWrapper(dummy, largeNc=False, z_override=0.0)

    # test inputs
    u = np.array([0.2])
    up = np.array([0.6])
    theta = 0.7
    z_sample = 0.3

    out_none = wrap_none(u, up, z_sample, theta)
    out_z0_called = wrap_z0(u, up, z_sample, theta)
    out_z0_direct = wrap_z0(u, up, 0.0, theta)

    v_none = float(np.array(out_none).reshape(-1)[0])
    v_z0_called = float(np.array(out_z0_called).reshape(-1)[0])
    v_z0_direct = float(np.array(out_z0_direct).reshape(-1)[0])

    print(f"wrapper (no override) received z -> {v_none}")
    print(f"wrapper (override=0.0) called with z_sample -> {v_z0_called}")
    print(f"wrapper (override=0.0) called with explicit z=0 -> {v_z0_direct}")

    assert np.isclose(v_none, z_sample), "Expected no-override wrapper to pass sampled z through"
    assert np.isclose(v_z0_called, 0.0), "Expected override wrapper to replace sampled z with 0.0"
    assert np.isclose(v_z0_called, v_z0_direct), "Override wrapper must behave like calling quadrupole with explicit z=0"

    print("Quadrupole override wrapper behaves as expected.")


if __name__ == "__main__":
    main()
