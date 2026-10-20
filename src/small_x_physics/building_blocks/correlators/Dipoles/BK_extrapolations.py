# Continuations of an rcBK dipole above the x0 where its evolution starts.
#
# An rcBK solution N(r, Y) covers Y = ln(x0/x) >= 0 only. Cross sections that read
# the dipole at an x above x0 -- the 5D finite-energy cross section does, at
# x_g = xB (1 + M^2/Q^2) -- need an assumption about what N is there. BKDipole on
# its own clamps to Y = 0, i.e. holds N at the initial condition ("frozen"). The
# classes below are alternatives. Below x0 every one of them is the ordinary
# BKDipole, so results differ only through the region the solution says nothing
# about, and the spread between them is a measure of that uncertainty.
#
# None of these is evolution: BK is an initial-value problem forward in Y, with
# its initial condition defined at x0. They are models.

import numpy as np

from small_x_physics.building_blocks.correlators.Dipoles.BK_dipole import BKDipole


class ExponentialBackwardBKDipole(BKDipole):
    """Continue N below Y = 0 at the solution's own rate.

        N(r, x) = N(r, x0) (x0/x)^lambda(r),
        lambda(r) = [ln N(r, Y_1) - ln N(r, Y_0)] / (Y_1 - Y_0)       for x > x0,

    with Y_0, Y_1 the first two rapidities of the grid. The continuation matches
    the solution at x0 in value and in slope, treats each dipole size with its own
    rate, and keeps N strictly positive. Where N vanishes at either of the first
    two rapidities it falls back to the frozen value. Within the grid this is the
    ordinary BKDipole, bit for bit.

    Multiplicative rather than additive on purpose: an additive continuation drives
    N through zero at small r, where N ~ 1e-13 and its slope is of the same order.
    """

    def N(self, r, Y=None):
        Y = self._resolve_Y(Y)
        r_arr, Y_arr = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(Y, dtype=float)
        )

        below = Y_arr < self.Y_min
        in_range = super().N(r_arr, np.maximum(Y_arr, self.Y_min))
        if not np.any(below):
            return in_range

        Y0 = self.Y_min
        dY = float(self.Y_grid[1] - self.Y_grid[0])
        N0 = super().N(r_arr, np.full(r_arr.shape, Y0))
        N1 = super().N(r_arr, np.full(r_arr.shape, Y0 + dY))

        safe = (N0 > 0) & (N1 > 0)
        ratio = np.where(safe, N1 / np.where(N0 > 0, N0, 1.0), 1.0)
        extended = np.where(safe, N0 * ratio ** ((Y_arr - Y0) / dY), N0)

        return np.clip(np.where(below, extended, in_range), 0.0, 1.0)


class LargeXFalloffBKDipole(BKDipole):
    """Suppress the dipole at x0 by a large-x factor:

        N(r, x) = N(r, x0) [(1 - x)/(1 - x0)]^n        for x > x0.

    The factor is 1 at x0, so the dipole joins the solution without a jump, and it
    vanishes as x -> 1, as gluon densities do: they fall like (1 - x)^n near x = 1
    (quark counting gives n ~ 5 for gluons in the proton; IP-Sat uses 5.6), and the
    dipole inherits it since N ~ r^2 x g(x) for small r. The target therefore
    becomes transparent at x = 1, which the frozen and exponential continuations
    do not. n = 0 is frozen; as n grows the result tends to dropping the region
    above x0 altogether. Within the grid this is the ordinary BKDipole.

    x0 is the one recorded in the bk file, where its evolution starts; Y = 0 there,
    and x = x0 exp(-Y) above it.

    Parameters
    ----------
    bkfile : str or Path
    n : float
        The power of the falloff, e.g. 3, 5 or 7.
    **kwargs
        Passed on to BKDipole.
    """

    def __init__(self, bkfile, n, **kwargs):
        super().__init__(bkfile, **kwargs)
        if self.Y_min != 0:
            raise ValueError(
                f"{bkfile}: the grid starts at Y = {self.Y_min}, not 0, so its bottom "
                f"row is not N(r, x0)"
            )
        self.n = float(n)

    def N(self, r, Y=None):
        Y = self._resolve_Y(Y)
        r_arr, Y_arr = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(Y, dtype=float)
        )

        # In range this is the ordinary dipole; above x0 it is N(r, x0).
        N_clamped = super().N(r_arr, np.maximum(Y_arr, self.Y_min))
        above_x0 = Y_arr < self.Y_min
        if not np.any(above_x0):
            return N_clamped

        x = self.x0 * np.exp(-Y_arr)
        factor = np.clip((1 - x) / (1 - self.x0), 0.0, 1.0) ** self.n
        return np.where(above_x0, N_clamped * factor, N_clamped)
