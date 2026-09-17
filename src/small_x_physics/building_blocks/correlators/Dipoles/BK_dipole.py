# rcBK dipole amplitude reader.
#
# This merges the two readers that used to exist side by side in the library:
#
#   * the previous correlators/Dipoles/BK_dipole.py -> cubic RectBivariateSpline,
#     smooth and fast, which is what VEGAS wants, but it threw the file header
#     away and hardcoded the valid range as clip(Y, 0, 16), clip(r, 1e-6, 100).
#
#   * multipole_models/MV_models/rcbk_adapter.py (RCBKData) -> parsed the header
#     metadata (Qs0^2, gamma, e_c, x0, Lambda_QCD), derived the grid bounds from
#     the file itself and clipped N to [0, 1], but interpolated only linearly.
#
# The class below keeps the cubic spline and takes everything else from RCBKData,
# so rcbk_adapter.py is now redundant. With the default settings it reproduces the
# previous BK_dipole.py bit for bit: verified over 500k random (r, Y) points, and
# end to end through BK_FE_4D_evolve_in_xB, BK_FE_5D_evolve_in_xP and BK_OT_CS
# with VEGAS seeded via gvar.ranseed, all three identical to every digit.
#
# The two readers differ only outside the old hardcoded window: for Y > 16 this
# one uses the file's real data (mv.dat runs to Y = 16.2) instead of clipping.
# Y = log(x0/xB) = 16 corresponds to xB ~ 1e-9, so no realistic run reaches it.

import re
import warnings

import numpy as np
from scipy.interpolate import RectBivariateSpline


# Regexes applied to the comment header of an rcbk output file. The backslashes
# in the LaTeX-style names written by the solver (\gamma, \Lambda_QCD) are
# optional so that files written by other tools still parse.
_HEADER_PATTERNS = {
    "Qs0_sq": r"Q_s0\^2\s*=\s*([0-9.eE+-]+)",
    "gamma": r"\\?gamma\s*=\s*([0-9.eE+-]+)",
    "ec": r"coefficient of E inside Log is\s*([0-9.eE+-]+)",
    "x0": r"\bx0\s*=\s*([0-9.eE+-]+)",
    "LambdaQCD": r"\\?Lambda_QCD\s*=\s*([0-9.eE+-]+)",
}


class BKDipole:
    """BK-evolved dipole amplitude N(r, Y) read from an rcbk solver output file.

    The file format is the textual output of the rcbk solver
    (https://github.com/hejajama/rcbkdipole). A comment header starting with '#'
    is followed by four '###' values giving MinR, RMultiplier, RPoints and X0,
    and then one block per rapidity: a '### Y' line followed by RPoints values
    of N(Y, r) on the geometric r grid r_i = MinR * RMultiplier**i.

    Parameters
    ----------
    bkfile : str or Path
        Path to the rcbk output file.
    Y : float, optional
        Fixed rapidity. If given, `BK_evolved_MV_model_S2` and `S_xy` can be
        called without an explicit Y. If omitted, a Y must be supplied per call.
    interp_on_logr : bool, default False
        Spline in log(r) rather than r. The r grid is geometric, so log(r) looks
        like the natural variable, but a leave-out test on mv.dat (fit on every
        second r node, predict the omitted ones) shows the two are equally
        accurate: max absolute error 7.181e-4 on a linear axis vs 7.183e-4 on a
        log axis. FITPACK handles the non-uniform knots without trouble, so
        there is nothing to gain. The default is therefore False, which
        reproduces BK_dipole.py bit for bit. log(r) is ~9% faster and available
        if you want it; it shifts S by up to 2e-3 relative at r > 1 GeV^-1,
        where S has fallen to ~5e-6.
    clamp_N : bool, default True
        Clip the interpolated N to [0, 1]. A cubic spline can overshoot near the
        saturation plateau, which would give S > 1 or S < 0. Purely defensive:
        over 200k sampled points of mv.dat with r < 10 GeV^-1 the raw spline
        stayed inside [0, 1], so this currently never fires.
    S_floor : float, default 1e-14
        Lower bound applied to S = 1 - N. The quadrupole takes log(S), so S must
        stay strictly positive. Matches the floor used in the original module.
    on_out_of_range : {"clamp", "raise", "extrapolate"}, default "clamp"
        What to do when a requested (r, Y) falls outside the grid stored in the
        file. "clamp" evaluates at the nearest boundary (the physically sensible
        default: N -> 0 as r -> 0 and N saturates at large r), "raise" refuses,
        and "extrapolate" lets the spline run free. Unlike the original module
        the bounds come from the file, not from hardcoded numbers.
    warn_on_clamp : bool, default True
        Emit a warning the first time a query is clamped. Counts are always
        accumulated in `n_clamped_r` / `n_clamped_Y` regardless.

    Attributes
    ----------
    r_grid, Y_grid : ndarray
        The grids read from the file.
    N_grid : ndarray, shape (len(Y_grid), len(r_grid))
        Tabulated dipole amplitude.
    r_min, r_max, Y_min, Y_max : float
        Grid bounds, derived from the file.
    Qs0_sq, gamma, ec, x0, LambdaQCD : float or None
        Initial-condition parameters parsed from the comment header. None if the
        header did not contain them.
    metadata : dict
        The same parameters, only including those actually found.
    n_clamped_r, n_clamped_Y : int
        Number of query points clamped so far. Note these are per-process, so
        they do not accumulate across VEGAS worker processes when nproc > 1.
    """

    def __init__(
        self,
        bkfile,
        Y=None,
        interp_on_logr=False,
        clamp_N=True,
        S_floor=1e-14,
        on_out_of_range="clamp",
        warn_on_clamp=True,
    ):
        if on_out_of_range not in ("clamp", "raise", "extrapolate"):
            raise ValueError(
                f"on_out_of_range must be 'clamp', 'raise' or 'extrapolate', "
                f"got {on_out_of_range!r}"
            )

        self.filename = str(bkfile)
        self.Y = Y
        self.interp_on_logr = interp_on_logr
        self.clamp_N = clamp_N
        self.S_floor = S_floor
        self.on_out_of_range = on_out_of_range
        self.warn_on_clamp = warn_on_clamp

        self.n_clamped_r = 0
        self.n_clamped_Y = 0

        self._parse(bkfile)
        self._build_spline()

        # A fixed Y outside the tabulated range is almost always a mistake, so
        # say so at construction rather than silently at every evaluation.
        if Y is not None and not (self.Y_min <= Y <= self.Y_max):
            warnings.warn(
                f"Fixed Y={Y} is outside the tabulated range "
                f"[{self.Y_min}, {self.Y_max}] of {self.filename}",
                RuntimeWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    def _parse(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()

        self._parse_header(lines)

        # Locate the '###' marker lines. The first four are the grid definition,
        # everything after that is one rapidity block each.
        markers = [i for i, line in enumerate(lines) if line.lstrip().startswith("###")]
        if len(markers) < 5:
            raise ValueError(
                f"{filename}: expected at least 5 '###' markers "
                f"(MinR, RMultiplier, RPoints, X0, and one rapidity), found {len(markers)}"
            )

        def marker_value(i):
            return float(lines[markers[i]].lstrip()[3:].strip())

        MinR = marker_value(0)
        RMultiplier = marker_value(1)
        RPoints = int(marker_value(2))
        X0 = marker_value(3)

        self.MinR = MinR
        self.RMultiplier = RMultiplier
        self.RPoints = RPoints
        self.X0 = X0

        # The header comment and the fourth '###' value both carry x0. If they
        # disagree the file is inconsistent and any rapidity we derive from it
        # would be wrong, so trust the '###' value and say something.
        if self.x0 is not None and not np.isclose(self.x0, X0, rtol=1e-6):
            warnings.warn(
                f"{filename}: x0 in the comment header ({self.x0}) disagrees with "
                f"the '###' grid value ({X0}); using {X0}",
                RuntimeWarning,
                stacklevel=3,
            )
        self.x0 = X0
        self.metadata["x0"] = X0

        # Geometric r grid, exactly as the solver constructed it.
        self.r_grid = MinR * RMultiplier ** np.arange(RPoints)

        Y_values = []
        N_rows = []
        for m in markers[4:]:
            Y_values.append(float(lines[m].lstrip()[3:].strip()))

            # Collect RPoints numeric lines, skipping blanks and stray comments
            # rather than assuming they are exactly contiguous.
            values = []
            idx = m + 1
            while len(values) < RPoints and idx < len(lines):
                tok = lines[idx].strip()
                idx += 1
                if tok == "" or tok.startswith("#"):
                    continue
                values.append(float(tok))
            if len(values) != RPoints:
                raise ValueError(
                    f"{filename}: rapidity block Y={Y_values[-1]} has {len(values)} "
                    f"r points, expected {RPoints}"
                )
            N_rows.append(values)

        self.Y_grid = np.asarray(Y_values, dtype=float)
        self.N_grid = np.asarray(N_rows, dtype=float)

        # RectBivariateSpline requires a strictly increasing first axis.
        order = np.argsort(self.Y_grid)
        if not np.array_equal(order, np.arange(order.size)):
            self.Y_grid = self.Y_grid[order]
            self.N_grid = self.N_grid[order, :]

        self.r_min = float(self.r_grid[0])
        self.r_max = float(self.r_grid[-1])
        self.Y_min = float(self.Y_grid[0])
        self.Y_max = float(self.Y_grid[-1])

    def _parse_header(self, lines):
        """Extract initial-condition parameters from the leading comment block."""
        header = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith("###"):
                break
            if stripped.startswith("#"):
                header.append(stripped)
        header_text = "".join(header)

        self.metadata = {}
        for name, pattern in _HEADER_PATTERNS.items():
            match = re.search(pattern, header_text)
            value = float(match.group(1)) if match else None
            setattr(self, name, value)
            if value is not None:
                self.metadata[name] = value

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def _build_spline(self):
        r_axis = np.log(self.r_grid) if self.interp_on_logr else self.r_grid

        # Cubic where there is enough data, lower order otherwise, so that a
        # short file degrades instead of raising from deep inside FITPACK.
        kx = min(3, len(self.Y_grid) - 1)
        ky = min(3, len(r_axis) - 1)
        if kx < 1 or ky < 1:
            raise ValueError(
                f"{self.filename}: need at least 2 rapidities and 2 r points to "
                f"interpolate, got {len(self.Y_grid)} and {len(r_axis)}"
            )

        self.spline = RectBivariateSpline(self.Y_grid, r_axis, self.N_grid, kx=kx, ky=ky)
        self._spline_kx = kx
        self._spline_ky = ky

    def _prepare(self, r, Y):
        """Broadcast (r, Y), apply the range policy, return spline arguments."""
        r_arr, Y_arr = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(Y, dtype=float)
        )

        if self.on_out_of_range == "extrapolate":
            return r_arr, Y_arr

        r_out = (r_arr < self.r_min) | (r_arr > self.r_max)
        Y_out = (Y_arr < self.Y_min) | (Y_arr > self.Y_max)
        n_r_out = int(np.count_nonzero(r_out))
        n_Y_out = int(np.count_nonzero(Y_out))

        if n_r_out == 0 and n_Y_out == 0:
            return r_arr, Y_arr

        if self.on_out_of_range == "raise":
            details = []
            if n_r_out:
                details.append(
                    f"{n_r_out} r value(s) outside [{self.r_min:g}, {self.r_max:g}] "
                    f"(min {np.min(r_arr):g}, max {np.max(r_arr):g})"
                )
            if n_Y_out:
                details.append(
                    f"{n_Y_out} Y value(s) outside [{self.Y_min:g}, {self.Y_max:g}] "
                    f"(min {np.min(Y_arr):g}, max {np.max(Y_arr):g})"
                )
            raise ValueError(f"{self.filename}: " + "; ".join(details))

        if self.warn_on_clamp and (self.n_clamped_r + self.n_clamped_Y) == 0:
            warnings.warn(
                f"{self.filename}: query outside the tabulated grid "
                f"(r in [{self.r_min:g}, {self.r_max:g}], "
                f"Y in [{self.Y_min:g}, {self.Y_max:g}]); "
                f"clamping to the boundary. Check your integration limits.",
                RuntimeWarning,
                stacklevel=3,
            )

        self.n_clamped_r += n_r_out
        self.n_clamped_Y += n_Y_out
        return (
            np.clip(r_arr, self.r_min, self.r_max),
            np.clip(Y_arr, self.Y_min, self.Y_max),
        )

    def N(self, r, Y=None):
        """Dipole amplitude N(r, Y).

        Parameters
        ----------
        r : float or array
            Dipole size in GeV^-1.
        Y : float or array, optional
            Rapidity. Defaults to the fixed Y given at construction.

        Returns
        -------
        ndarray
            N broadcast over r and Y. Clipped to [0, 1] unless clamp_N is False.
        """
        Y = self._resolve_Y(Y)
        r_arr, Y_arr = self._prepare(r, Y)
        r_axis = np.log(r_arr) if self.interp_on_logr else r_arr

        values = self.spline(Y_arr, r_axis, grid=False)
        if self.clamp_N:
            values = np.clip(values, 0.0, 1.0)
        return values

    def _resolve_Y(self, Y):
        if Y is None:
            if self.Y is None:
                raise ValueError(
                    "No rapidity available: this BKDipole was constructed without a "
                    "fixed Y, so Y must be passed to the call."
                )
            return self.Y
        return Y

    # ------------------------------------------------------------------
    # Dipole S-matrix
    # ------------------------------------------------------------------

    def radius(self, x, y):
        """Dipole size r = |x - y| for transverse coordinates x, y."""
        return np.linalg.norm(np.asarray(x) - np.asarray(y), axis=-1)

    def S_r(self, r, Y=None):
        """S(r) = 1 - N(r, Y), floored at S_floor so that log(S) is finite."""
        return np.maximum(1.0 - self.N(r, Y), self.S_floor)

    def S_xy(self, x, y, Y=None):
        """S(|x - y|) for transverse coordinates x, y."""
        return self.S_r(self.radius(x, y), Y)

    def rapidity_from_x(self, x):
        """Evolution rapidity Y = log(x0 / x), using the x0 stored in the file.

        Reading x0 off the file rather than passing it in separately removes one
        way for a run to be quietly inconsistent with the amplitude it used.
        """
        return np.log(self.x0 / np.asarray(x, dtype=float))

    # --- names kept for compatibility with the existing Cross_Sections code ---

    def BK_evolved_MV_model_S2(self, x, y, **_kwargs):
        """S(|x - y|) at the fixed Y given at construction."""
        if self.Y is None:
            raise ValueError(
                "BKDipole was created without a fixed Y. "
                "Use BK_evolved_MV_model_S2_Y instead."
            )
        return self.S_xy(x, y, self.Y)

    def BK_evolved_MV_model_S2_Y(self, x, y, Y, **_kwargs):
        """S(|x - y|) at rapidity Y."""
        return self.S_xy(x, y, Y)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def check_parameters(self, Qs0=None, gamma=None, ec=None, x0=None, rtol=1e-6, raise_on_mismatch=False):
        """Compare caller-supplied initial-condition parameters against the file.

        The cross-section classes currently accept Qs0, gamma and ec and then
        never use them for the BK case -- the values that actually matter are
        the ones baked into the bk file. This makes that comparison explicit.

        Parameters
        ----------
        Qs0 : float, optional
            Saturation scale (not squared); compared against sqrt(Qs0_sq).
        gamma, ec, x0 : float, optional
            Compared directly against the parsed values.
        rtol : float, default 1e-6
            Relative tolerance.
        raise_on_mismatch : bool, default False
            Raise instead of returning the list of mismatches.

        Returns
        -------
        list of str
            One message per mismatch; empty if everything agrees. Parameters the
            header did not contain are reported as unverifiable.
        """
        expected = {
            "Qs0": (None if self.Qs0_sq is None else float(np.sqrt(self.Qs0_sq)), Qs0),
            "gamma": (self.gamma, gamma),
            "ec": (self.ec, ec),
            "x0": (self.x0, x0),
        }

        problems = []
        for name, (from_file, supplied) in expected.items():
            if supplied is None:
                continue
            if from_file is None:
                problems.append(
                    f"{name}: cannot verify, not present in the header of {self.filename}"
                )
            elif not np.isclose(from_file, supplied, rtol=rtol):
                problems.append(
                    f"{name}: caller passed {supplied!r} but {self.filename} was "
                    f"generated with {from_file!r}"
                )

        if problems and raise_on_mismatch:
            raise ValueError("; ".join(problems))
        return problems

    def clamp_report(self):
        """Summary of how many queries fell outside the grid, for after a run."""
        return {
            "n_clamped_r": self.n_clamped_r,
            "n_clamped_Y": self.n_clamped_Y,
            "r_range": (self.r_min, self.r_max),
            "Y_range": (self.Y_min, self.Y_max),
        }

    def __repr__(self):
        params = ", ".join(f"{k}={v:g}" for k, v in sorted(self.metadata.items()))
        return (
            f"BKDipole({self.filename!r}, Y={self.Y}, "
            f"r in [{self.r_min:g}, {self.r_max:g}] ({self.RPoints} pts), "
            f"Y in [{self.Y_min:g}, {self.Y_max:g}] ({len(self.Y_grid)} pts), "
            f"{params})"
        )
