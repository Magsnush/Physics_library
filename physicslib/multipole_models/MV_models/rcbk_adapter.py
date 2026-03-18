import numpy as np
from scipy.interpolate import RegularGridInterpolator

class RCBKData:
    """Load rcbk output file and provide interpolated N(Y, r) and S(Y, r).

    The file format is the textual output of the rcbk solver. The header
    contains lines starting with '###' giving MinR, RMultiplier, RPoints
    and X0, followed by blocks for each rapidity with a '### Y' header and
    RPoints floating values for N(y, r).
    """
    def __init__(self, filename, interp_on_logr=True, fill_value=None):
        self.filename = filename
        self.interp_on_logr = interp_on_logr
        self.fill_value = fill_value
        self._load_file(filename)
        self._build_interpolator()

    def _load_file(self, filename):
        y_vals = []
        data_rows = []
        with open(filename, 'r') as f:
            lines = f.readlines()

        idx = 0
        # skip leading comment lines starting with a single '#' but keep '###' header lines
        while idx < len(lines):
            s = lines[idx].lstrip()
            if s.startswith('###'):
                break
            if s.startswith('#'):
                idx += 1
                continue
            break

        def read_val():
            nonlocal idx
            if idx >= len(lines):
                raise ValueError("Unexpected EOF while parsing rcbk file")
            tok = lines[idx].strip()
            if not tok.startswith('###'):
                raise ValueError(f"Expected '###' header at line {idx+1}, got: {tok[:40]}")
            val = float(tok[3:].strip())
            idx += 1
            return val

        MinR  = read_val()
        RMult = read_val()
        RPoints = int(read_val())
        X0 = read_val()

        self.MinR = MinR
        self.RMultiplier = RMult
        self.RPoints = RPoints
        self.X0 = X0

        while idx < len(lines):
            line = lines[idx].strip()
            if line == '':
                idx += 1
                continue
            if not line.startswith('###'):
                idx += 1
                continue
            yval = float(line[3:].strip())
            idx += 1
            values = []
            while len(values) < RPoints and idx < len(lines):
                tok = lines[idx].strip()
                idx += 1
                if tok == '' or tok.startswith('#'):
                    continue
                values.append(float(tok))
            if len(values) != RPoints:
                raise ValueError(f"Not enough r points for Y = {yval}")
            y_vals.append(yval)
            data_rows.append(values)

        self.y_vals = np.array(y_vals)
        r_i = MinR * (RMult ** np.arange(RPoints))
        self.r_vals = np.array(r_i)
        self.N_table = np.array(data_rows)

        # ensure y sorted
        sort_idx = np.argsort(self.y_vals)
        if not np.all(sort_idx == np.arange(len(sort_idx))):
            self.y_vals = self.y_vals[sort_idx]
            self.N_table = self.N_table[sort_idx, :]

    def _build_interpolator(self):
        if self.interp_on_logr:
            r_axis = np.log(self.r_vals)
            points = (self.y_vals, r_axis)
            values = self.N_table
            self._interp = RegularGridInterpolator(points, values,
                                                  bounds_error=(self.fill_value is None),
                                                  fill_value=self.fill_value)
        else:
            points = (self.y_vals, self.r_vals)
            self._interp = RegularGridInterpolator(points, self.N_table,
                                                  bounds_error=(self.fill_value is None),
                                                  fill_value=self.fill_value)

    def N(self, Y, r):
        Ys = np.atleast_1d(Y)
        rs = np.atleast_1d(r)
        YY, RR = np.broadcast_arrays(Ys.reshape(-1,1), rs.reshape(1,-1))
        pts = np.stack([YY.ravel(), (np.log(RR.ravel()) if self.interp_on_logr else RR.ravel())], axis=-1)
        vals = self._interp(pts).reshape(YY.shape)
        if np.isscalar(Y) and np.isscalar(r):
            return float(vals.ravel()[0])
        return vals

    def S(self, Y, r):
        return 1.0 - self.N(Y, r)
