# Test data

## mv.dat

A small rcBK solution used only for testing and for the example runs in
`test_bk_integration.py`. It is not physics input for any published result, and
nothing in `src/small_x_physics` reads it implicitly — scripts either take an
explicit `--bkfile` argument or resolve this path relative to the repository
root.

It was produced by the rcBK solver (https://github.com/hejajama/rcbkdipole) and
its own comment header records the full command line and the parameters:

| parameter | value |
|---|---|
| initial condition | MV |
| Q_s0^2 | 0.104 GeV^2 |
| gamma | 1 |
| e_c (coefficient of E inside the log) | 1 |
| x0 | 0.01 |
| Lambda_QCD | 0.241 GeV |
| running coupling | Balitsky, C^2 = 14.5 |
| r grid | 1e-6 to 100 GeV^-1, 400 points, multiplier 1.047249 |
| Y grid | 0 to 16.2, step 0.2 |

`BKDipole` parses that header, so those values are available at runtime as
`dipole.Qs0_sq`, `dipole.gamma`, `dipole.ec`, `dipole.x0` and
`dipole.LambdaQCD`, and `dipole.check_parameters(...)` will compare them against
whatever a caller passed in.

Usage:

```
python3 src/small_x_physics/Cross_Sections/InclusiveDIS/withBK/test_bk_integration.py \
    --bkfile data/mv.dat --xB 1e-3
```
