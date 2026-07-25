# desi-evalue-paper

Code for "A Sequentially-Valid Reanalysis of DESI's Dynamical Dark Energy
Signal" (Kim, Mota & Tamosiunas, 2026): an anytime-valid e-process applied to
the DESI DR1 → DR2 BAO preference for dynamical dark energy.

## Use

```bash
pip install -r requirements.txt
```

```python
from desi_evalue import analyses as A

dr1, dr2 = A.load_releases()
A.running_mixture(dr1, dr2)       # M_DR1 = 1.05, M_DR2 = 33.97
A.prior_sensitivity(dr1, dr2)     # Table 1
A.localisation(dr2)               # Table 2
A.joint_sequential(dr1, dr2)      # M_joint = 66.7, anytime-valid p = 0.015
A.background_sensitivity(dr1, dr2)
```

Functions return their numbers rather than printing them.

## Modules

- `constants.py` — assumptions, fixed parameters, alternative choices
- `cosmology.py` — w0waCDM distances, Gaussian BAO likelihood
- `data.py` — DR1/DR2 loading, bin structure, cross-release matching
- `evalues.py` — mixture, LOO, split, universal-inference e-values
- `sequential.py` — joint (DR1, DR2) martingale, year-scaling model
- `analyses.py` — the paper's results

## Notes

The prior grid is part of the test specification. A uniform mixture over a
finite pre-specified set of points is an exact e-value, so a different
resolution is a different but equally valid test; the Default box gives 33.97 at
30×30 and 35.75 at 120×120. The freezing box is small and its integrand peaks
sharply near one corner, so it needs a finer grid (`FREEZING_GRID_N`).

Leave-one-out and per-bin products fold at bin level, not measurement level.
The two measurements within a redshift bin are anti-correlated, and the
independence these constructions rely on holds only across bins.

## Tests

```bash
python -m pytest tests/
```

## Data

DESI DR1 and DR2 BAO measurements (mean and covariance) from
[CobayaSampler/bao_data](https://github.com/CobayaSampler/bao_data).
