"""Anytime-valid e-process reanalysis of the DESI DR1 -> DR2 BAO data.

    from desi_evalue import analyses
    dr1, dr2 = analyses.load_releases()
    analyses.running_mixture(dr1, dr2)     # {'M_DR1': 1.05, 'M_DR2': 33.97, ...}
"""

from . import analyses, constants
from .cosmology import BASELINE, Background
from .data import BAOData, load
from .evalues import Mixture, markov_p, mixture_e, rejects
from .sequential import JointEProcess

__all__ = ["analyses", "constants", "Background", "BASELINE", "BAOData", "load",
           "Mixture", "mixture_e", "markov_p", "rejects", "JointEProcess"]
