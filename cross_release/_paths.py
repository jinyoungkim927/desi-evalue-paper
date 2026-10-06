"""Paths shared by the scripts in cross_release/ and figures/. Everything is resolved relative
to the repository root, so the scripts run from any location and any working directory.

  code/                       the analysis modules imported here (data_loader, evalue_analysis,
                              eprocess_joint, eprocess_hierarchical_mc, literature_priors, ...)
  data/                       the DESI DR1 and DR2 BAO mean vectors and covariances
  cross_release/inputs/       outputs of record of the Monte Carlo scripts in
                              cross_release/simulations/, their logs (logs/), and in results/ the
                              JSON written by code/*.py from which the paper's numbers are computed
  cross_release/numbers/      the numbers file of record, numbers.json
  results/                    output directory (not tracked): the JSON written by code/*.py, the
                              working copy of numbers.json, logs, simulations/ and figures/

A script that needs a results/*.json file reads the copy in results/ if the corresponding
code/*.py script has been run, else the copy of record in cross_release/inputs/results/
(result_json). A script that updates a section of the numbers file starts from the working
copy results/numbers.json if it exists, else from the copy of record (load_numbers), and writes
the working copy (save_numbers); the copy of record is never overwritten.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
CROSS_RELEASE = ROOT / 'cross_release'
INPUTS = CROSS_RELEASE / 'inputs'
RECORD = CROSS_RELEASE / 'numbers'
OUT_DIR = RESULTS
FIGURES = RESULTS / 'figures'
NUMBERS = OUT_DIR / 'numbers.json'


def add_code_path():
    """Make the modules in code/ importable by their bare names, as code/*.py do."""
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))


def result_json(name):
    """results/<name> if present (written by code/*.py), else the copy of record."""
    p = RESULTS / name
    if not p.exists():
        p = INPUTS / 'results' / name
    with open(p) as fh:
        return json.load(fh)


def load_numbers():
    """The working copy results/numbers.json if it exists, else the copy of record."""
    p = NUMBERS
    if not p.exists():
        p = RECORD / 'numbers.json'
    with open(p) as fh:
        return json.load(fh)


def save_numbers(obj):
    """Write the working copy results/numbers.json (never the copy of record)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(NUMBERS, 'w') as fh:
        json.dump(obj, fh, indent=1)
