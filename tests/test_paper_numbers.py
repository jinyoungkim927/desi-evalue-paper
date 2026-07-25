"""Pins the numbers quoted in the paper."""

import pytest

from desi_evalue import analyses as A


@pytest.fixture(scope="module")
def releases():
    return A.load_releases()


def test_running_mixture(releases):
    r = A.running_mixture(*releases)
    assert r["M_DR1"] == pytest.approx(1.05, abs=0.01)
    assert r["M_DR2"] == pytest.approx(33.97, abs=0.01)
    assert r["markov_p"] == pytest.approx(0.029, abs=0.001)
    assert r["rejects"]


def test_likelihood_geometry(releases):
    g = A.likelihood_geometry(releases[1])
    assert g["mle"][0] == pytest.approx(-0.856, abs=0.001)
    assert g["mle"][1] == pytest.approx(-0.430, abs=0.001)
    assert g["delta_chi2"] == pytest.approx(16.86, abs=0.01)
    assert g["wilks_sigma"] == pytest.approx(3.70, abs=0.01)
    assert g["delta_mle"] == pytest.approx(4.06, abs=0.01)


@pytest.mark.parametrize("prior, m_dr1, m_dr2", [
    ("Narrow", 2.60, 144.0), ("Default", 1.05, 33.97),
    ("Wide", 0.320, 10.7), ("Ong", 0.160, 4.81),
    ("REGROW delta=1", 1.50, 7.58), ("REGROW delta=2", 3.02, 72.5),
    ("REGROW delta=3", 5.25, 297.0),
])
def test_table_one(releases, prior, m_dr1, m_dr2):
    row = next(r for r in A.prior_sensitivity(*releases) if r["prior"] == prior)
    assert row["M_DR1"] == pytest.approx(m_dr1, rel=0.01)
    assert row["M_DR2"] == pytest.approx(m_dr2, rel=0.01)


def test_localisation(releases):
    loc = A.localisation(releases[1])
    assert loc["loo"]["average"] == pytest.approx(10.17, abs=0.01)
    assert loc["loo"]["per_bin"]["LRG2"]["e"] == pytest.approx(55.98, abs=0.01)
    assert loc["loo_average_without_LRG2"] == pytest.approx(2.54, abs=0.01)
    assert loc["M_DR2_without_LRG2"] == pytest.approx(0.49, abs=0.01)
    assert loc["per_bin_regrow"]["product"] == pytest.approx(4.46, abs=0.01)
    assert loc["per_bin_regrow"]["mean"] == pytest.approx(1.52, abs=0.01)
    assert loc["lrg2_share"] == pytest.approx(0.786, abs=0.001)
    # The look-elsewhere-corrected statistics must not reach the threshold.
    assert loc["per_bin_regrow"]["product"] < 20
    assert loc["per_bin_regrow"]["mean"] < 20


def test_joint_martingale(releases):
    j = A.joint_sequential(*releases)
    assert j["M_joint"] == pytest.approx(66.7, abs=0.1)
    assert j["incremental"] == pytest.approx(63.3, abs=0.1)
    assert j["anytime_valid_p"] == pytest.approx(0.015, abs=0.001)
    assert j["cov_y23_min_eigenvalue"] == pytest.approx(7.3e-3, rel=0.05)
    # The snapshot is the conservative member of the pair.
    assert j["M_snapshot"] < j["M_joint"]


def test_year_weight_robustness(releases):
    r = A.year_weight_robustness(*releases)
    values = list(r["by_alpha"].values())
    assert min(values) == pytest.approx(65, abs=1)
    assert max(values) == pytest.approx(131, abs=1)
    assert r["max_admissible_alpha"] == pytest.approx(0.470, abs=0.001)


@pytest.mark.parametrize("column, m_dr2, no_lrg2", [
    ("baseline", 33.97, 0.49), ("TT,TE,EE+lowE+lensing", 381.7, 2.37),
    ("+lensing+BAO", 8.70, 0.22), ("TT,TE,EE+lowE", 1417.0, 5.55),
])
def test_background_sensitivity(releases, column, m_dr2, no_lrg2):
    rows = A.background_sensitivity(*releases)
    assert rows[column]["M_DR2"] == pytest.approx(m_dr2, rel=0.01)
    assert rows[column]["M_DR2_without_LRG2"] == pytest.approx(no_lrg2, rel=0.01)
    # No column rejects once LRG2 is removed.
    assert rows[column]["M_DR2_without_LRG2"] < 20


def test_background_spread(releases):
    assert A.background_sensitivity(*releases)["spread"] == pytest.approx(163, rel=0.02)


def test_physical_priors(releases):
    p = A.physical_priors(*releases)
    assert p["thawing"]["M_DR2"] == pytest.approx(1.1e3, rel=0.05)
    assert p["thawing"]["M_DR1"] == pytest.approx(15, rel=0.05)
    # Freezing does not reject; see FREEZING_GRID_N on why this is not 21.
    assert p["freezing"]["M_DR2"] == pytest.approx(14.8, rel=0.02)
    assert p["freezing"]["M_DR2"] < 20


def test_shell_family(releases):
    m = A.minimum_concentration(releases[1])
    assert m["delta_chi2"] == pytest.approx(16.5, abs=0.1)
    assert m["roots"][0] == pytest.approx(0.87, abs=0.01)
    assert m["roots"][1] == pytest.approx(13.5, abs=0.1)
    assert m["delta_max"] == pytest.approx(2.7, abs=0.05)
    s = A.shell_vs_regrow(releases[1])
    assert s["shell"] == pytest.approx(30.9, rel=0.01)
    assert s["regrow"] == pytest.approx(7.6, rel=0.01)


def test_data_split_is_underpowered(releases):
    assert A.data_split(releases[1])["e"] == pytest.approx(1.43, abs=0.01)


def test_dr3_forecast_separates_only_on_existing_bins(releases):
    f = A.dr3_forecast(releases[1])
    # New high-z bins barely discriminate between the two truths.
    assert f["new_bins_w0wa"] == pytest.approx(f["new_bins_lcdm"], rel=0.2)
    # Tightening the bin that carries the evidence does.
    assert f["tightened_LRG2_w0wa"] > 20
    assert f["tightened_LRG2_lcdm"] < 1
