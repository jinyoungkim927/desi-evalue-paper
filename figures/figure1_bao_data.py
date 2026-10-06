#!/usr/bin/env python3
"""Figure 1 of the paper (Section 3): figure1_bao_data.pdf.

(a) the DESI DR2 BAO measurements as residuals from the Planck 2018 LCDM prediction, in
    units of each measurement's error, with the w0waCDM prediction at the DR2 BAO-only MLE
    (w0, wa) = (-0.856, -0.430) drawn as its offset from LCDM in the same units (sigma
    interpolated between bins);
(b) the (w0, wa) plane: the Default box, the Fisher 1, 2, 3 sigma ellipses of DR2 BAO at
    LCDM, the Caldwell-Linder thawing band on its analysis support w0 in [-1, -0.85]
    (edges at w0 = -0.85: wa from -0.45 to -0.15) and the freezing-quintessence box of
    Section 4.3, with LCDM and the DR2 MLE marked.

The DR2 mean vector is written out below (the values of
data/dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt to eight decimals) and the covariance is read
from data/dr2/ (the CobayaSampler/bao_data files). The distances use the fixed Planck 2018
background of the paper (h = 0.6766, Omega_m = 0.3111, r_d = 147.05 Mpc).

Run (from the repository root): python figures/figure1_bao_data.py
Writes results/figures/figure1_bao_data.pdf.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.integrate import quad

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cross_release'))
from _paths import DATA, FIGURES  # noqa: E402

FIGURES.mkdir(parents=True, exist_ok=True)

C_KM_S = 299792.458
H0_TO_HM = 100.0  # H0 = 100*h km/s/Mpc

# Fiducial cosmology (Planck 2018)
H_FID = 0.6766
OM_FID = 0.3111
RD_FID = 147.05
OR_FID = 9.0e-5
ODE_FID = 1.0 - OM_FID - OR_FID  # flat

# DR2 MLE (BAO-only, Planck-fixed background)
W0_MLE = -0.856
WA_MLE = -0.430


def E_z(z, w0=-1.0, wa=0.0):
    """Dimensionless Hubble parameter E(z) for w0waCDM."""
    a = 1.0 / (1.0 + z)
    de_exp = -3.0 * (1.0 + w0 + wa) * np.log(a) - 3.0 * wa * (1.0 - a)
    de = ODE_FID * np.exp(de_exp)
    matter = OM_FID * (1.0 + z)**3
    rad = OR_FID * (1.0 + z)**4
    return np.sqrt(matter + rad + de)


def DH(z, w0=-1.0, wa=0.0):
    """Hubble distance c/H(z) in Mpc."""
    return C_KM_S / (H0_TO_HM * H_FID * E_z(z, w0, wa))


def DC(z, w0=-1.0, wa=0.0):
    """Line-of-sight comoving distance in Mpc."""
    res, _ = quad(lambda zp: 1.0 / E_z(zp, w0, wa), 0, z)
    return C_KM_S / (H0_TO_HM * H_FID) * res


def DM(z, w0=-1.0, wa=0.0):
    """Transverse comoving distance in Mpc (flat universe, so DM = DC)."""
    return DC(z, w0, wa)


def DV(z, w0=-1.0, wa=0.0):
    """Volume-averaged distance [z DH DM^2]^(1/3) in Mpc."""
    return (z * DH(z, w0, wa) * DM(z, w0, wa)**2)**(1.0/3.0)


def predict_at(z, qty, w0=-1.0, wa=0.0):
    """BAO distance ratio (DM, DH, or DV) / r_d at redshift z."""
    if 'DM' in qty:
        return DM(z, w0, wa) / RD_FID
    elif 'DH' in qty:
        return DH(z, w0, wa) / RD_FID
    elif 'DV' in qty:
        return DV(z, w0, wa) / RD_FID
    raise ValueError(qty)


# DR2 BAO data (z, quantity, value)
DR2_BAO = [
    (0.295, 'DV_over_rs', 7.94167639),
    (0.510, 'DM_over_rs', 13.58758434),
    (0.510, 'DH_over_rs', 21.86294686),
    (0.706, 'DM_over_rs', 17.35069094),
    (0.706, 'DH_over_rs', 19.45534918),
    (0.934, 'DM_over_rs', 21.57563956),
    (0.934, 'DH_over_rs', 17.64149464),
    (1.321, 'DM_over_rs', 27.60085612),
    (1.321, 'DH_over_rs', 14.17602155),
    (1.484, 'DM_over_rs', 30.51190063),
    (1.484, 'DH_over_rs', 12.81699964),
    (2.330, 'DH_over_rs', 8.63154567),
    (2.330, 'DM_over_rs', 38.98897396),
]

COV = np.loadtxt(DATA / "dr2" / "desi_gaussian_bao_ALL_GCcomb_cov.txt")

z_eff = np.array([row[0] for row in DR2_BAO])
quantities = [row[1] for row in DR2_BAO]
data = np.array([row[2] for row in DR2_BAO])
sigma_diag = np.sqrt(np.diag(COV))

theory_lcdm = np.array([predict_at(z, q, w0=-1.0, wa=0.0)
                         for z, q in zip(z_eff, quantities)])
residuals = (data - theory_lcdm) / sigma_diag


# Smooth model curves for the MLE-vs-LCDM pulls in the left panel.
z_smooth = np.linspace(0.25, 2.4, 200)
DM_lcdm_smooth = np.array([DM(z) / RD_FID for z in z_smooth])
DH_lcdm_smooth = np.array([DH(z) / RD_FID for z in z_smooth])
DM_mle_smooth = np.array([DM(z, W0_MLE, WA_MLE) / RD_FID for z in z_smooth])
DH_mle_smooth = np.array([DH(z, W0_MLE, WA_MLE) / RD_FID for z in z_smooth])

# Sigma interpolation (DM and DH separately)
z_dm, s_dm = [], []
z_dh, s_dh = [], []
for i, (z, q) in enumerate(zip(z_eff, quantities)):
    if 'DM' in q:
        z_dm.append(z); s_dm.append(sigma_diag[i])
    elif 'DH' in q:
        z_dh.append(z); s_dh.append(sigma_diag[i])
z_dm, s_dm = np.array(z_dm), np.array(s_dm)
z_dh, s_dh = np.array(z_dh), np.array(s_dh)
idx_dm = np.argsort(z_dm); z_dm, s_dm = z_dm[idx_dm], s_dm[idx_dm]
idx_dh = np.argsort(z_dh); z_dh, s_dh = z_dh[idx_dh], s_dh[idx_dh]

sig_dm_smooth = np.interp(z_smooth, z_dm, s_dm)
sig_dh_smooth = np.interp(z_smooth, z_dh, s_dh)
pull_dm = (DM_mle_smooth - DM_lcdm_smooth) / sig_dm_smooth
pull_dh = (DH_mle_smooth - DH_lcdm_smooth) / sig_dh_smooth


def fisher_at_lcdm(h=5e-3):
    """Fisher matrix F = J^T C^{-1} J at (w0, wa) = (-1, 0) via finite differences."""
    def theory_vec(w0, wa):
        return np.array([predict_at(z, q, w0, wa) for z, q in zip(z_eff, quantities)])
    mu0 = theory_vec(-1.0, 0.0)
    dmu_dw0 = (theory_vec(-1.0 + h, 0.0) - theory_vec(-1.0 - h, 0.0)) / (2*h)
    dmu_dwa = (theory_vec(-1.0, h) - theory_vec(-1.0, -h)) / (2*h)
    Cinv = np.linalg.inv(COV)
    F = np.array([
        [dmu_dw0 @ Cinv @ dmu_dw0, dmu_dw0 @ Cinv @ dmu_dwa],
        [dmu_dw0 @ Cinv @ dmu_dwa, dmu_dwa @ Cinv @ dmu_dwa]
    ])
    return F


F = fisher_at_lcdm()
F_inv = np.linalg.inv(F)


def fisher_ellipse(F_mat, chi2_level, theta_0=(-1.0, 0.0), n_points=200):
    """(n_points, 2) array of (w0, wa) points on the Fisher confidence ellipse."""
    eigvals, eigvecs = np.linalg.eigh(F_mat)
    half_axes = np.sqrt(chi2_level / eigvals)
    phis = np.linspace(0, 2 * np.pi, n_points)
    pts_eigen = np.stack([half_axes[0] * np.cos(phis),
                          half_axes[1] * np.sin(phis)], axis=1)
    pts = pts_eigen @ eigvecs.T
    pts[:, 0] += theta_0[0]; pts[:, 1] += theta_0[1]
    return pts


e1 = fisher_ellipse(F, 2.30)
e2 = fisher_ellipse(F, 6.17)
e3 = fisher_ellipse(F, 11.83)


# All text in this figure scales together with FONT_SCALE.
FONT_SCALE = 1.1


def fs(size):
    return size * FONT_SCALE


plt.rcParams.update({
    'font.size': fs(18),
    'axes.labelsize': fs(17),
    'axes.titlesize': fs(17),
    'legend.fontsize': fs(12),
    'xtick.labelsize': fs(17),
    'ytick.labelsize': fs(17),
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# === LEFT PANEL ===
ax1.axhline(0, color='black', lw=0.6)
ax1.axhspan(-1, 1, color='gray', alpha=0.15, label='_nolegend_')
ax1.axhspan(-2, 2, color='gray', alpha=0.07, label='_nolegend_')

ax1.plot(z_smooth, pull_dm, color='crimson', linestyle='--', linewidth=1.8,
         label=r'$D_M/r_d$ pull at DR2 MLE')
ax1.plot(z_smooth, pull_dh, color='steelblue', linestyle=':', linewidth=1.8,
         label=r'$D_H/r_d$ pull at DR2 MLE')

# Markers share their pull-curve colour: D_M crimson, D_H steelblue, D_V grey.
plotted = set()
for i, (z, q) in enumerate(zip(z_eff, quantities)):
    if 'DV' in q:
        marker, color, lab = 'o', 'dimgray', r'$D_V/r_d$ (BGS)'
    elif 'DM' in q:
        marker, color, lab = 's', 'crimson', r'$D_M/r_d$'
    elif 'DH' in q:
        marker, color, lab = '^', 'steelblue', r'$D_H/r_d$'
    lbl = lab if lab not in plotted else None
    plotted.add(lab)
    ax1.errorbar(z, residuals[i], yerr=1, fmt=marker, color=color,
                 markersize=8, capsize=3, lw=1.2, label=lbl, zorder=5)

# Label positions chosen so that no label sits on an error bar.
for tag, z, ytext, ha in [
    ('BGS',      0.295, 0.65, 'center'),
    ('LRG1',     0.510, 1.95, 'center'),
    ('LRG2',     0.706, -3.50, 'center'),
    ('LRG3+ELG1', 0.970, -2.90, 'left'),
    ('ELG2',     1.321, 1.75, 'center'),
    ('QSO',      1.600, 0.75, 'left'),
    (r'Ly$\alpha$', 2.330, 1.25, 'center'),
]:
    ax1.annotate(tag, xy=(z, 0), xytext=(z, ytext),
                 fontsize=fs(13), ha=ha, color='black',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='none', alpha=0.7))

ax1.set_xlim(0, 2.6)
# The upper limit leaves room for the legend above the LRG1 label; the lower one for the
# LRG2 label below its D_H error bar.
ax1.set_ylim(-4.0, 5.8)
ax1.set_xlabel(r'Effective redshift $z_{\rm eff}$')
ax1.set_ylabel(r'(data $-$ $\Lambda$CDM) / $\sigma_{\rm data}$')
ax1.set_title(r'(a) DR2 BAO residuals from $\Lambda$CDM')
ax1.legend(loc='upper right', fontsize=fs(14), framealpha=0.9, ncol=2)


# === RIGHT PANEL ===
default_box = patches.Rectangle((-1.5, -2.0), 1.0, 3.0,
                                 linewidth=1, edgecolor='gray',
                                 facecolor='none', linestyle=':',
                                 label='Default box (flat prior)')
ax2.add_patch(default_box)

l1, = ax2.plot(e1[:, 0], e1[:, 1], color='black', lw=1.2)
l2, = ax2.plot(e2[:, 0], e2[:, 1], color='black', lw=1.0, linestyle='--', alpha=0.7)
l3, = ax2.plot(e3[:, 0], e3[:, 1], color='black', lw=0.8, linestyle=':', alpha=0.4)

# Caldwell-Linder thawing band: w_a in [-3(1+w_0), -(1+w_0)],
# drawn on its analysis support w_0 in [-1, -0.85] (Section 4.3):
# edges at w_0 = -0.85 are w_a = -0.45 and -0.15.
w0_thaw = np.linspace(-1.0, -0.85, 100)
wa_thaw_lo = -3.0 * (1 + w0_thaw)
wa_thaw_hi = -1.0 * (1 + w0_thaw)
ax2.fill_between(w0_thaw, wa_thaw_lo, wa_thaw_hi,
                 color='tab:green', alpha=0.30,
                 label='Caldwell–Linder thawing band')
ax2.plot(w0_thaw, wa_thaw_lo, color='tab:green', lw=1.0, alpha=0.7)
ax2.plot(w0_thaw, wa_thaw_hi, color='tab:green', lw=1.0, alpha=0.7)

freezing_box = patches.Rectangle((-0.95, 0.0), 0.20, 0.30,
                                  linewidth=1.5, edgecolor='tab:purple',
                                  facecolor='tab:purple', alpha=0.25,
                                  label='Freezing quintessence')
ax2.add_patch(freezing_box)

ax2.plot(-1, 0, marker='+', color='black', markersize=16, mew=2.5,
         label=r'$\Lambda$CDM $(-1, 0)$', linestyle='None')
ax2.plot(W0_MLE, WA_MLE, marker='X', color='crimson', markersize=14,
         markeredgecolor='black', markeredgewidth=1,
         label=f'DR2 BAO MLE $({W0_MLE:.3f}, {WA_MLE:.3f})$',
         linestyle='None')

ax2.set_xlim(-1.6, -0.4)
ax2.set_ylim(-2.2, 1.2)
ax2.set_xlabel(r'$w_0$')
ax2.set_ylabel(r'$w_a$')
ax2.set_title(r'(b) $(w_0, w_a)$ plane: priors and DESI DR2 Fisher constraint')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# The legend of panel (b) sits in a strip below the figure, because the Default box fills
# most of the panel and a legend inside the axes would cover the box or the 3 sigma ellipse.
# Three columns keep the strip to two rows. It is added after tight_layout so that the axes
# keep their size; the strip is included in the saved bounding box. The three Fisher
# ellipses share one entry (solid, dashed, dotted side by side) so the legend stays short.
from matplotlib.legend_handler import HandlerTuple
h, l = ax2.get_legend_handles_labels()
h.insert(1, (l1, l2, l3)); l.insert(1, r'DESI DR2 Fisher $1, 2, 3\sigma$')
fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=3, fontsize=fs(14),
           framealpha=0.95, handlelength=3.6, columnspacing=1.5,
           handler_map={tuple: HandlerTuple(ndivide=3, pad=0.15)})

out_pdf = FIGURES / "figure1_bao_data.pdf"
plt.savefig(out_pdf, bbox_inches='tight', dpi=200)
print(f"Wrote {out_pdf}")
