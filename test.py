# Begin AI-written code (OpenAI Codex).
"""Plot the UniverseMachine fit and its slope at fixed redshift.

Run with ``python test.py`` to save the figures in the working directory.
M_halo denotes the peak halo mass passed to the symlib fit.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import palette
from palette import pc
import symlib

from invert import dlny_dlnx, highest_mvir, invert_func, invert_smhm
from invert import vir_from_mstar


ZS = [0, 0.5, 1, 2, 3, 4]
COLORS = ["r", "o", "g", "b", "p", "n"]


def log_slope(um_fit, m_halo, z, step=1e-3):
    """Return d ln(M_star) / d ln(M_halo) at fixed z; step is in dex."""
    log_m_halo = np.log10(m_halo)
    log_ms_hi = np.log10(
        um_fit.m_star(10**(log_m_halo + step), z, scatter=0))
    log_ms_lo = np.log10(
        um_fit.m_star(10**(log_m_halo - step), z, scatter=0))
    # The log base cancels in the ratio of logarithmic derivatives.
    return (log_ms_hi - log_ms_lo) / (2*step)


def halo_axis(ax):
    ax.set_xscale("log")
    ax.set_xlim(1e10, 1e15)
    ax.set_xlabel(r"$M_{\rm halo}\ (M_\odot)$")
    ax.legend()


def plot_mstar(um_fit):
    m_halo = np.logspace(10, 15, 1000)
    fig, ax = plt.subplots()
    for z, color in zip(ZS, COLORS):
        m_star = um_fit.m_star(m_halo, z, scatter=0)
        ax.plot(m_halo, m_star, c=pc(color), label=f"$z={z:g}$")

    ax.set_yscale("log")
    ax.set_ylabel(r"$M_\star\ (M_\odot)$")
    ax.set_ylim(1e7, 1e12)
    halo_axis(ax)
    fig.savefig("um_mstar.png")
    plt.close(fig)


def plot_slope(um_fit):
    m_halo = np.logspace(10, 15, 1000)
    fig, ax = plt.subplots()
    for z, color in zip(ZS, COLORS):
        slope = log_slope(um_fit, m_halo, z)
        ax.plot(m_halo, slope, c=pc(color), label=f"$z={z:g}$")

    ax.set_ylabel(r"$\mathrm{d}\ln M_\star/\mathrm{d}\ln M_{\rm halo}$")
    ax.axhline(0, c=pc("a"), ls="--", lw=1)
    halo_axis(ax)
    fig.savefig("um_log_slope.png")
    plt.close(fig)


def plot_highest_mvir(um_fit):
    zs = np.linspace(0, 5, 200)
    slope_limits = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    colors = [pc(c) for c in ["r", "o", "g", "b", "p", "n"]]
    fig, ax = plt.subplots()
    for limit, color in zip(slope_limits, colors):
        masses = np.empty(len(zs))
        for i, z in enumerate(zs):
            mstar_func = lambda mvir: um_fit.m_star(mvir, z, scatter=0)
            masses[i] = highest_mvir(mstar_func, slope_limit=limit)
        # Infinite results mean no upper mass limit; leave gaps in the curve.
        masses[~np.isfinite(masses)] = np.nan
        ax.plot(zs, masses, c=color, label=f"${limit:g}$")

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$M_{\rm vir,max}\ (M_\odot)$")
    ax.set_yscale("log")
    ax.legend(title="slope_limit")
    fig.savefig("um_highest_mvir.png")
    plt.close(fig)


def inversion_curves(um_fit, slope_limit):
    curves = []
    max_error = 0
    for z in ZS:
        mstar_func = lambda mvir: um_fit.m_star(mvir, z, scatter=0)
        upper = min(highest_mvir(mstar_func, slope_limit), 1e15)
        m_halo = np.logspace(10, np.log10(upper), 1000)
        m_star = mstar_func(m_halo)
        inverse = np.array([invert_func(mstar_func, ms) for ms in m_star])
        max_error = max(max_error, np.max(np.abs(inverse/m_halo - 1)))
        beta = dlny_dlnx(mstar_func, inverse)
        inv_mu, _ = invert_smhm(inverse, -1.92, beta, um_fit.scatter)
        curves.append((m_star, m_halo, inverse, inv_mu))

    print(f"Maximum invert_func relative mass error: {max_error:.3g}")
    return curves


def inversion_panel(ax, curves, method):
    index = {"invert_func": 2, "invert_smhm": 3}[method]
    for z, color, curve in zip(ZS, COLORS, curves):
        m_star, m_halo = curve[:2]
        ax.plot(m_star, m_halo, c=pc(color), label=f"$z={z:g}$")
        ax.plot(m_star, curve[index], c=pc(color, 0.65), ls="--", lw=2)

    ax.set_box_aspect(1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$M_{\rm halo}\ (M_\odot)$")
    redshift_legend = ax.legend(loc="upper left", ncol=2)
    ax.add_artist(redshift_legend)
    handles = [Line2D([], [], c=pc("k"), label="Direct SMHM"),
               Line2D([], [], c=pc("k", 0.65), ls="--", lw=2,
                      label=method)]
    ax.legend(handles=handles, loc="lower right")


def ratio_panel(ax, curves, method):
    index = {"invert_func": 2, "invert_smhm": 3}[method]
    for color, curve in zip(COLORS, curves):
        m_star, m_halo = curve[:2]
        ax.plot(m_star, curve[index]/m_halo,
                c=pc(color, 0.65), ls="--", lw=2)

    ax.set_xlabel(r"$M_\star\ (M_\odot)$")
    ax.set_ylabel("Ratio to\ndirect SMHM", fontsize=18)
    ax.axhline(1, c=pc("a"), lw=1, zorder=0)
    ax.set_ylim(0, 1.08)


def inversion_figure(um_fit, slope_limit, method):
    curves = inversion_curves(um_fit, slope_limit)
    fig, (ax, ax_ratio) = plt.subplots(
        2, 1, sharex=True, figsize=(7, 10), layout="constrained",
        gridspec_kw={"height_ratios": [1, 0.3], "hspace": 0.03})
    inversion_panel(ax, curves, method)
    ratio_panel(ax_ratio, curves, method)
    ax.set_title(rf"$\sigma_\star={um_fit.scatter:g}$ dex, "
                 rf"$\beta_{{\rm min}}={slope_limit:g}$", fontsize=18)
    return fig


def plot_invert_func(um_fit):
    fig = inversion_figure(um_fit, 0.1, "invert_func")
    fig.savefig("um_invert_func.png")
    plt.close(fig)


def plot_invert_smhm(um_fit, slope_limit=0.1):
    fig = inversion_figure(um_fit, slope_limit, "invert_smhm")
    fig.savefig(f"um_invert_smhm_beta_{slope_limit:g}.png")
    if slope_limit == 0.1:
        fig.savefig("um_invert_smhm.png")
    plt.close(fig)


def cutoff_values(um_fit, z):
    slope_limits = np.linspace(0, 0.5, 501)
    mstar_func = lambda mvir: um_fit.m_star(mvir, z, scatter=0)
    masses = np.array([highest_mvir(mstar_func, limit)
                       for limit in slope_limits])
    finite = np.isfinite(masses)
    ratios = np.full(len(masses), np.nan)
    beta = dlny_dlnx(mstar_func, masses[finite])
    if not np.allclose(beta, slope_limits[finite], atol=1e-5, rtol=0):
        raise RuntimeError("highest_mvir did not reach the requested slope")
    corrected, _ = invert_smhm(masses[finite], -1.92, beta, um_fit.scatter)
    ratios[finite] = corrected/masses[finite]

    # There is no finite evaluation mass when highest_mvir returns infinity.
    masses[~finite] = np.nan
    return slope_limits, masses, ratios


def cutoff_axis(ax, um_fit, z):
    ax.set_xlabel(r"Slope cutoff $\beta_{\rm min}$")
    ax.set_xlim(0, 0.5)
    ax.set_title(rf"$z={z:g}$, $\sigma_\star={um_fit.scatter:g}$ dex")


def plot_cutoff_ratio(um_fit, z=2):
    limits, _, ratios = cutoff_values(um_fit, z)
    fig, ax = plt.subplots()
    ax.plot(limits, ratios, c=pc("b"))
    ax.set_ylabel("Ratio to direct SMHM")
    ax.set_ylim(0, 1)
    cutoff_axis(ax, um_fit, z)
    fig.savefig("um_cutoff_ratio_z2.png")
    plt.close(fig)


def plot_cutoff_mass(um_fit, z=2):
    limits, masses, _ = cutoff_values(um_fit, z)
    fig, ax = plt.subplots()
    ax.plot(limits, masses, c=pc("b"))
    ax.set_ylabel(r"$M_{\rm vir,max}\ (M_\odot)$")
    ax.set_yscale("log")
    cutoff_axis(ax, um_fit, z)
    fig.savefig("um_cutoff_mass_z2.png")
    plt.close(fig)


def vir_figure(target, ylabel):
    log_mstar = np.linspace(8, 12, 250)
    fig, ax = plt.subplots()
    for z, color in zip(ZS, COLORS):
        values, out_of_range = vir_from_mstar(log_mstar, z, target)
        ax.plot(10**log_mstar, values, c=pc(color), label=f"$z={z:g}$")
        ax.scatter(10**log_mstar[out_of_range][::10],
                   values[out_of_range][::10], s=35,
                   facecolors="none", edgecolors=pc(color), zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_\star\ (M_\odot)$")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title("vir_from_mstar", fontsize=18)
    ax.set_xlim(1e8, 1e12)
    return fig


def plot_mvir():
    fig = vir_figure("mvir", r"$M_{\rm vir}\ (M_\odot)$")
    fig.savefig("um_vir_from_mstar_mvir.png")
    plt.close(fig)


def plot_rvir():
    fig = vir_figure("rvir", r"$R_{\rm vir}\ ({\rm kpc})$")
    fig.savefig("um_vir_from_mstar_rvir.png")
    plt.close(fig)


def main():
    palette.configure(False)
    um_fit = symlib.UniverseMachineMStarFit(scatter=0)
    scatter_fit = symlib.UniverseMachineMStarFit(scatter=0.2)

    """
    plot_mstar(um_fit)
    plot_slope(um_fit)
    plot_highest_mvir(um_fit)
    plot_invert_func(scatter_fit)
    plot_invert_smhm(scatter_fit, slope_limit=0.1)
    plot_invert_smhm(scatter_fit, slope_limit=0.2)
    plot_invert_smhm(scatter_fit, slope_limit=0.3)
    plot_cutoff_ratio(scatter_fit)
    plot_cutoff_mass(scatter_fit)
    """
    
    plot_mvir()
    plot_rvir()


if __name__ == "__main__":
    main()
# End AI-written code (OpenAI Codex).
