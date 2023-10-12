from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import sys, os

def add_panel_labels(axes, label0="a"):
    for i, ax in enumerate(axes):
        ax.annotate(
            f"({chr(ord(label0)+i)})",
            (0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
        )


def plot_fitting_points(
    dset,
    model_dset,
    xf,
    yf,
    cmap=plt.cm.viridis,
    residual=False,
    model=True,
    legend=4,
    colorbar=0,
):
    """Plotting script for the fitting results with all points

    Parameters
    ==========
    dset : xarray.Dataset
        data containg all points
    model_dset : xarry.Dataset
        model constructed by the get_model function
        using the fitting results from the regress function
    xf : str
        field name for the x-axis
    yf : str
        field name for the y-axis
    cmap : matplotlib.colors.ListedColormap
        color map for the third dimension (e.g. Zgas)
    residual : bool
        plot simulation points if False or residual if True
    model : sklearn.linear_model._base.LinearRegression or False
        calculate residual comparing xf and yf if False, otherwise compare with model
    legend : int
        add legend for the third dimension if not zero
        the number corresponds to legend location
    coloarbar : int
        add colorbar for the third dimension if not zero
        the number corresponds to legend location
    """

    # The model has two x-axes (x1, x2).
    # We will use one for the x-axis of the plot (xf) and the other for coloarbar (cf)
    x2, x1 = model_dset.dims
    (cf,) = set(model_dset.dims) - {xf}

    # set up color bar properties from model
    cmin, cmax = model_dset[cf].min().data, model_dset[cf].max().data
    norm = LogNorm(vmin=cmin, vmax=cmax)

    # get the median values of the simulation results
    # and assign color given coloarbar normalization set by the model range
    df = dset.sel(q="50").to_dataset(dim="variable").to_dataframe()
    c = cmap(norm(df[cf]))
    marker = ["s" if Zd == 0.025 else "o" for Zd in df["Zdust"]]

    if cf == "Zgas":
        sm = plt.cm.ScalarMappable(
            cmap=cmap, norm=Normalize(np.log10(cmin), np.log10(cmax))
        )
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ec
    ec = plt.rcParams['axes.labelcolor']
    # caclulate residual
    if model:
        Y2 = np.array(
            [
                model_dset.interp({x1: x1v, x2: x2v}).data
                for x1v, x2v in zip(df[x1], df[x2])
            ]
        )
        res = Y2 / df[yf] - 1
    else:
        res = df[yf] / df[xf] - 1

    # use percentiles for errorbar
    qlow = dset.sel(q='16').to_dataset(dim="variable").to_dataframe()
    qmid = dset.sel(q='50').to_dataset(dim="variable").to_dataframe()
    qhigh = dset.sel(q='84').to_dataset(dim="variable").to_dataframe()
    qx = np.array([qlow[xf], qmid[xf], qhigh[xf]])
    qy = np.array([qlow[yf], qmid[yf], qhigh[yf]])

    if residual:
        # residual plot
        print(f"L1 : {np.abs(res).mean()}")
        print(f"L2 : {np.sqrt((res**2).mean())}")
        for x_, y_, c_, m_ in zip(df[xf], res, c, marker):
            sca = plt.scatter(x_, y_, color=c_, marker=m_, s=5)
        if model:
            plt.ylabel("rel. err.")
        else:
            plt.ylabel("rel. diff.")
        plt.ylim(-1, 1)
    else:
        # data plot with model
        xerr = np.array([qx[1] - qx[0], qx[2] - qx[1]]).T
        yerr = np.array([qy[1] - qy[0], qy[2] - qy[1]]).T

        for x_, y_, xerr_, yerr_, c_, m_ in zip(qx[1], qy[1], xerr, yerr, c, marker):
            plt.errorbar(
                x_,
                y_,
                xerr=xerr_.reshape(2, 1),
                yerr=yerr_.reshape(2, 1),
                c=c_,
                marker=m_,
                markersize=5,
                elinewidth=1,
                ecolor=ec,
                markeredgecolor=ec,
                ls="",
                zorder=10,
            )

        # plot model lines
        if model:
            for cval in model_dset[cf]:
                plt.plot(
                    model_dset[xf],
                    model_dset.interp({cf: cval}),
                    color=cmap(norm(cval)),
                    lw=1,
                    # alpha=0.5,
                )

        # add y label
        if yf in labels:
            plt.ylabel(labels[yf])
        plt.yscale("log")

        # add legend
        if legend:
            pkwargs = dict(ls="", marker="o", markeredgecolor=ec)

            from matplotlib.lines import Line2D

            Zmodels = sorted(np.unique(df[cf]))
            colors = cmap(norm(Zmodels))
            custom_lines = []
            for c in colors:
                custom_lines.append(Line2D([0], [0], color=c, **pkwargs))
            leg1 = plt.legend(
                custom_lines,
                Zmodels,
                title=labels[cf],
                fontsize="xx-small",
                title_fontsize="x-small",
                loc=legend,
            )

        # add colorbar
        if colorbar:
            ax = plt.gca()
            axins = inset_axes(ax, loc=colorbar, borderpad=1, width="30%", height="5%")

            if cf == "Zgas":
                cbar = plt.colorbar(
                    sm,
                    cax=axins,
                    orientation="horizontal",
                    label=r"$\log$" + labels[cf],
                )
                cbar.set_ticks([-1, 0])
            else:
                cbar = plt.colorbar(
                    sm, cax=axins, orientation="horizontal", label=labels[cf]
                )
            plt.sca(ax)
    # add x label
    if xf in labels:
        plt.xlabel(labels[xf])
    plt.xscale("log")


def scifmt(value, fmt=":9.2e"):
    """format the number into string with scientific format
    """
    maxdigits = int(fmt.split(".")[1][0]) + 1
    sp = "{{{}}}".format(fmt).format(value).split("e")
    digits = int(sp[1])
    if abs(digits) < maxdigits:
        return "{{{}}}".format(":9.{}f".format(maxdigits - digits - 1)).format(value)
    else:
        return sp[0] + "\\cdot 10^{{{}}}".format(int(sp[1]))

def regress(df, x1="W", x2="Zgas", yf="Ytot", **fit_kwargs):
    """bi-variate power-law fitting using linear regression
    """
    from sklearn import linear_model

    # import statsmodels.api as sm
    df_ = df[[x1, x2, yf]].dropna()
    x = df_[[x1, x2]]
    y = df_[yf]

    regr = linear_model.LinearRegression(**fit_kwargs)
    regr.fit(x, y)

    # print("Intercept: \n", regr.intercept_)
    # print("Coefficients: \n", regr.coef_)
    return regr


def get_model(
    regr, Wmin=2.5, Wmax=6.5, Zmin=-1.5, Zmax=0.5, nW=100, nZ=100, dims=["Zgas", "W"]
):
    """calculate a model using the fitting results
    """
    Z = np.logspace(Zmin, Zmax, nZ)
    if nZ == 3:
        Z = np.array([0.1, 0.3, 1])
    W = np.logspace(Wmin, Wmax, nW)

    model = (
        regr.coef_[0] * np.log10(W[np.newaxis, :])
        + regr.coef_[1] * np.log10(Z[:, np.newaxis])
        + regr.intercept_
    )

    model_dset = xr.DataArray(10.0**model, coords=[Z, W], dims=dims)

    return model_dset


def set_labels():
    """set up labels to be used in plots
    """
    Punit_label = r"$/k_B\,[{\rm cm^{-3}\,K}]$"
    sfr_unit_label = r"$\,[M_\odot{\rm \,kpc^{-2}\,yr}]$"
    v_unit_label = r"$\,[{\rm km/s}]$"

    Plabels = {
        "Ptot": r"${P}_{\rm tot}$",
        "Pth": r"${P}_{\rm th}$",
        "Pturb": r"${P}_{\rm turb}$",
        "Pimag": r"${\Pi}_{\rm mag}$",
        "dPimag": r"${\Pi}_{\delta B}$",
        "oPimag": r"${\Pi}_{\overline{B}}$",
        "W": r"$\mathcal{W}$",
        "Wext": r"$\mathcal{W}_{\rm ext}$",
        "Wsg": r"$\mathcal{W}_{\rm sg}$",
        "PDE": r"$P_{\rm DE}$",
        "Prad": r"$\Delta P_{\rm rad}$",
    }
    Ulabels = {
        "Ptot": r"$\Upsilon_{\rm tot}$",
        "Pth": r"$\Upsilon_{\rm th}$",
        "Pturb": r"$\Upsilon_{\rm turb}$",
        "Pimag": r"$\Upsilon_{\rm mag}$",
        "dPimag": r"$\Upsilon_{\delta B}$",
        "oPimag": r"$\Upsilon_{\overline{B}}$",
    }
    labels = dict()
    for k, v in Plabels.items():
        labels[k] = v + Punit_label
    labels["PDE_2p_avg_approx"] = labels["PDE"]
    labels["Ptot_hot"] = r"$P_{\rm tot,hot}$" + Punit_label
    for k, v in Ulabels.items():
        newk = k.replace("Pi", "Y").replace("P", "Y")
        labels[newk] = v + v_unit_label
    labels["Ynonth"] = r"$\Upsilon_{\rm turb+mag}$" + v_unit_label
    labels["Zgas"] = r"$Z_{\rm gas}^\prime$"
    labels["Zdust"] = r"$Z_{\rm dust}^\prime$"

    labels["sfr"] = r"$\Sigma_{\rm SFR}$" + sfr_unit_label
    labels["sfr10"] = labels["sfr40"] = labels["sfr100"] = labels["sfr"]
    labels["tdep40"] = r"$t_{\rm dep}\,[{\rm Myr}]$"
    labels["tdep"] = r"$t_{\rm dep}\,[{\rm Myr}]$"
    labels["nH"] = r"$n_{\rm H}\,[{\rm cm^{-3}}]$"
    labels["rhotot"] = r"$\rho_{\rm tot}\,[{M_\odot\,{\rm pc^{-3}}}]$"
    labels["sigma_eff"] = r"$\sigma_{\rm eff}$" + v_unit_label
    labels["sigma_eff_mid"] = r"$\sigma_{\rm eff,mid}$" + v_unit_label
    labels["sigma_turb"] = r"$\sigma_{\rm turb}$" + v_unit_label
    labels["sigma_turb_mid"] = r"$\sigma_{\rm turb,mid}$" + v_unit_label
    labels["sigma_th"] = r"$\sigma_{\rm th}$" + v_unit_label
    labels["sigma_th_mid"] = r"$\sigma_{\rm th,mid}$" + v_unit_label
    labels["Sigma_gas"] = r"$\Sigma_{\rm gas}\,[M_\odot\,{\rm pc^{-2}}]$"
    labels["H"] = r"$H_{\rm gas}\,[{\rm pc}]$"

    return labels

# set labels as a global variable
labels = set_labels()