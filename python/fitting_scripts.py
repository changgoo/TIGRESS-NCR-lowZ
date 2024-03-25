from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# import pandas as pd
# import sys, os
import cmasher as cmr


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
    ec = plt.rcParams["axes.labelcolor"]
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
    qlow = dset.sel(q="16").to_dataset(dim="variable").to_dataframe()
    qmid = dset.sel(q="50").to_dataset(dim="variable").to_dataframe()
    qhigh = dset.sel(q="84").to_dataset(dim="variable").to_dataframe()
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


def plot_fitting_mean(
    dset,
    model_dset,
    xf,
    yf,
    cmap=cmr.guppy,
    residual=False,
    model=True,
    legend=4,
    colorbar=0,
    nostd=True,
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
    xf2, xf1 = model_dset.dims
    (cf,) = set(model_dset.dims) - {xf}

    # set up color bar properties from model
    cmin, cmax = model_dset[cf].min().data, model_dset[cf].max().data
    norm = Normalize(vmin=cmin, vmax=cmax)

    # get the median values of the simulation results
    # and assign color given coloarbar normalization set by the model range
    x1 = np.log10(dset.sel(variable=xf, q="mean"))
    x1std = (
        dset.sel(variable=xf, q="std") / dset.sel(variable=xf, q="mean") / np.log(10)
    )
    x2 = np.log10(dset.sel(variable=cf, q="mean"))
    x2std = (
        dset.sel(variable=cf, q="std") / dset.sel(variable=cf, q="mean") / np.log(10)
    )
    y = np.log10(dset.sel(variable=yf, q="mean"))
    ystd = dset.sel(variable=yf, q="std") / dset.sel(variable=yf, q="mean") / np.log(10)

    c = cmap(norm(x2.data))
    marker = []
    for name in list(dset["name"].data):
        if "Zd0.025" in name:
            marker.append("s")
        elif "Om01" in name:
            marker.append("v")
        elif "Om02" in name:
            marker.append("^")
        elif "b10" in name and "S05" not in name:
            marker.append("*")
        else:
            marker.append("o")

    if cf == "Zgas":
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(cmin, cmax))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # ec
    ec = plt.rcParams["axes.labelcolor"]
    # caclulate residual
    if model:
        Y2 = np.array(
            [model_dset.interp({xf1: x1v, xf2: x2v}).data for x1v, x2v in zip(x1, x2)]
        )
        res = 10**Y2 / 10**y - 1
    else:
        res = 10**y / 10**x1 - 1

    if residual:
        # residual plot
        print(f"L1 : {np.abs(res.data).mean()}")
        print(f"L2 : {np.sqrt((res.data**2).mean())}")
        for x_, y_, c_, m_ in zip(x1, res, c, marker):
            sca = plt.scatter(x_, y_, color=c_, marker=m_, s=5)
        if model:
            plt.ylabel("rel. err.")
        else:
            plt.ylabel("rel. diff.")
        plt.ylim(-1, 1)
    else:
        # data plot with model
        xerr = x1std
        yerr = ystd

        if nostd:
            xerr = np.zeros_like(x1std)
            yerr = np.zeros_like(ystd)

        for x_, y_, xerr_, yerr_, c_, m_ in zip(x1, y, xerr, yerr, c, marker):
            plt.errorbar(
                x_,
                y_,
                xerr=xerr_,
                yerr=yerr_,
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
                    alpha=0.7,
                )

        # add y label
        if yf in labels:
            plt.ylabel(r"$\log\,$" + labels[yf])

        # add legend
        if legend:
            pkwargs = dict(ls="", marker="o", markeredgecolor=ec)

            Zmodels = sorted(np.unique(x2))
            Zlabels = [f"{Z:.1f}" for Z in 10.0 ** np.array(Zmodels)]
            colors = cmap(norm(Zmodels))
            custom_lines = []
            for c in colors:
                custom_lines.append(Line2D([0], [0], color=c, **pkwargs))
            leg1 = plt.legend(
                custom_lines,
                Zlabels,
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
        plt.xlabel(r"$\log\,$" + labels[xf])


def scifmt(value, fmt=":9.2e"):
    """format the number into string with scientific format"""
    maxdigits = int(fmt.split(".")[1][0]) + 1
    sp = "{{{}}}".format(fmt).format(value).split("e")
    digits = int(sp[1])
    if abs(digits) < maxdigits:
        return "{{{}}}".format(":9.{}f".format(maxdigits - digits - 1)).format(value)
    else:
        return sp[0] + "\\cdot 10^{{{}}}".format(int(sp[1]))


def regress(df, x1="W", x2="Zgas", yf="Ytot", **fit_kwargs):
    """bi-variate power-law fitting using linear regression"""
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


def fit_odr(dset, xf1="W", xf2="Zgas", yf="Ytot", std=False, **fit_kwargs):
    """bi-variate power-law fitting using orthogonal distance regression"""
    from scipy import odr

    def _lin_fcn(B, x):
        a, b = B[0], B[1:]
        b.shape = (b.shape[0], 1)

        return a + (x * b).sum(axis=0)

    x1 = np.log10(dset.sel(variable=xf1, q="mean"))
    x1std = (
        dset.sel(variable=xf1, q="std") / dset.sel(variable=xf1, q="mean") / np.log(10)
    )
    x2 = np.log10(dset.sel(variable=xf2, q="mean"))
    x2std = (
        dset.sel(variable=xf2, q="std") / dset.sel(variable=xf2, q="mean") / np.log(10)
    )
    y = np.log10(dset.sel(variable=yf, q="mean"))
    ystd = dset.sel(variable=yf, q="std") / dset.sel(variable=yf, q="mean") / np.log(10)

    model = odr.Model(_lin_fcn)
    if std:
        data = odr.RealData([x1, x2], y, sx=[x1std, np.ones_like(x2std)], sy=ystd)
    else:
        data = odr.RealData([x1, x2], y)
    myodr = odr.ODR(data, model, beta0=[0, 1, 0])
    myoutput = myodr.run()
    myoutput.pprint()

    return myoutput


def get_model(
    regr, Wmin=2.5, Wmax=6.5, Zmin=-1.5, Zmax=0.5, nW=100, nZ=100, dims=["Zgas", "W"]
):
    """calculate a model using the fitting results"""
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


def get_model_odr(
    odr_out, Wmin=2.5, Wmax=6.5, Zmin=-1.5, Zmax=0.5, nW=100, nZ=100, dims=["Zgas", "W"]
):
    """calculate a model using the fitting results"""
    Z = np.linspace(Zmin, Zmax, nZ)
    if nZ == 3:
        Z = np.array([0.1, 0.3, 1])
    W = np.linspace(Wmin, Wmax, nW)

    model = (
        odr_out.beta[1] * (W[np.newaxis, :])
        + odr_out.beta[2] * (Z[:, np.newaxis])
        + odr_out.beta[0]
    )

    model_dset = xr.DataArray(model, coords=[Z, W], dims=dims)

    return model_dset


def set_labels():
    """set up labels to be used in plots"""
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
    labels["PDE_2p_avg_approx_sp"] = labels["PDE"]
    labels["Ptot_hot"] = r"$P_{\rm tot,hot}$" + Punit_label
    for k, v in Ulabels.items():
        newk = k.replace("Pi", "Y").replace("P", "Y")
        labels[newk] = v + v_unit_label
    labels["Ynonth"] = r"$\Upsilon_{\rm turb+mag}$" + v_unit_label
    labels["Zgas"] = r"$Z_{\rm g}^\prime$"
    labels["Zdust"] = r"$Z_{\rm d}^\prime$"

    labels["sfr"] = r"$\Sigma_{\rm SFR}$" + sfr_unit_label
    labels["sfr10"] = labels["sfr40"] = labels["sfr100"] = labels["sfr"]
    labels["tdep40"] = r"$t_{\rm dep}\,[{\rm Myr}]$"
    labels["tdep"] = r"$t_{\rm dep}\,[{\rm Myr}]$"
    labels["tdyn"] = r"$t_{\rm dyn}\,[{\rm Myr}]$"
    labels["edyn"] = r"$\epsilon_{\rm dyn}$"
    labels["nH"] = r"$n_{\rm H,mid}\,[{\rm cm^{-3}}]$"
    labels["rhotot"] = r"$\rho_{\rm tot}\,[{M_\odot\,{\rm pc^{-3}}}]$"
    labels["sigma_eff"] = r"$\sigma_{\rm z,eff}$" + v_unit_label
    labels["sigma_eff_mid"] = r"$\sigma_{\rm z,eff,mid}$" + v_unit_label
    labels["sigma_turb"] = r"$\sigma_{\rm z,turb}$" + v_unit_label
    labels["sigma_turb_mid"] = r"$\sigma_{\rm z,turb,mid}$" + v_unit_label
    labels["sigma_th"] = r"$\sigma_{\rm th}$" + v_unit_label
    labels["sigma_th_mid"] = r"$\sigma_{\rm th,mid}$" + v_unit_label
    labels["Sigma_gas"] = r"$\Sigma_{\rm gas}\,[M_\odot\,{\rm pc^{-2}}]$"
    labels["H"] = r"$H_{\rm gas}\,[{\rm pc}]$"

    return labels


# set labels as a global variable
labels = set_labels()
