import os
from pyathena.tigress_ncr.ncr_paper_lowz import LowZData
import xarray as xr


def get_fraction_time_series(zp_ph, zpw):
    zarr = zpw.z.where(zpw.z > 0, drop=True).data
    fv = []
    fm = []
    fvq = []
    fmq = []
    for zmax in zarr:
        tavg = (
            zp_ph.sel(z=slice(-zmax, zmax)).sum(dim="z")
            / zpw.sel(z=slice(-zmax, zmax)).sum(dim="z")
        ).mean(dim="time")
        tq = (
            zp_ph.sel(z=slice(-zmax, zmax)).sum(dim="z")
            / zpw.sel(z=slice(-zmax, zmax)).sum(dim="z")
        ).quantile(q=[0.025, 0.16, 0.25, 0.5, 0.75, 0.84, 0.975], dim="time")

        fv.append(tavg["A"].data)
        fm.append(tavg["d"].data)
        fvq.append(tq["A"].data)
        fmq.append(tq["d"].data)
    dset = xr.Dataset()
    dset["fvq"] = xr.DataArray(fvq, coords=[zarr, tq["quantile"].data], dims=["z", "q"])
    dset["fmq"] = xr.DataArray(fmq, coords=[zarr, tq["quantile"].data], dims=["z", "q"])
    dset["fv"] = xr.DataArray(fv, coords=[zarr], dims=["z"])
    dset["fm"] = xr.DataArray(fm, coords=[zarr], dims=["z"])

    return dset


if __name__ == "__main__":
    pdata = LowZData()

    for m in pdata.mlist:
        s = pdata.sa.set_model(m)
        zp = s.read_zprof_new(flist=["A", "d"])

    for m in pdata.mlist:
        print(m)
        s = pdata.sa.set_model(m)
        zp = s.zp
        shorthands = s.get_phase_shorthand()
        rename_dict = dict()
        for i, pname in enumerate(shorthands):
            rename_dict["phase{}".format(i + 1)] = pname
        zp = zp.to_array().to_dataset("phase").rename(rename_dict)
        torb = s.torb_Myr
        if s.torb_Myr < 50:
            trange = slice(torb * 5, torb * 15)
        elif s.torb_Myr > 300:
            trange = slice(torb * 1.5, torb * 5)
        else:
            trange = slice(torb * 2, torb * 5)
        zp = zp.to_array("phase").to_dataset("variable").sel(time=trange)
        zpw = zp.sel(phase=["c", "u", "w1", "w2", "h1", "h2"]).sum(dim="phase")
        zp_CMM = zp.sel(phase=["cmm"]).sum(dim="phase")
        zp_CNM = zp.sel(phase=["cnm"]).sum(dim="phase")
        zp_UNM = zp.sel(phase=["unm"]).sum(dim="phase")
        zp_WNM = zp.sel(phase=["wnm"]).sum(dim="phase")
        zp_WIM = zp.sel(phase=["uim", "wpim", "wcim"]).sum(dim="phase")
        zp_HIM = zp.sel(phase=["h1", "h2"]).sum(dim="phase")
        dlist = []
        for zp_ph in [zp_CMM, zp_CNM, zp_UNM, zp_WNM, zp_WIM, zp_HIM]:
            dset = get_fraction_time_series(zp_ph, zpw)
            dlist.append(dset)
        phlist = ["CMM", "CNM", "UNM", "WNM", "WIM", "HIM"]
        zp_ph = xr.concat(
            [d.assign_coords(phase=[ph]) for d, ph in zip(dlist, phlist)], dim="phase"
        )
        s.zp_ph = zp_ph

    outdir = "./data/lowZ-filling-data"
    os.makedirs(outdir, exist_ok=True)
    for m in pdata.mlist:
        print(m)
        s = pdata.sa.set_model(m)
        s.zp_ph.to_netcdf(os.path.join(outdir, f"{m}.filling.nc"))
