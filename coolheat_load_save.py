import os
import xarray as xr
import numpy as np
import shutil
import gc

from pyathena.tigress_ncr.zprof import zprof_rename
from pyathena.tigress_ncr.ncr_paper_lowz import LowZData


def load_save_zprof():
    tmp_outdir = "./lowZ-zprof-data"
    os.makedirs(tmp_outdir, exist_ok=True)

    flist = ["A", "d", "Ek1", "Ek2", "Ek3", "dEk2", "cool", "heat", "net_cool", "P"]
    pdata = LowZData()
    for m in pdata.mlist:
        print(m)
        s = pdata.sa.set_model(m)
        f = os.path.join(tmp_outdir, f"{s.basename}_newzp.nc")

        s.zp = s.read_zprof_new(flist=flist)
        trange = pdata.get_trange(s)
        area = s.domain["Lx"][0] * s.domain["Lx"][1]
        s.newzp = zprof_rename(s).sel(time=trange) * area
        s.newzp.to_netcdf(f)
        delattr(s, "zp")
        gc.collect()


def copy_files(m):
    basedir = "/scratch/gpfs/changgoo/TIGRESS-NCR"
    savdir = os.path.join(basedir, m)

    indir = os.path.join(savdir, "hst2")
    outdir = "./lowZ-coolheat-data"
    os.makedirs(outdir, exist_ok=True)
    for f in ["PEheating.nc", "phase_vmeans.nc", "phase_nmeans.nc"]:
        infile = os.path.join(indir, f)
        outfile = os.path.join(outdir, f"{m}_{f}")
        shutil.copy2(infile, outfile)
    return


if __name__ == "__main__":
    # "qdset" is a xarray.Dataset storing the (16, 50, 84) percentile values and mean and std for each model's time series into an effectively 3D array.
    with xr.open_dataarray("percentiles_all.nc") as qdset:
        # get median values and convert it into pandas.DataFrame
        mid_df = qdset.sel(q="50").to_dataset(dim="variable").drop("q").to_dataframe()
        mid_logdf = np.log10(mid_df)

    for m in qdset["name"].data:
        # da,vavg,navg = retrieve_timeseries(m)
        copy_files(m)

    load_save_zprof()
