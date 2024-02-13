import sys
import os
import xarray as xr
import pandas as pd
import numpy as np

def load_data_and_save():
    """script to load simulation data and get information relevant for fitting
    """
    from pyathena.tigress_ncr.ncr_paper_lowz import LowZData

    pdata = LowZData()

    alldata = []
    allpoints = []
    for m in pdata.mlist:
        print(m)
        trange = None

        # get percentiles
        try:
            q_low = pdata.collect_zpdata(
                m, trange=trange, func=np.nanpercentile, q=16, silent=True
            )
        except KeyError:
            q_low = pdata.collect_zpdata(
                m, trange=trange, recal=True, func=np.nanpercentile, q=16
            )
        q_mid = pdata.collect_zpdata(
            m, trange=trange, func=np.nanpercentile, q=50, silent=True
        )
        q_high = pdata.collect_zpdata(
            m, trange=trange, func=np.nanpercentile, q=84, silent=True
        )
        q_mean = pdata.collect_zpdata(m, trange=trange, func=np.nanmean, silent=True)
        q_std = pdata.collect_zpdata(m, trange=trange, func=np.nanstd, silent=True)
        alldata.append(
            xr.concat([q_low, q_mid, q_high, q_mean, q_std], dim="q").assign_coords(
                q=[16, 50, 84, "mean", "std"]
            )
        )

        # get all points
        data = pdata.collect_zpdata(
            m, trange=trange, reduce=False, recal=False, silent=True
        )
        allpoints.append(data.drop("name").to_dataset("variable").to_dataframe())

    # all points
    pt_df = pd.concat(allpoints, ignore_index=True)
    outfile = "points_all.p"
    if os.path.isfile(outfile):
        os.remove(outfile)
    pt_df.to_pickle(outfile)

    # percentile
    qdset = xr.concat(alldata, dim="name")
    outfile = "percentiles_all.nc"
    if os.path.isfile(outfile):
        os.remove(outfile)
    qdset.to_netcdf(outfile)

if __name__ == "__main__":
    narg = len(sys.argv)
    if narg == 1:
        load_data_and_save()
    else:
        print(sys.argv[1])
