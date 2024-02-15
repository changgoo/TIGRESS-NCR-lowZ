import os
import xarray as xr
import numpy as np
import shutil
def copy_files(m):
    basedir = '/scratch/gpfs/changgoo/TIGRESS-NCR'
    savdir = os.path.join(basedir,m)

    indir = os.path.join(savdir,'hst2')
    outdir = './lowZ-coolheat-data'
    os.makedirs(outdir,exist_ok=True)
    for f in ['PEheating.nc','phase_vmeans.nc','phase_nmeans.nc']:
        infile = os.path.join(indir,f)
        outfile = os.path.join(outdir,f'{m}_{f}')
        shutil.copy2(infile,outfile)
    return

# "qdset" is a xarray.Dataset storing the (16, 50, 84) percentile values and mean and std for each model's time series into an effectively 3D array.
with xr.open_dataarray("percentiles_all.nc") as qdset:
    # get median values and convert it into pandas.DataFrame
    mid_df = qdset.sel(q="50").to_dataset(dim="variable").drop("q").to_dataframe()
    mid_logdf = np.log10(mid_df)

for m in qdset['name'].data:
    # da,vavg,navg = retrieve_timeseries(m)
    copy_files(m)
