import xarray as xr
import numpy as np
from mpi4py import MPI
import os

# import pyathena as pa
# from pyathena.tigress_ncr.ncr_paper_lowz import get_PW_zprof
from pyathena.classic.utils import gradient as gradient_classic
from pyathena.tigress_ncr.ncr_paper_lowz import LowZData

# from pyathena.util.derivative import gradient,deriv_direct,deriv_central

# import matplotlib.pyplot as plt
# import cmasher as cmr
# from labellines import labelLines


def get_stress(s, data):
    # set parameters
    qshear = s.par["problem"]["qshear"]
    Omega = s.par["problem"]["Omega"]
    Lx, Ly, Lz = s.domain["Lx"]
    dx, dy, dz = s.domain["dx"]

    stress = xr.Dataset()
    # Reynold stress
    vy0 = -qshear * Omega * data.x
    dvy = data["velocity2"] - vy0

    stress["Rxy"] = qshear * Omega * data["velocity1"] * data["density"] * dvy

    # Maxwell stress
    stress["Mxy"] = -(
        qshear * Omega * data["cell_centered_B1"] * data["cell_centered_B2"]
    )

    # Gravitational stress
    gx1, gy1, gz1 = gradient_classic(
        data["gravitational_potential"].data, s.domain["dx"]
    )
    # gx2,gy2,gz2 = gradient(data['gravitational_potential'].data,
    #                     data.x.data,data.y.data,data.z.data)
    s.u.g = s.u.mass.cgs.value
    Gconst = 6.674299999999999e-8 / (s.u.cm * s.u.cm * s.u.cm / (s.u.g * s.u.s * s.u.s))
    four_pi_G = 4 * np.pi * Gconst
    Gxy = qshear * Omega * gx1 * gy1 / four_pi_G
    # Gxy2 = (qshear*Omega*gx2*gy2/four_pi_G)
    stress["Gxy"] = xr.DataArray(Gxy, coords=data.coords)

    return stress.mean(dim="y")

if __name__ == "__main__":
    pdata = LowZData()
    nmodels = len(pdata.mlist)
    COMM = MPI.COMM_WORLD
    mylist = [pdata.mlist[i] for i in range(nmodels) if i % COMM.size == COMM.rank]
    print(COMM.rank, mylist)

    for m in mylist:
        print(m)
        if "Zg3" in m: continue
        s = pdata.sa.set_model(m)
        outdir = os.path.join(s.savdir, "stress")
        os.makedirs(outdir, exist_ok=True)

        for num in s.nums:
            outfile = os.path.join(outdir, f"stress.{num:04d}.nc")
            if os.path.isfile(outfile): continue
            print(num)
            ds = s.load_vtk(num)

            data = ds.get_field(ds.field_list)
            stress = get_stress(s, data)

            stress.assign_coords(time=ds.domain["time"]).to_netcdf(outfile)
