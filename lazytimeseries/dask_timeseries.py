import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import zarr
import dask.array as da
from MDAnalysisTests.datafiles import PSF, DCD
import numpy as np
import time


def align_trajectory(u, outputfile):
    average = align.AverageStructure(
        u, u, select="protein and name CA", ref_frame=0
    ).run()
    ref = average.results.universe
    aligner = align.AlignTraj(
        u,
        ref,
        select="protein and name CA",
        filename=outputfile,
        in_memory=False,
    ).run()


def convert_universe_to_pos_timeseries(u, zarr_filename):
    shape = (u.trajectory.n_frames, u.atoms.n_atoms, 3)
    z = zarr.open(zarr_filename, mode="w", shape=shape, dtype="f4")
    for i in range(len(u.trajectory)):
        z[i] = u.trajectory[i].positions


def get_dask_timeseries_from(zarr_filename):
    z = zarr.open(zarr_filename)
    return da.from_zarr(z)


def calculate_rmsf_displace(positions):
    # scalar_dispacement = da.sqrt((positions**2).sum(axis=2))
    mean_positions = positions.mean(axis=0)
    subtracted_positions = positions - mean_positions
    squared_deviations = subtracted_positions**2
    avg_squared_deviations = squared_deviations.mean(axis=0)
    sqrt_avg_squared_deviations = da.sqrt(avg_squared_deviations)
    return da.sqrt((sqrt_avg_squared_deviations**2).sum(axis=1))


if __name__ == "__main__":
    # u = mda.Universe(PSF, DCD)
    # align_trajectory(u, "positions_aligned.dcd")
    u_aligned = mda.Universe(
        "yiip_equilibrium/YiiP_system.pdb",
        "yiip_positions_aligned.xtc",
    )
    # convert_universe_to_pos_timeseries(u_aligned, "traj_aligned.zarr")

    dask_timeseries = get_dask_timeseries_from("yiip_positions.zarr")
    np.testing.assert_allclose(
        dask_timeseries[:], u_aligned.trajectory.timeseries(order="fac")
    )

    myt1 = time.time()
    rmsf_last = calculate_rmsf_displace(dask_timeseries)
    myresult2 = rmsf_last.compute()
    myt2 = time.time()

    mdt1 = time.time()
    R = rms.RMSF(u_aligned.atoms).run()
    mdt2 = time.time()

    print(f"Dask RMSF time was: {myt2 - myt1}")
    print(f"MDAnalysis RMSF time was: {mdt2 - mdt1}")

    np.testing.assert_allclose(R.results.rmsf, myresult2)
    # Good enough!
