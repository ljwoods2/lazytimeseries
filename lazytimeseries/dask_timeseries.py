import MDAnalysis as mda
from MDAnalysis.analysis import align
import zarr
import dask.array as da

def align_trajectory(u, outputfile):
    average = align.AverageStructure(u, u, select='protein and name CA',
                                 ref_frame=0).run()
    ref = average.results.universe
    aligner = align.AlignTraj(u, ref,
                           select='protein and name CA',
                           filename=outputfile,
                           in_memory=False).run()

def convert_universe_to_pos_timeseries(u, zarr_filename):
    shape = (u.trajectory.n_frames, u.atoms.n_atoms, 3)
    z = zarr.open(zarr_filename, mode='w', shape=shape, dtype='f4')
    for i in range(len(u.trajectory)):
        z[i] = u.trajectory[i].positions
        
def get_dask_timeseries_from(zarr_filename):
    z = zarr.open(zarr_filename)
    return da.from_zarr(z)

def calculate_rmsf(positions):
    """
    Calculate the Root Mean Squared Fluctuation (RMSF) for each atom given a dask array of positions.
    
    Parameters:
        positions (dask.array.Array): A Dask array of shape (n_frames, n_atoms, n_dimensions),
                                      where n_frames is the number of frames, n_atoms is the number
                                      of atoms, and n_dimensions is the spatial dimensions (e.g., 3 for x, y, z).
    
    Returns:
        dask.array.Array: A Dask array of shape (n_atoms,) containing the RMSF for each atom.
    """
    # Compute the mean positions of each atom across all frames
    mean_positions = positions.mean(axis=0)
    
    # Compute the squared deviations from the mean for each frame
    squared_deviations = (positions - mean_positions) ** 2
    
    # Compute the mean squared deviations for each atom
    mean_squared_deviations = squared_deviations.mean(axis=0)
    
    # Compute the RMSF by taking the square root of the mean squared deviations
    rmsf = da.sqrt(mean_squared_deviations.mean(axis=1))
    
    return rmsf

if __name__ == "__main__":
    #u = mda.Universe("yiip_equilibrium/YiiP_system.pdb", "yiip_equilibrium/YiiP_system_90ns_center.xtc")
    #align_trajectory(u, 'yiip_positions_aligned.xtc')
    u_aligned = mda.Universe("yiip_equilibrium/YiiP_system.pdb", "yiip_positions_aligned.xtc")
    convert_universe_to_pos_timeseries(u_aligned, 'yiip_positions_aligned.zarr')
    dask_timeseries = get_dask_timeseries_from('yiip_positions_aligned.zarr')
    rmsf = calculate_rmsf(dask_timeseries)
    rmsf.compute() 