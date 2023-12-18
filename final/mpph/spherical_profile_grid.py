from mpph.my_format import GridSnapshot
import numpy as np
from astropy.constants import m_p, k_B
from scipy.interpolate import CubicSpline

__all__ = ['get_spherical_profile_grid',
           'get_temperature_profile_grid',
           'get_xfrac_profile_grid',
           'get_density_profile_grid',
           'interp_xfrac_profile']

def get_spherical_profile_grid(data,nbins):
    """
    General function to extract the spherical profile of a quantity from a grid snapshot
    
    This method implicitely assumes that the origin is at the center of the grid
    and that the grid is even (N is even)

    It essentially computes the average value of data in spherical shells
    """
    N = data.shape[0]

    # First, we get the radius from the origin of each cell
    # in the grid, in grid coordinates
    xsp_full = np.linspace(-(N-1)/2 , +(N-1)/2, N)
    xx,yy,zz = np.meshgrid(xsp_full,xsp_full,xsp_full)
    rr = np.sqrt(xx**2 + yy**2 + zz**2)

    # Define the radial bins, still in grid coordinates
    rbin_edges = np.linspace(0,N/2.,nbins)
    rbin_centers = rbin_edges[:-1] + np.diff(rbin_edges) / 2.

    # Now, we work on the flat arrays (easier for indexing)
    rr = rr.flatten()
    data = data.flatten()

    # Loop through spherical shells and find average value of data
    data_binned = np.empty(nbins-1)
    for i in range(nbins-1):
        i_shell = np.where(np.logical_and(rr > rbin_edges[i],rr <= rbin_edges[i+1]))
        data_shell = data[i_shell]
        data_binned[i] = data_shell.mean()

    # Return the mean value in each shell, along with the center of the corresponding radial bin
    return data_binned, rbin_centers

def get_xfrac_profile_grid(filename,nbins,ionized=False):
    """
    Extract the spherical profile of the ionized/neutral fraction

    Parameters
    ----------
    filename : string
        Name of the grid snapshot file
    nbins : int
        Number of radial bins to use (should not be >N/2)
    ionized : bool
        Plot the ionized rather than neutral fraction

    Returns
    -------
    rbin_centers : array
        Centers of the radial bins in kpc
    x_profile : array
        Mean value of the ionized fraction in each bin
    """
    gs = GridSnapshot(filename)
    N = gs.N
    boxsize_kpc = gs.boxsize * 1000 # Grid stores boxsize in Mpc
    dr_kpc = boxsize_kpc / N

    if ionized:
        xfrac = gs.xfrac
    else:
        xfrac = 1.0 - gs.xfrac
    
    x_shells, rbin_centers = get_spherical_profile_grid(xfrac,nbins)

    # Convert to physical units
    rbin_centers *= dr_kpc

    return rbin_centers, x_shells

def get_temperature_profile_grid(filename,nbins,XH=1.0,gamma=5.0/3.0):
    """
    Extract the spherical profile of the temperature

    Parameters
    ----------
    filename : string
        Name of the grid snapshot file
    nbins : int
        Number of radial bins to use (should not be >N/2)
    XH : float
        Mass fraction of hydrogen in the gas. Default: 1.0
    gamma : float
        Adiabatic index. Default: 5/3

    Returns
    -------
    rbin_centers : array
        Centers of the radial bins in kpc
    temp_profile : array
        Mean value of the temperature in each bin
    """
    gs = GridSnapshot(filename)
    N = gs.N
    boxsize_kpc = gs.boxsize * 1000 # Grid stores boxsize in Mpc
    dr_kpc = boxsize_kpc / N
    internal_energy = gs.u_cgs

    # Compute mean molecular weight and temperature
    mu_grid = 1.0 / (XH * (1+gs.xfrac+0.25*(1.0/XH-1.0)))
    temp = ((gamma-1) * mu_grid * m_p.cgs.value/k_B.cgs.value * internal_energy)

    temp_shells, rbin_centers = get_spherical_profile_grid(temp,nbins)

    # Convert to physical units
    rbin_centers *= dr_kpc

    return rbin_centers, temp_shells


def get_density_profile_grid(filename,nbins,XH=1.0):
    """
    Extract the spherical profile of the hydrogen number density

    Parameters
    ----------
    filename : string
        Name of the grid snapshot file
    nbins : int
        Number of radial bins to use (should not be >N/2)
    XH : float
        Mass fraction of hydrogen in the gas. Default: 1.0

    Returns
    -------
    rbin_centers : array
        Centers of the radial bins in kpc
    ndens_profile : array
        Mean value of the H number density in each bin
    """
    gs = GridSnapshot(filename)
    N = gs.N
    boxsize_kpc = gs.boxsize * 1000 # Grid stores boxsize in Mpc
    dr_kpc = boxsize_kpc / N
    mdens_gas = gs.dens_cgs * m_p.cgs.value
    ndens_hydrogen = (XH * mdens_gas) / m_p.cgs.value

    ndens_shells, rbin_centers = get_spherical_profile_grid(ndens_hydrogen,nbins)

    # Convert to physical units
    rbin_centers *= dr_kpc

    return rbin_centers, ndens_shells

def interp_xfrac_profile(r,xHI,val):
    """
    Assuming the neutral fraction profile is steady,
    estimate the radius at which they reach a given value.
    This can be used e.g. to compute the radius of the neutral
    core of a halo.

    Parameters
    ----------
    r : array
        Radii of the profile
    xHI : array
        Neutral fraction profile
    val : float
        Value of the ionized fraction of which to estimate the radius
    """
    N = len(r)
    
    # Function must be monotically increasing for interpolation methods
    negative_log_val = -np.log10(val)
    negative_log_xHI = -np.log10(xHI)
    
    # We need to find the range of the profile where x is monotically increasing
    dlx = np.diff(negative_log_xHI)
    ii = np.argmin(np.abs(negative_log_xHI - negative_log_val)) # Value of profile closest to val
    delt = N // 10 + 1 # Start the window at N/10
    il = 0
    ir = N
    # Start by testing the whole profile, if not incr, use a window around ii of decreasing size
    while np.any(dlx[il:ir] < 0):
        delt -= 1
        il = ii - delt
        ir = ii + delt + 1
        if delt <= 0:
            raise RuntimeError("Cannot find a monotically increasing range of the profile")
    
    # Interpolate the inverse function r(-log10(x)) at x = val    
    itp = CubicSpline(negative_log_xHI[il:ir],r[il:ir])
    rval = itp(negative_log_val)
    return rval