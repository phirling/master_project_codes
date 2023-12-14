import numpy as np
from pNbody import mapping
from my_format import GridSnapshot
from swiftsimio import load
import unyt

def indices_particles_in_box(pos,x0,x1):
    in_x = np.logical_and(pos[:,0] >= x0, pos[:,0] <= x1)
    in_y = np.logical_and(pos[:,1] >= x0, pos[:,1] <= x1)
    in_z = np.logical_and(pos[:,2] >= x0, pos[:,2] <= x1)
    in_all = np.logical_and(np.logical_and(in_x,in_y),in_z)
    return np.where(in_all == 1)[0]


def swift_to_grid(fname,N,b,frsp,output,xHII = 2.0e-4):
    """
    """
    grid_boxsize_kpc = 2*b*unyt.kpc #(xmax-xmin)

    data = load(fname)
    boxsize_kpc = data.metadata.boxsize[2].to(unyt.kpc).value
    # We project only the gas particles!
    mass_1e10Ms = (data.gas.masses.to(1e10 * unyt.Solar_Mass).value).astype(np.float32) # IMPORTANT: otherwise mkmap freaks out
    pos_kpc = data.gas.coordinates.to(unyt.kpc).value
    hsml_kpc = data.gas.smoothing_length.to(unyt.kpc).value
    u_kmps2 = (data.gas.internal_energy.to((unyt.km/unyt.s)**2).value).astype(np.float32)
    npart = pos_kpc.shape[0]

    dr = grid_boxsize_kpc / N
    dV = dr**3

    print("Ngrid:                   ",N)
    print("Number of particles:     ",npart)
    print("SWIFT Box size [kpc]:    ",boxsize_kpc)
    print("Grid Box size [kpc]:     ",grid_boxsize_kpc)
    print("Grid cell size [kpc]:    ",dr.value)

    # We normalize the positions to the grid box size. Any particle outside
    # the box size will be excluded from the gaussian map
    xmin = boxsize_kpc/2.0 - b
    xmax = boxsize_kpc/2.0 + b
    idx = indices_particles_in_box(pos_kpc,xmin,xmax)
    total_mass_in_box_1e10Ms = mass_1e10Ms[idx].sum()

    grid_pos_norm = ((pos_kpc - xmin) / grid_boxsize_kpc).value
    h = (frsp * hsml_kpc / grid_boxsize_kpc).value

    print(f"INFO: Doing SPH smoothing with rsp factor {frsp:.3f}")
    print("Smoothing mass...")
    M_filtered = mapping.mkmap3dksph(grid_pos_norm,mass_1e10Ms,np.ones(npart,np.float32),h, (N, N, N),verbose=1)
    print("Smoothing internal energy...")
    U_filtered = mapping.mkmap3dksph(grid_pos_norm,mass_1e10Ms * u_kmps2,np.ones(npart,np.float32),h, (N, N, N),verbose=1)
    u_filtered = np.where(M_filtered > 0,U_filtered / M_filtered,0.0)

    # Set correct units and convert
    M_filtered = M_filtered * 1e10 * unyt.Solar_Mass
    u_filtered = u_filtered * (unyt.km/unyt.s)**2
    dens_cgs = (M_filtered/dV).to('g/cm**3')
    ndens_cgs = (dens_cgs/unyt.mass_hydrogen).to('1/cm**3')
    u_cgs = u_filtered.to(unyt.erg/unyt.g)
    print(ndens_cgs.max())
    mass_cons_err = ( M_filtered.value.sum()/1e10 - total_mass_in_box_1e10Ms) / total_mass_in_box_1e10Ms
    print("Mass conservation error (relative):", mass_cons_err)

    # Save result
    xfrac = float(xHII) * np.ones((N,N,N))
    gs = GridSnapshot(N=N,dens_cgs=ndens_cgs.value,u_cgs=u_cgs.value,xfrac=xfrac,boxsize=grid_boxsize_kpc/1000,time=0.0)
    gs.write(str(output))