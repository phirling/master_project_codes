import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import argparse

def get_data(file,z=None):
    with open(file,"rb") as f:
        data = pkl.load(f)
    s = file.split('_')[-1]
    if s[1] == '.':
        i = int(s[0])
    else:
        i = int(s[0:2])
    N = data.shape[0]
    if z is None:
        z = N//2 - 1
    return data, z, i

def plot_xfracslice(file,ax,z=None,xmin=2e-4,xmax=1):
    data, z, i = get_data(file,z)
    #t_Myr = i * dt_Myr
    #ax.set_title(f"$t={t_Myr:.2f}$ Myr")
    return ax.imshow(data[:,:,z].T,norm='log',vmin=xmin,vmax=xmax,origin='lower',cmap='Spectral_r')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="+", help="Nbody file")
    parser.add_argument("-z", type=int,default=None)
    args = parser.parse_args()
    fname = args.file[0]

    fig, ax = plt.subplots()
    im = plot_xfracslice(fname,ax)
    plt.colorbar(im)
    plt.show()