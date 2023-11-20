import numpy as np
import h5py

class GridSnapshot:
    def __init__(self,file=None,N=None,dens_cgs = None, temp_cgs = None, xfrac = None,boxsize = None) -> None:
        if file is None:
            if N is None:
                raise ValueError("When no input file is given, need to provide at least a grid size N")
            else:
                self.N = N
                if dens_cgs is None:    self.dens_cgs = np.zeros((N,N,N))
                else:                   self.dens_cgs = dens_cgs

                if temp_cgs is None:    self.temp_cgs = np.zeros((N,N,N))
                else:                   self.temp_cgs = temp_cgs

                if xfrac is None:       self.xfrac = np.zeros((N,N,N))
                else:                   self.xfrac = xfrac

                if boxsize is None:     self.boxsize = 0.0
                else:                   self.boxsize = boxsize

        else:
            with h5py.File(file,"r") as f:
                self.dens_cgs = np.array(f['dens_cgs'],dtype=np.float64)
                self.temp_cgs = np.array(f['temp_cgs'],dtype=np.float64)
                self.xfrac = np.array(f['xfrac'],dtype=np.float64)
                self.boxsize = float(f.attrs['boxsize'])
                self.N = self.dens_cgs.shape[0]
            pass

    def write(self,fname):
        print("Writing to",fname," ... ")
        with h5py.File(fname,"w") as f:
            f.attrs['boxsize'] = self.boxsize
            f.create_dataset('dens_cgs',data=self.dens_cgs,dtype=np.float64)
            f.create_dataset('temp_cgs',data=self.temp_cgs,dtype=np.float64)
            f.create_dataset('xfrac',data=self.xfrac,dtype=np.float64)
        print("done.")