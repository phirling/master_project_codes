import numpy as np
import matplotlib.pyplot as plt
import argparse
from my_format import GridSnapshot
from matplotlib.colors import CenteredNorm

parser = argparse.ArgumentParser()
parser.add_argument("file", nargs="+", help="Nbody file")
parser.add_argument("-z", type=int,default=None)
parser.add_argument("--relative",action='store_true')
args = parser.parse_args()

gs0 = GridSnapshot(args.file[0])
gs1 = GridSnapshot(args.file[1])

xfrac0 = gs0.xfrac
xfrac1 = gs1.xfrac
boxsize = gs0.boxsize

if args.relative:
    xfracdiff = (xfrac0 - xfrac1) / xfrac1
else:
    xfracdiff = (xfrac0 - xfrac1)

print(xfracdiff.max())
print(xfracdiff.min())

extent = (0,1000*boxsize,0,1000*boxsize)
N = xfrac0.shape[0]
ctr = N//2-1

if args.z is None:
    zl = ctr
else:
    zl = int(args.z)

fig, ax = plt.subplots(1,figsize=(5,5))
ax.set_xlabel("$x$ [kpc]")
ax.set_ylabel("$y$ [kpc]")

im0 = ax.imshow(xfracdiff[:,:,zl].T,extent=extent,origin='lower',norm=CenteredNorm(),cmap='RdBu')

plt.colorbar(im0)

plt.show()