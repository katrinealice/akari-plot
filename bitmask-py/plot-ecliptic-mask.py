#!/usr/bin/env python3

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

fname = "mask_ecl_30deg_inv_n8192.fits"

m = hp.read_map(fname)
nside = hp.get_nside(m)
print("NSIDE =", nside)
print("min =", np.min(m))
print("max =", np.max(m))
print("unique values =", np.unique(m)[:20])

hp.mollview(
            m,
                title=fname,
                    cmap="binary",
                        min=0,
                            max=1
                            )

hp.graticule()

ww_path = '/mn/stornext/d16/www_cmb/katrinag/'

plt.savefig(ww_path+'bitmask/n'+f'{nside}/'+fname+'.png',
                    bbox_inches='tight',
                    dpi=300)
