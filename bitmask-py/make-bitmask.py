import healpy as hp
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import os

"""
22/06/2026 README:

It's much faster to work with the nside 2048 files, so this script is set up so we
can perform all the smoothing and threshold on the n2048 files and then ud_grade 
and re-binarise for higher NSIDES. 

* If your nside_out = 2048 it skips the ud_grade step. 

* ! OBS For the 65um band remember to include the inverse 30 deg ecliptic mask for the 
    zodi mask.

* OUTPUT map automatically contains NSIDE, BAND and the three thresholds in the 
  order as given below (zodi, gain, ncorr)

* INPUT map can be taken from the latest chain folder of the correct nside, e.g. 
  `globe/akari/latest_chains`

Ask Katrine if you have any questions to this script.

"""


def upgrade_bitmask(bitmask, nside_out):
    """
    Bit-masks cant just be ud_graded as the bitwise information
    is lost. This upgrade the different bits individually then recombines 
    """
    bits = [2, 4, 32]

    out = np.zeros(hp.nside2npix(nside_out), dtype=np.int32)

    for bit in bits:
        mask = ((bitmask & bit) != 0).astype(float)
        mask = hp.ud_grade(mask, nside_out) > 0.5
        out[mask] |= bit

    return out

########### Set paths and file names

parentpath = '/mn/stornext/d5/data/katrinag/work_comm3/akari/gain_mask/bitmask-py/'

input_sky = 'inputs/input_sky_model_AKARI_160-01.fits'
mask_ecl = 'mask_ecl_30deg_inv_n2048.fits'

nside_out = 8192
www_path = f'/mn/stornext/d16/www_cmb/katrinag/bitmask/n{nside_out}/'

### Set MASK THRESHOLDS 

include_ecliptic_mask = False        # Currently set this to True for 65um only
                  # Old values:
                  # [65um, 90um, 140um, 160um]
thres_zodi = 8    # [6,    8,    8,     8]
thres_gain = 100  # [100,  100,  100,   100]
thres_ncorr = 20  # [20,   12,   20,    20]

########### No need to change anything below this line

m = hp.read_map(parentpath+input_sky, field=0, dtype=np.float64)
nside = hp.get_nside(m)

# Smooth to 30 arcmin for the thresholding
fwhm = np.radians(30/60)
m_smooth = hp.smoothing(m, fwhm=fwhm)

bitmask = np.zeros(len(m), dtype=np.int32)
bitmask[m_smooth > thres_ncorr] |= 2  # Ncorr bit 2
bitmask[m_smooth > thres_gain] |= 4   # Gain bit 4

if include_ecliptic_mask:
    ecl_mask = hp.read_map(parentpath+mask_ecl, field=0, dtype=np.float64)
    zodi = (ecl_mask > 0.5) | (m_smooth > thres_zodi)
else:
    zodi = (m_smooth > thres_zodi)
bitmask[zodi] |= 32                   # Zodi bit 32

if nside_out > nside:
    bitmask = upgrade_bitmask(bitmask, nside_out)


# Save fits file to parentpath and plotted map to www_path
band = input_sky.split('AKARI_')[1].rsplit('-', 1)[0]
outfile = f'bitmask_{nside_out}_akari_{band}_{thres_zodi}-{thres_gain}-{thres_ncorr}'

hp.write_map(parentpath+outfile+'.fits', bitmask, dtype=np.int32, overwrite=True)
print(f'Written {outfile}.fits')

hp.mollview(bitmask, title=outfile, cmap='plasma', norm=None)
hp.graticule()
plt.savefig(
    os.path.join(www_path, outfile + '.png'),
    bbox_inches='tight',
    dpi=300
)
plt.close()
print(f'Saved {www_path}{outfile}.png')
