import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
import astropy.units as u

"""
22/06/2026 README:

    This script generates an inverse ecliptic mask at 30 deg
    This is used for the zodi mask in the combined bitmask (make-bitmask.py) 
    for the 65um band, which is the most dominated by zodi.
    
    It is much faster to generate a 2048 map, so all acitons are performed 
    on nside 2048 and then ud_graded and rebinarised to the desired nside_out

    Ask Katrine if you have any questions to this script. 

"""

nside_base = 2048
nside_out = 8192      # set to desired output NSIDE
cut = 30.0            # ecliptic latitude cut in degrees

######### No need to change anything below this line
# Generate at NSIDE=2048
npix = hp.nside2npix(nside_base)
pix = np.arange(npix)

theta, phi = hp.pix2ang(nside_base, pix)

c = SkyCoord(
            l=np.degrees(phi) * u.deg,
            b=(90.0 - np.degrees(theta)) * u.deg,
            frame="galactic"
            )

ecl = c.transform_to(BarycentricMeanEcliptic())

# Inverse mask: 1 outside ±30°, 0 inside
mask = (np.abs(ecl.lat.deg) > cut).astype(np.float32)

# Upgrade if requested
if nside_out != nside_base:
        mask = hp.ud_grade(mask, nside_out=nside_out)

        # Re-binarise after ud_grade
        mask = (mask > 0.5).astype(np.int16)

hp.write_map(f"mask_ecl_{int(cut)}deg_inv_n{nside_out}.fits",
             mask,
             dtype=np.int16,
             overwrite=True
             )

print(f"Written mask_ecl_{int(cut)}deg_inv_n{nside_out}.fits")
print(f"Masked fraction = {1 - mask.mean():.4f}")
