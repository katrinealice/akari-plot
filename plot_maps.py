import os
import glob

import healpy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cosmoglobe as cg
import cmocean.cm as cm
import cmasher as cmr

from astropy.io import fits

from matplotlib.colors import ListedColormap

# Sets style for all plots
plt.rcParams['font.size'] = '16'
plt.rcParams['savefig.facecolor']='white'
plt.rcParams['axes.titlepad'] = 18
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
plt.rcParams["errorbar.capsize"] = 5

# Utility functions
def get_akari_bands(chains):
    """Extracts the specific AKARI bands of the run"""
    bands = sorted(filename.split('_')[-1].replace('.txt', '')
                   for filename in os.listdir(chains)
                   if filename.startswith('filelist_AKARI_') and filename.endswith('.txt')
                   )

    return bands

def get_nside(fits_file):
    """Extracts the NSIDE value from the header of a FITS file."""  
    with fits.open(fits_file) as hdul:
        for hdu in hdul:
            if 'NSIDE' in hdu.header:
                return hdu.header['NSIDE']

    raise KeyError(f"No NSIDE keyword found in {fits_file}")

def _nonlinear_stretch(x):
    x = np.asarray(x)
    return np.sign(x) * np.log10(
        0.5 * (np.abs(x) + np.sqrt(4.0 + x**2))
    )

# Plotting class for AKARI maps
class AKARIMapPlotter:

    DEFAULT_MAP_CONFIGS = {
        'map':   {'vmin': -30, 'vmax': 30, 'toggle_last_only': False},
        'ncorr': {'vmin': -10, 'vmax': 10, 'toggle_last_only': True},
        'res':   {'vmin': -1,  'vmax': 1,  'toggle_last_only': True},
        'rms':   {'vmin': 0,   'vmax': 2,  'toggle_last_only': True},
    }

    def __init__(self, out_dir, chains_dir, bands, nside):
        self.out_dir = out_dir
        self.chains_dir = chains_dir
        self.bands = bands
        self.nside = nside
        self.resol = f'n{nside}'

    @classmethod
    def from_chains(cls, chains_dir, out_dir):

        bands = get_akari_bands(chains_dir)

        nside = get_nside(os.path.join(chains_dir,f'tod_AKARI_{bands[0]}_inst_c0001_k000001.fits'))

        print('bands:', bands)
        print('nside:', nside)

        return cls(
            out_dir=out_dir,
            chains_dir=chains_dir,
            bands=bands,
            nside=nside,
        )


    def _select_files(self, file_list, toggle_last_only):
        """reduces the filelist to max 11 maps (first 10 iterations + final iteration) if toggle_last_only is False, otherwise only the final iteration"""
        if len(file_list) == 0:
            return []

        if toggle_last_only:
            return [file_list[-1]]

        if len(file_list) <= 10:
            return file_list

        return file_list[:10] + [file_list[-1]]

    def _plot_single_map(self, map_data, band, outfile, vmin, vmax, cmap, figsize=(8,5), nonlinear=False):
        """Helper function to plot a single map with the given parameters and save it to the specified output file."""
        plot_vmin = vmin
        plot_vmax = vmax


        if nonlinear:
            map_data = _nonlinear_stretch(hp.ma(map_data))
            plot_vmin = _nonlinear_stretch(vmin)
            plot_vmax = _nonlinear_stretch(vmax)
            
            outfile = outfile.replace('.p', '_nonlinear.p')

        fig, ax = plt.subplots(figsize=figsize)
        plt.axes(ax)

        hp.mollview(
            map_data,
            title='',
            min=plot_vmin,
            max=plot_vmax,
            cmap=cmap,
            hold=True,
            cbar=False, # if true, roughly 50% cbar
            # unit=r'MJy/sr',
        )

        plt.annotate(f'{int(band)} $\\mu$m', 
                xy=(0.05, 0.95), 
                xycoords='axes fraction', 
                fontsize=18, 
                verticalalignment='top')

        hp.graticule(dmer=360, dpar=360, alpha=0)

        ## Full width cbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("bottom", size="5%", pad="1%")
 
        ## 70% cbar
        # cax = fig.add_axes([0.15, 0.08, 0.70, 0.03]) # left, bottom, width, height

        ## 50% cbar
        cax = fig.add_axes([0.25, 0.12, 0.50, 0.03]) # left, bottom, width, height

        sm = mpl.cm.ScalarMappable(
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=plot_vmin, vmax=plot_vmax)
        )
        sm.set_array([])

        print(f'vmin: {vmin}, vmax: {vmax}')
        print(f'vmin: {plot_vmin}, vmax: {plot_vmax}')

        cbar = fig.colorbar(
            sm,
            cax=cax,
            orientation='horizontal',
            ticks=[plot_vmin, plot_vmax]
        )

        cbar.set_ticklabels([f'{vmin:g}', f'{vmax:g}'])
    
        cbar.set_label(r'MJy/sr', labelpad=4)

        plt.savefig(outfile,
                    bbox_inches='tight',
                    dpi=300)

        plt.close()

    def plot_map_type(self, map_type, vmin, vmax, cmap, toggle_last_only=False, bands=None, figsize=(8,5), nonlinear=False, png=False):
        """Generates maps of the specified type for each band and saves them to the output directory. If bands is None, it defaults to all bands in self.bands."""

        if bands is None:
            bands = self.bands

        outtype = 'png' if png else 'pdf'

        for band in bands:
            print(f'generating {map_type} map for {band}')

            pattern = (f'{self.chains_dir}'f'tod_AKARI_{band}_{map_type}*')
            file_list = np.sort(glob.glob(pattern))

            selected_files = self._select_files(file_list,toggle_last_only)

            webpath = f'{self.out_dir}/{self.resol}/{band}/'
            os.makedirs(webpath, exist_ok=True)

            name_cut = 30 + len(band) + len(map_type)

            for filename in selected_files:
                if '.fits' not in filename:
                    continue

                map_data = hp.read_map(filename, memmap=True)

                file_title = filename[-name_cut:]
                outfile = f'{webpath}{file_title}.{outtype}'

                self._plot_single_map(map_data, band, outfile, vmin, vmax, cmap, figsize, nonlinear=nonlinear)

            print(f'{map_type} maps for {band} saved to {webpath}')

    def plot_band(self, band, map_type, vmin, vmax, cmap=cmr.fusion_r, toggle_last_only=True, figsize=(8,5), nonlinear=False, png=False):
        """wrapper for single band plot"""
        self.plot_map_type(map_type=map_type, vmin=vmin, vmax=vmax, cmap=cmap, 
                           toggle_last_only=toggle_last_only, bands=[band], figsize=figsize,
                           nonlinear=nonlinear, png=png)

    def plot_all(self, map_configs=None, cmap=cmr.fusion_r):
        """wrapper to plot all map types with their respective configurations"""
        if map_configs is None:
            map_configs = self.DEFAULT_MAP_CONFIGS

        for map_type, cfg in map_configs.items():

            self.plot_map_type(
                map_type=map_type,
                vmin=cfg['vmin'],
                vmax=cfg['vmax'],
                cmap=cmap,
                toggle_last_only=cfg['toggle_last_only'],
                figsize=cfg.get('figsize', (8, 5)),
                nonlinear=cfg.get('nonlinear', False),
                png=cfg.get('png', False)
            )

    def plot_zoom_ins(self, fov_deg=2, reso_arcmin=0.02, vmin=0,vmax=10,cmap=cm.thermal,toggle_last_only=True,bands=None, png=False):

        if bands is None:
            bands = self.bands

        coords = [
            (121.2, -21.6),                   # Andromeda
            (96, 30),                         # NEP
            (315, -55),                       # Southern Hole
            # (259.9663579319, 38.0740906071),  # U-Hydrae
        ]

        xsize = int(fov_deg * 60 / reso_arcmin)

        map_type = 'map'
        outtype = 'png' if png else 'pdf'

        for band in bands:
            print(f'generating zoom-in {map_type} map for {band}')

            pattern = (f'{self.chains_dir}tod_AKARI_{band}_{map_type}*')

            file_list = np.sort(glob.glob(pattern))
            selected_files = self._select_files(file_list,toggle_last_only)

            webpath = f'{self.out_dir}/{self.resol}/{band}/'
            os.makedirs(webpath, exist_ok=True)

            name_cut = 30 + len(band) + len(map_type)

            for filename in selected_files:
                fitsmap = hp.read_map(filename, memmap=True)
                for coord in coords:

                    hp.gnomview(
                        fitsmap,
                        rot=coord,
                        coord='G',
                        title=filename[-name_cut:],
                        min=vmin,
                        max=vmax,
                        cmap=cmap,
                        xsize=xsize,
                        reso=reso_arcmin,
                    )

                    outfile = (f'{webpath}gnomview_{filename[-name_cut:]}_{coord[0]}_{coord[1]}.{outtype}')

                    plt.savefig(outfile,
                                bbox_inches='tight',
                                dpi=300)
                    plt.close()

            print(f'done for {band}')

    def plot_cls(self,bands=None,map_type='map',outfile='cls.pdf'):
        """Compute and plot power spectra of the final map iteration."""

        if bands is None:
            bands = self.bands

        cls = []

        for band in bands:
            print(f'computing Cl for {band}')

            pattern = (f'{self.chains_dir}tod_AKARI_{band}_{map_type}*')

            file_list = np.sort(glob.glob(pattern))
            if len(file_list) == 0:
                print(f'No files found for {band}')
                continue

            map_data = hp.read_map(file_list[-1],memmap=True)

            cls.append(hp.anafast(map_data))

        fig, axs = plt.subplots(
            len(bands),
            1,
            sharex=True,
            figsize=(6, 2.5*len(bands)),
            gridspec_kw={'hspace': 0})

        if len(bands) == 1:
            axs = [axs]

        for ax, band, cl_band in zip(axs, bands, cls):
            ax.loglog(cl_band,label=f'{band} $\\mu$m',c='k')
            ax.legend(frameon=False)
            axs[-1].set_xlabel(r'Multipole $\ell$')

        fig.supylabel(r'Power spectrum, $C_\ell$ (MJy$^2$/sr)',x=-0.02)

        webpath = f'{self.out_dir}/{self.resol}/'
        os.makedirs(webpath, exist_ok=True)

        plt.savefig(
            f'{webpath}{outfile}',
            bbox_inches='tight',
            dpi=300)

        plt.close()

        print(f'Cl plot saved to {webpath}{outfile}')


    def plot_official_maps(self, official_path, nside=None, factor=1.0, vmin=-30, vmax=30, cmap=cmr.fusion_r, bands=None, nonlinear=False, png=False):

        if bands is None:
            bands = self.bands

        if nside is None:
            nside = self.nside

        outtype = 'png' if png else 'pdf'

        for band in bands:

            print(f'generating official map for {band}')
            filename = os.path.join(official_path,f'akari_{band}_1_{nside}.fits')
            print(f'looking for file: {filename}')

            if not os.path.exists(filename):
                print(f'file not found: {filename}')
                continue

            map_data = hp.read_map(filename,memmap=True)
            map_data *= factor

            webpath = f'{self.out_dir}/n{nside}/{band}/'
            os.makedirs(webpath, exist_ok=True)

            outfile = (f'{webpath}official_akari_{band}_x{factor}.{outtype}')
            self._plot_single_map(map_data, band, outfile, vmin, vmax, cmap, nonlinear=nonlinear)

            print(f'saved {outfile}')
  

def main():
    OFFICIAL_PATH = ('/mn/stornext/d23/cmbco/globe/akari/maps/')
    parent_dir = "/mn/stornext/u3/katrinag/data_path/work_comm3/akari/gain_mask/n2048/"
    chains_dir = os.path.join(parent_dir, "chains_akari_160_n2048_20/")
    out_dir = "/mn/stornext/d16/www_cmb/katrinag/akari_160_bitmask-20/"

    # Defines the plotter object
    plotter = AKARIMapPlotter.from_chains(chains_dir=chains_dir, out_dir=out_dir)

    # Defines the cmap
    parchment = ListedColormap(np.loadtxt("parchment1.dat")/255.)
    cmap = parchment # cmr.fusion_r

    # ===================== COSMOGLOBE MAPS ===================== 
    ### FOR CREATING ALL THE DEFAULT OUTPUTS -- UNCOMMENT TO RUN
    MAP_CONFIGS = {
        'map':   {'vmin': -30, 'vmax': 30, 'toggle_last_only': True, 'figsize': (16, 10)},
        'ncorr': {'vmin': -10, 'vmax': 10, 'toggle_last_only': True},
        'res':   {'vmin': -1, 'vmax': 1, 'toggle_last_only': True},
        'rms':   {'vmin': 0, 'vmax': 8, 'toggle_last_only': True},
    }

    plotter.plot_all(map_configs=MAP_CONFIGS, cmap=cmap)

    ### FOR CREATING A SINGLE MAP TYPE WITH CUSTOM CONFIGURATION -- UNCOMMENT TO RUN
    # plotter.plot_band(band='160', 
    #                   map_type='map', 
    #                   vmin=-10,          #-30
    #                   vmax=1000,         #30
    #                   cmap=cmr.fusion_r, #parchment
    #                   nonlinear=True, 
    #                   toggle_last_only=True, 
    #                   figsize=(16,10), 
    #                   png=True,
    #                   )

    ### FOR CREATING ZOOM-IN PLOTS -- UNCOMMENT TO RUN
    #plotter.plot_zoom_ins(vmin=-27, vmax=102)

    ### FOR CREATING ZOOM-IN PLOTS FOR A SINGLE BAND -- UNCOMMENT TO RUN
    #plotter.plot_zoom_ins(bands=['160'], fov_deg=2, reso_arcmin=0.02, vmin=-27, vmax=102)

    ### FOR PLOTTING POWER SPECTRA -- UNCOMMENT TO RUN
    #plotter.plot_cls()

    # ===================== OFFICIAL MAPS =====================
    #plotter.plot_official_maps(official_path=OFFICIAL_PATH, nside=8192,factor=1,vmin=-10,vmax=10,)



if __name__ == "__main__":
    main()
