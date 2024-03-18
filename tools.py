import numpy as np
import matplotlib.pyplot as plt
import xarray as xr 
from tqdm import tqdm
import pyproj
from timeit import default_timer as timer
import warnings
import tempfile 
import glob
import os
import shutil

from luti.xarray import invert_data_array
from luti import Alphachecker
from luti import LinearInterpolator

from .retrieval_functions import fast_retrieve
from .conveniences import read_LUT, save_as_netcdf
from .paths import LIBRADTRAN_PATH


class PixelInterpolator(object):
    """The PixelInterpolator is supposed to replace the runmacs BadPixelFixer 
    functionality. Since A(C)³ scenes are typically dark with low sun angle,
    some pixels are unreliable, even though they might work for a different 
    scene. You can choose to interpolate invalid pixels, according to the 
    'valid' variable, or dynamically find unreliable pixel rows. data should
    contain a 'radiance' variable."""

    # TODO: Add dimension to output array, indicating what pixels were interpolated

    def __init__(self, swir_ds, window=None):
        self.data = swir_ds.copy()
        self.window = window
        self.cutoffs = None
        self.interpolated_from_list = False
        self.applied_filter = False

        # Estimate unreliable pixels 
        int_slopes = (np.square(swir_ds.radiance.differentiate(coord='x'))
                      .integrate(coord='time'))
        self.data['int_slopes'] = int_slopes/int_slopes.mean(dim='x')
        if self.window is not None:
            self.data['int_slopes_mva'] = self.data.int_slopes\
                                          .rolling(x=self.window).mean()

    def show_signals(self):
        # Plots normalised signal to choose thresholds.
        for wvl in self.data.wavelength: 
            fig, ax = plt.subplots(nrows=1, figsize=(11,3), sharex=True)
            self.data.int_slopes.sel(wavelength=wvl, method='nearest')\
            .plot(linewidth=0.9, linestyle='dotted', 
                  label='Integrated square spatial slopes')
            if self.window is not None:
                self.data.int_slopes_mva.sel(wavelength=wvl, method='nearest')\
                .plot(linewidth=0.9, 
                      label=f"Rolling average x={self.window} applied")
            if self.cutoffs is not None:
                cutoff = self.data.sel(wavelength=wvl).cutoff.values
                ax.axhline(cutoff, 
                           color='red', label=f"Filter cutoff at {cutoff}")
            ax.set_ylabel(r'Normalised signal $D(x)$')
            plt.grid()
            plt.ylim((0,5))
            plt.legend()
            plt.show()

    def add_cutoffs(self, cutoffs):
        # Add cutoffs as list necessary before filtering. show_signals() again 
        # to adapt values
        if len(cutoffs) != len(self.data.wavelength):
            raise Exception("""Number of passed cutoffs does not equal 
                            number of wavelengths in data""")
        self.cutoffs = cutoffs
        cutoff_da = xr.DataArray(data=self.cutoffs, dims="wavelength",
                                 coords={"wavelength":self.data\
                                         .wavelength.values})
        self.data = self.data.assign(cutoff=cutoff_da)

    def interpolated_radiance(self, with_bpl=True):
        # Interpolated over nan values and invalid pixels per default. Does not
        # apply filter
        if with_bpl:
            interpolated_radiance = (self.data.radiance
                                     .where(self.data.valid==1)
                                     .interpolate_na(dim='x', method='spline', 
                                                     use_coordinate=False))
        else: 
            interpolated_radiance = (self.data.radiance
                                    .interpolate_na(dim='x', method='spline', 
                                                     use_coordinate=False))
        self.interpolated_from_list = True

        return interpolated_radiance

    def filtered_radiance(self, with_bpl=True, remove=False):
        # Applies filter and interpolates all nan values
        if self.cutoffs is None:
            raise Exception("""Enter cutoff list with 'add_cutoffs()'
                            before applying filter""")
        
        if with_bpl:
            selection = (self.data.radiance
                        .where(self.data.valid==1)
                        .where(self.data.int_slopes_mva<self.data.cutoff))
        else: 
            selection = (self.data.radiance
                        .where(self.data.int_slopes_mva<self.data.cutoff))
        
        if remove:
            filtered_radiance = selection.dropna(dim='x', how='all')

        if not remove:
            filtered_radiance = selection.interpolate_na(dim='x', method='spline', 
                                                         use_coordinate=False)
        self.applied_filter = True

        return filtered_radiance
 
    
def solar_flux_kurudz():
    """Loads the .dat file from the libradtran directory specified in .paths.py
    and formats it as an xarray dataarray to make compatible with other 
    datasets, such as solar positions and camera data."""
    solar_flux = np.genfromtxt(LIBRADTRAN_PATH+"/data/solar_flux/kurudz_0.1nm.dat")

    solar_flux = xr.DataArray(data=solar_flux[:,1], dims=["wavelength"], 
                                        coords=dict(wavelength = solar_flux[:,0],), 
                                        attrs=dict(measurement="Solar flux as in Kurudz", 
                                        units="mW m**-2 nm**-1",))

    solar_flux.wavelength.attrs["units"] = r'nm'
    
    return solar_flux
    

class SceneInterpreter(object):
    """Takes camera at different wavelengths, as well as view angles and solar
    positions that cane be sources from the conveniences.py submodule 
    functions. Provides functions to compute additional variables and interpret 
    spectral data."""

    def __init__(self, camera_data, view_angles, solar_positions):
        self.camera_data = camera_data.copy()
        self.view_angles = (view_angles.copy()
                            .interp(time=self.camera_data.time)
                            .interp(x=self.camera_data.x))
        self.solar_positions = (solar_positions.copy()
                                .interp(time=self.camera_data.time))
        
    def umu(self):
        # As to be passed to uvspec
        umu = np.cos(2.*np.pi*self.view_angles.vza/360.)
        return umu.rename('umu')

    def phi(self):
        # Relative solar azimuth as to be passed to uvspec.
        # saa and vaa relative to WGS84
        # --> 180 deg corresponds to sun behind sensor
        # --> 0 deg corresponds to sensor view towards sun
        phi = (self.view_angles.vaa - self.solar_positions.saa.mean())%360
        return phi.rename('phi')
    
    def reflectivity(self):
        """Returns an additional variable to calibrated SWIR or VNIR data, 
        based on a nas file containing halo data and the radiance measurements, 
        that can be merged with the other datsets. Now also considers Earth-Sun 
        distance."""
        
        solar_flux = (solar_flux_kurudz()
                      .interp(wavelength=self.camera_data.wavelength.values))
        
        reflectivity = (self.camera_data["radiance"]
                        *(self.solar_positions.d**2)
                        *np.pi
                        /(solar_flux*np.cos(2.*np.pi
                                            *self.solar_positions.sza/360.)))
        
        return reflectivity.rename('reflectivity')
    
    def reflectance(self):
        """Returns reflectance as e.g. defined in Ehrlich 2008 ice index paper.
        """

        solar_flux = (solar_flux_kurudz()
                      .interp(wavelength=self.camera_data.wavelength.values))
        
        reflectance = (self.camera_data["radiance"]
                      *(self.solar_positions.d**2)
                      *np.pi
                      /solar_flux)
        
        return reflectance.rename('reflectance')
    
    def ice_index_Ehrlich2008(self):
        """Returns spectral ice index following the equation 
        by Ehrlich et al. 2008"""
        
        I_s = (self.reflectance().polyfit(dim='wavelength', deg=1).sel(degree=1)
               /self.reflectance().sel(wavelength=1640, method='nearest'))
        I_s = I_s.rename({'polyfit_coefficients':'ice_index_Ehrlich2008'})*100
        
        return I_s.ice_index_Ehrlich2008
        
    def ice_index_Jaekel2013(self):
        """Is positive for ice clouds and negative for water clouds."""
        I_J = ((self.camera_data.radiance.sel(wavelength=1700, method='nearest')
               - self.camera_data.radiance.sel(wavelength=1550, method='nearest'))
               /self.camera_data.radiance.sel(wavelength=1700, method='nearest'))
        I_J = I_J.rename('ice_index_Jaekel2013')*100
        return I_J
    
    def ice_index_Knap2002(self):
        """Is zero for water clouds and positive for ice clouds."""
        I_K = ((self.reflectivity().sel(wavelength=1700, method='nearest')
               - self.reflectivity().sel(wavelength=1640, method='nearest'))
               /self.reflectivity().sel(wavelength=1640, method='nearest'))
        I_K = I_J.rename('ice_index_Knap2002')*100
        return I_K
    
    
    def overview(self):
        vza = self.view_angles.vza
        vaa = self.view_angles.vaa

        umu = self.umu()
        phi = self.phi()

        sza_mean = self.solar_positions.sza.mean().values
        saa_mean = self.solar_positions.saa.mean().values

        fig, ax = plt.subplots(ncols=2, figsize=(16,4), sharey=True)
        vza.plot(ax=ax[0], x='time', alpha=0.8)
        plt.clabel(vza.plot.contour(ax=ax[0], x='time', cmap='coolwarm_r', 
                                    levels=10), 
                   inline=True, fontsize=12)
        ax[0].set_title("")
        vaa.plot(ax=ax[1], x='time', alpha=0.8)
        plt.clabel(vaa.plot.contour(ax=ax[1], x='time', cmap='coolwarm_r', 
                                    levels=10), 
                   inline=True, fontsize=12)
        ax[1].set_title("")
        ax[1].set_ylabel(None)
        fig.suptitle("Absolute viewing angles")
        plt.tight_layout()

        fig, ax = plt.subplots(ncols=2, figsize=(16,4), sharey=True)
        umu.plot(ax=ax[0], x='time', alpha=0.8)
        plt.clabel(umu.plot.contour(ax=ax[0], x='time', cmap='coolwarm_r', 
                                    levels=10), 
                   inline=True, fontsize=12)
        ax[0].set_title("")
        phi.plot(ax=ax[1], x='time', alpha=0.8)
        plt.clabel(phi.plot.contour(ax=ax[1], x='time', cmap='coolwarm_r', 
                                    levels=10), 
                   inline=True, fontsize=12)
        ax[1].set_title("")
        ax[1].set_ylabel(None)
        fig.suptitle("Relative viewing angles")
        plt.tight_layout()
        
        fig, ax = plt.subplots(ncols=2, figsize=(16,4))
        self.solar_positions.sza.plot(ax=ax[0])
        ax[0].axhline(sza_mean, color='red', linestyle='dotted', 
                      label=f"Mean solar zenith angle = {sza_mean}")
        ax[0].legend()
        self.solar_positions.saa.plot(ax=ax[1])
        ax[1].axhline(saa_mean, color='red', linestyle='dotted', 
                      label=f"Mean solar azimuth angle = {saa_mean}")
        ax[1].legend()
        fig.suptitle("Solar positions along track")
        plt.tight_layout()

        umu_min, umu_max = umu.min().values, umu.max().values
        phi_min, phi_max = phi.min().values, phi.max().values

        vza_min, vza_max = vza.min().values, vza.max().values
        vaa_min, vaa_max = vaa.min().values, vaa.max().values

        fig, ax = plt.subplots(ncols=2, figsize=(16,4))
        vza.plot.hist(bins=150, ax=ax[0], 
                      label=f"vza PDF with min={vza_min:.2f} and max={vza_max:.2f}", 
                      density=True)
        ax[0].legend()
        ax[0].set_title(None)
        vaa.plot.hist(bins=150, ax=ax[1], 
                      label=f"vaa PDF with min={vaa_min:.2f} and max={vaa_max:.2f}", 
                      density=True)
        ax[1].legend()
        ax[1].set_title(None)
        fig.suptitle("Absolute viewing angle relevenat ranges")
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(ncols=2, figsize=(16,4))
        umu.plot.hist(bins=150, ax=ax[0], 
                      label=f"umu PDF with min={umu_min:.2f} and max={umu_max:.2f}", 
                      density=True)
        ax[0].legend()
        ax[0].set_title(None)
        phi.plot.hist(bins=150, ax=ax[1], 
                      label=f"phi PDF with min={phi_min:.2f} and max={phi_max:.2f}", 
                      density=True)
        ax[1].legend()
        ax[1].set_title(None)
        fig.suptitle("Realtive viewing angle relevenat ranges")
        plt.tight_layout()
        plt.show()

        return sza_mean, saa_mean, umu_min, umu_max, phi_min, phi_max
    
    def cloud_pixels(self, wavelenth=1550, threshold=3.):

        return

    
    def merged_data(self):

        merged_data = xr.merge([self.camera_data, self.umu(), self.phi(),
                                self.reflectivity(),
                                self.solar_positions, self.view_angles])
        
        return merged_data
    
    def cloud_properties_fast_BSR(self, invertedLUT, wvl1, wvl2, R1_name, R2_name,
                                  umu_bins=None, phi_bins=None,
                                  interpolate=False):
        """Takes an inverted LUT, i.e. with r_eff and tau550 variables, and 
        resamples according to SWIR pixels of approximate equal geometry. 
        Returns r_eff, tau550 as tuple."""

                            # From retrieval_functions.py
        cloud_properties = fast_retrieve(invertedLUT, self.merged_data(),
                                         wvl1, wvl2, R1_name, R2_name,
                                         vza_bins=umu_bins, phi_bins=phi_bins,
                                         interpolate=interpolate)
        return cloud_properties
    

class BSRLookupTable(object):

    def __init__(self, LUT):
        if LUT.wvl.size != 2:
            raise Exception("Please choose two wavelengths.")
        self.dataset = LUT.copy()
        self.wvl1 = self.dataset.isel(wvl=0).wvl.values.item()
        self.wvl2 = self.dataset.isel(wvl=1).wvl.values.item()
        # Choose names to be used in LUTI inversion
        self.Rone_name = f"R({self.wvl1}nm)"
        self.Rtwo_name = f"R({self.wvl2}nm)"

    @classmethod
    def from_path(cls, path):
        LUT = read_LUT(path)

        return cls(LUT)

    def display_nadir(self, sza=None):
        fig, ax = plt.subplots(figsize=(14,10))

        if sza is not None:
            LUTcut = self.dataset.sel(sza=sza, method='nearest')
        else: 
            LUTcut = self.dataset.isel(sza=0)

        LUTcut = LUTcut.isel(phi=0, umu=0, ic_habit=0)

        LUTcut1 = LUTcut.sel(wvl=self.wvl1)
        LUTcut2 = LUTcut.sel(wvl=self.wvl2)

        for r_eff in LUTcut.coords['r_eff'].values:
            ax.plot(LUTcut1.sel(r_eff=r_eff).reflectivity.to_numpy(), 
                    LUTcut2.sel(r_eff=r_eff).reflectivity.to_numpy(),
                    linewidth=1, label=np.round(r_eff, 2))

        for itau550, tau550 in enumerate(LUTcut.coords['tau550'].values):
            ax.plot(LUTcut1.isel(tau550=itau550).reflectivity.to_numpy(), 
                    LUTcut2.isel(tau550=itau550).reflectivity.to_numpy(),
                    "--", color="black",
                    linewidth=0.7)
            
            x = np.max(LUTcut1.isel(tau550=itau550).reflectivity.to_numpy())
            y = np.max(LUTcut2.isel(tau550=itau550).reflectivity.to_numpy())+0.03
            eq = r"$\tau=$"
            if tau550<=2:
                plt.text(x,y, f"{eq}{tau550:.2f}", fontsize=11)
                
        wvl1 = LUTcut1.wvl.values
        wvl2 = LUTcut2.wvl.values
            
        ax.set_xlabel(f"Reflectivity at {wvl1}nm")
        ax.set_ylabel(f"Reflectivity at {wvl2}nm")
        ax.legend(title=r"Effective radius [$\mu$m]", ncols=3)
        plt.title("Nadir perspective")
        plt.tight_layout()
        plt.show()

    def display_at(self, sza, umu, phi, ic_habit):
        fig, ax = plt.subplots(figsize=(14,10))


        LUTcut = (LUTcut.isel(sza=sza, phi=phi, umu=umu, method='nearest')
                  .sel(ic_habit=ic_habit))

        LUTcut1 = LUTcut.sel(wvl=self.wvl1)
        LUTcut2 = LUTcut.sel(wvl=self.wvl2)

        for r_eff in LUTcut.coords['r_eff'].values:
            ax.plot(LUTcut1.sel(r_eff=r_eff).reflectivity.to_numpy(), 
                    LUTcut2.sel(r_eff=r_eff).reflectivity.to_numpy(),
                    linewidth=1, label=np.round(r_eff, 2))

        for itau550, tau550 in enumerate(LUTcut.coords['tau550'].values):
            ax.plot(LUTcut1.isel(tau550=itau550).reflectivity.to_numpy(), 
                    LUTcut2.isel(tau550=itau550).reflectivity.to_numpy(),
                    "--", color="black",
                    linewidth=0.7)
            
            x = np.max(LUTcut1.isel(tau550=itau550).reflectivity.to_numpy())
            y = np.max(LUTcut2.isel(tau550=itau550).reflectivity.to_numpy())+0.03
            eq = r"$\tau=$"
            if tau550<=2:
                plt.text(x,y, f"{eq}{tau550:.2f}", fontsize=11)
                
        wvl1 = LUTcut1.wvl.values
        wvl2 = LUTcut2.wvl.values
            
        ax.set_xlabel(f"Reflectivity at {wvl1}nm")
        ax.set_ylabel(f"Reflectivity at {wvl2}nm")
        ax.legend(title=r"Effective radius [$\mu$m]", ncols=3)
        plt.title("Nadir perspective")
        plt.tight_layout()
        plt.show()

    def display(self):
        return
    
    def reflectivity_range(self):
        Rone_min = self.dataset.sel(wvl=self.wvl1).reflectivity.min().values.item()
        Rone_max = self.dataset.sel(wvl=self.wvl1).reflectivity.max().values.item()

        Rtwo_min = self.dataset.sel(wvl=self.wvl2).reflectivity.min().values.item()
        Rtwo_max = self.dataset.sel(wvl=self.wvl2).reflectivity.max().values.item()

        return Rone_min, Rone_max, Rtwo_min, Rtwo_max
    

    def inverted(self, name1=None, name2=None, num=200, alpha=4.0,
                 interpolator=LinearInterpolator()):
        # Powered by LUTI 
        if name1 is None:
            name1 = self.Rone_name
        else: 
            self.Rone_name = name1
        if name2 is None:
            name2 = self.Rtwo_name
        else: 
            self.Rtwo_name = name2

        Rone_min, Rone_max, Rtwo_min, Rtwo_max = self.reflectivity_range()
        Rone = np.linspace(Rone_min, Rone_max, num=num)
        Rtwo = np.linspace(Rtwo_min, Rtwo_max, num=num)

        self.dataset.coords['wvl'] = ('wvl', [name1, name2])

        start_time = timer()
        inverted = invert_data_array(self.dataset.reflectivity,
                                     input_params=['r_eff', 'tau550'],
                                     output_dim='wvl',
                                     output_grid={name1:Rone, name2:Rtwo},
                                     checker=Alphachecker(alpha=alpha),
                                     interpolator=interpolator)
        end_time = timer()
        inversion_time = (end_time-start_time)/60. #[min]
        self.dataset.coords['wvl'] = ('wvl', [self.wvl1, self.wvl2])
        inverted.attrs = dict(inversion_time = f"{inversion_time} min")

        return inverted
        

"""The following tools are for handling of polarized data and simulations.
----------------------------------------------------------------------------"""

class PolLookupTable(object):
    """Takes mystic output Stokes parameters and standard deviations in xarray 
    dataset and provides calibration with srfs."""

    def __init__(self, polLUT):
        self.data = polLUT.copy()
        self.calibrated = False

    @classmethod
    def from_path(cls, path):
        data = read_LUT(path)
        return cls(data)
    
    def vza(self):
        return np.rad2deg(np.arccos(self.data.umu)).rename({'umu':'theta'})
    
    def over_theta_in_pp(self, sza, method='nearest'):
        """Returns dataset with scattering angle as coordinate, merged for all
        available umu with phi=0 (towards sun) and phi=180 (away from sun).
        TODO: implement a check for phi=0° and 180° grid points"""

        # theta = 180° - sza - vza
        towards_sun = self.data.sel(phi=0).sel(sza=sza, method=method)
        towards_sun = (towards_sun.assign_coords(umu=180-towards_sun.sza.values
                                                 -np.rad2deg(np.arccos(towards_sun.umu.values)))
                                  .rename({'umu':'theta'}).sortby('theta'))
        # theta = 180° - sza + vza
        away_from_sun = self.data.sel(phi=180).sel(sza=sza, method=method)
        away_from_sun = (away_from_sun.assign_coords(umu=180-away_from_sun.sza.values
                                                     +np.rad2deg(np.arccos(away_from_sun.umu.values)))
                                      .rename({'umu':'theta'}).sortby('theta'))

        return xr.concat([towards_sun, 
                          away_from_sun], dim='theta').sortby('theta')


    def polarized_reflectivity_with_srfs(self, calibration_file, color='red'):
        self.srfs = (calibration_file.srfs.sel(color=color).mean(dim='angle'))
        self.normalized_srfs = self.srfs/self.srfs.integrate(coord='wvl')

        self.normalized_srfs_interpolated = self.normalized_srfs.interp(wvl=self.data.wvl)
        self.calibrated_data = (self.data * self.normalized_srfs_interpolated).integrate(coord='wvl')

        # print((self.normalized_srfs.integrate(coord='wvl') 
        # - self.normalized_srfs_interpolated.integrate(coord='wvl'))/self.normalized_srfs.integrate(coord='wvl'))

        solar_flux = (solar_flux_kurudz().rename({'wavelength':'wvl'})
                      .interp(wvl=self.data.wvl.values))
        integrated_solar_flux = ((solar_flux*self.normalized_srfs_interpolated)
                                 .integrate(coord='wvl'))
        
        reflectivity = (np.pi * np.sqrt(self.calibrated_data.Q**2 
                                        + self.calibrated_data.U**2)
                        /(integrated_solar_flux 
                          *np.cos(np.deg2rad(self.data.sza)))).rename("simulated_reflectivity")
        
        return reflectivity

    
    def polarized_reflectivity(self):
        # calibrated=False gives refl. per wavelength

        solar_flux = (solar_flux_kurudz().rename({'wavelength':'wvl'})
                      .interp(wvl=self.data.wvl.values))
        
        reflectivity = (np.pi * np.sqrt(self.data.Q**2 + self.data.U**2)
                        /(solar_flux 
                          *np.cos(np.deg2rad(self.data.sza))))
        
        return reflectivity
    
    def DOLP(self):
        DOLP = (np.sqrt(self.data.Q**2 + self.data.U**2)/self.data.I).rename("DOLP")

        return DOLP
    
    def DOLP_with_srfs(self, calibration_file, color='red'):
        
        self.srfs = (calibration_file.srfs.sel(color=color).mean(dim='angle'))
        self.normalized_srfs = self.srfs/self.srfs.integrate(coord='wvl')

        self.normalized_srfs_interpolated = self.normalized_srfs.interp(wvl=self.data.wvl)
        self.calibrated_data = (self.data * self.normalized_srfs_interpolated).integrate(coord='wvl')

        DOLP = (np.sqrt(self.calibrated_data.Q**2 + self.calibrated_data.U**2)
                /self.calibrated_data.I).rename("simulated_DOLP")
        
        return DOLP


class PolSceneInterpreter(object):
    """Takes pol dataset as well as nas file for scene. Provides functions to 
    compute additional variables and interpret polarized data.
    """

    def __init__(self, camera_data, nas_data, solar_positions):
        self.camera_data = camera_data.copy()
        self.nas_data = nas_data.copy()
        self.solar_positions = solar_positions.copy().rename({'time':'time_img_binned'})

    def polarized_reflectivity(self, calibration_file):
        # relative to absolute solar irradiance
        srfs = (calibration_file.srfs.mean(dim='angle'))
        normalized_srfs = srfs/srfs.integrate(coord='wvl')

        solar_flux = (solar_flux_kurudz().rename({'wavelength':'wvl'})
                      .interp(wvl=normalized_srfs.wvl.values))
        
        calibrated_solar_flux = (solar_flux*normalized_srfs).integrate(coord='wvl')
        
        reflectivity = (np.pi * np.sqrt(self.camera_data.mean_Q**2 
                                        + self.camera_data.mean_U**2)
                        /(calibrated_solar_flux 
                          *np.cos(np.deg2rad(self.solar_positions
                                  .sza.mean())))).rename("reflectivity")
        return reflectivity
    
    def polarized_radiance(self):
    
        radiance = (np.sqrt(self.camera_data.mean_Q**2 
                                        + self.camera_data.mean_U**2).rename("radiance"))

        return radiance
    
    def DOLP(self):
        DOLP = (np.sqrt(self.camera_data.mean_Q**2 + self.camera_data.mean_U**2)/
                self.camera_data.mean_I).rename("DOLP")
        return DOLP
    
    def _find_SWIR_indices(self, sample, swir_coords):
        # Takes a sample and information about SWIR pixel coordinates and 
        # returns the indices of the closest fitting point
        distance = np.sqrt((swir_coords.lat 
                            - self.camera_data.sel(sample=sample).lat_cloud)**2
                            +(swir_coords.lon 
                              - self.camera_data.sel(sample=sample).lon_cloud)**2)
        return distance.argmin(dim=("x", "time"))
    
    def allocated_cloud_properties(self, cloud_properties, swir_coords, 
                                   nchunks=10):
        """Searches for closest fitting retrieval results and returns two 
        xarray datasets containing the right dimensions."""

        if (cloud_properties.time.size != swir_coords.time.size or 
            cloud_properties.x.size != swir_coords.x.size):
            warnings.warn(r"""Dimensions of swir coordinates and cloud properties do not match! This would cause false assignment of SWIR data since locations are assigned by index, not value. 
                          Coordinates will now be mapped onto cloud property grid.""")
            swir_coords = swir_coords.interp(time=cloud_properties.time,
                                             x=cloud_properties.x)
            
        temp_dir_path = tempfile.mkdtemp()

        sample_chunks = np.array_split(self.camera_data.sample.to_numpy(),
                                       nchunks)
        
        for chunk in sample_chunks:

            sample_min, sample_max = np.min(chunk), np.max(chunk)
            print(f"Find samples {sample_min}-{sample_max} of {self.camera_data.sample.size-1}...")
            r_eff_array = np.zeros((len(chunk), 
                                    cloud_properties.ic_habit.size))*np.nan
            tau550_array = np.copy(r_eff_array)

            for index, sample in enumerate(tqdm(chunk)):

                SWIR_indices = self._find_SWIR_indices(sample, swir_coords)

                r_eff_array[index,:] = (cloud_properties.r_eff
                                        .isel(SWIR_indices).values)  
                tau550_array[index,:] = (cloud_properties.tau550
                                        .isel(SWIR_indices).values)

            r_eff = xr.DataArray(data=r_eff_array, 
                                dims=['sample', 'ic_habit'], 
                                coords=dict(sample=chunk,
                                            ic_habit=cloud_properties.ic_habit)).rename("r_eff")
            save_as_netcdf(r_eff,
                           fr"{temp_dir_path}\r_eff_chunks_{sample_min}-{sample_max}.nc")
                                                                                    
            tau550 = xr.DataArray(data=tau550_array, 
                                dims=['sample', 'ic_habit'], 
                                coords=dict(sample=chunk,
                                            ic_habit=cloud_properties.ic_habit)).rename("tau550")     
            save_as_netcdf(tau550,
                           fr"{temp_dir_path}\tau550_chunks_{sample_min}-{sample_max}.nc")
        
        print("Load chunked results to memory and merge...")
        print("For r_eff...")
        r_eff_paths = os.path.join(temp_dir_path, 'r_eff_chunks_*.nc')
        r_eff_path_list = glob.glob(r_eff_paths)
        r_eff = xr.open_dataset(r_eff_path_list[0])
        for r_eff_path in tqdm(r_eff_path_list[1::]):
            r_eff = xr.concat([r_eff, xr.open_dataset(r_eff_path)], dim='sample')

        print("For tau550...")
        tau550_paths = os.path.join(temp_dir_path, 'tau550_chunks_*.nc')
        tau550_path_list = glob.glob(tau550_paths)
        tau550 = xr.open_dataset(tau550_path_list[0])
        for tau550_path in tqdm(tau550_path_list[1::]):
            tau550 = xr.concat([tau550, xr.open_dataset(tau550_path)], dim='sample')

        shutil.rmtree(temp_dir_path, ignore_errors=True)
        print("Done!")

        return r_eff.sortby('sample'), tau550.sortby('sample')
    
    def allocated_swir_coordinates(self, swir_coords):
        """Searches for closest fitting retrieval results and returns two 
        xarray datasets containing the right dimensions."""

        time_array = []
        x_array = []

        print("Number of iterations: ", self.camera_data.sample.size)
        for index, sample in tqdm(enumerate(self.camera_data.sample)):
            SWIR_indices = self._find_SWIR_indices(sample, swir_coords)
            try:
                time = swir_coords.isel(SWIR_indices).time.values
                time_array.append(time)
                x_array.append(swir_coords.isel(SWIR_indices).x.values)
            except:
                continue

        time = xr.DataArray(data=time_array, 
                            dims=['sample'], 
                            coords=dict(sample=self.camera_data.sample)).rename("swir_time")
                                                                                
        x = xr.DataArray(data=x_array, 
                         dims=['sample'], 
                         coords=dict(sample=self.camera_data.sample)).rename("swir_x")
                                                                        
        return time, x
    
    def halo_positions(self):
        """Returns datasets containing halo position for each scattering angle."""

        lon_halo = (self.nas_data.lon
                    .interp(time=self.camera_data.time_img_binned)
                    .drop_vars('time')).rename("lon_halo")
        lat_halo = (self.nas_data.lat
                    .interp(time=self.camera_data.time_img_binned)
                    .drop_vars('time')).rename("lat_halo")
        height_halo = (self.nas_data.height
                       .interp(time=self.camera_data.time_img_binned)
                       .drop_vars('time')).rename("height_halo")

        return lon_halo, lat_halo, height_halo
    
    def _get_local_cartesian_vector(self, coord_tuple, utm_zone):
        lon, lat, height = coord_tuple
        wgs84 = pyproj.CRS("EPSG:4326")
        target_crs = pyproj.CRS(f"EPSG:326{utm_zone}")
        transformer = pyproj.Transformer.from_crs(wgs84, target_crs, 
                                                  always_xy=True, accuracy=1.)
        
        utm_easting, utm_northing, height = transformer.transform(lon, lat, 
                                                                  height)
        
        return np.array([utm_easting, utm_northing, height])
    
    def cartesian_cloud_positions(self, utm_zone=33):
        cloud_coords = (self.camera_data.lon_cloud, 
                        self.camera_data.lat_cloud,
                        self.camera_data.height_cloud)
        #shape (3, sample)
        position_cloud = self._get_local_cartesian_vector(cloud_coords, utm_zone)

        position_cloud_da = xr.Dataset({"easting":(("sample"),
                                                   position_cloud[0]),
                                       "northing":(("sample"), 
                                                   position_cloud[1]),
                                       "height":(("sample"), 
                                                 position_cloud[2])},
                                       coords={"sample":self.camera_data.sample})
        
        return position_cloud_da
    
    def cartesian_halo_positions(self, utm_zone=33):
        # shape (3, sample, theta_mean)
        position_halo = self._get_local_cartesian_vector(self.halo_positions(), 
                                                         utm_zone)
        
        position_halo_da = xr.Dataset({"easting":(("sample", "theta_mean"),
                                                   position_halo[0]),
                                       "northing":(("sample", "theta_mean"), 
                                                   position_halo[1]),
                                       "height":(("sample", "theta_mean"), 
                                                 position_halo[2])},
                                       coords={"sample":self.camera_data.sample,
                                               "theta_mean":self.camera_data.theta_mean})
    
        return position_halo_da
    
    def halo_cloud_distance(self, utm_zone=33):
        position_halo_da = self.cartesian_halo_positions(utm_zone=utm_zone)
        position_cloud_da = self.cartesian_cloud_positions(utm_zone=utm_zone)
        diff = position_halo_da - position_cloud_da

        norm = np.sqrt(np.square(diff.easting)+np.square(diff.northing)
                       +np.square(diff.height)).rename("halo_cloud_distance")

        return norm
    
    def absolute_view_angles(self, utm_zone=33):
        # pointing from cloud to halo
        diff = (self.cartesian_halo_positions(utm_zone=utm_zone) 
                - self.cartesian_cloud_positions(utm_zone=utm_zone))

        ground_distance = np.sqrt(np.square(diff.easting)+np.square(diff.northing))
        
        # shape (sample, theta)
        vza = np.rad2deg(np.arctan2(ground_distance, diff.height)).rename("vza")
        vaa = ((90 - np.rad2deg(np.arctan2(diff.easting, diff.northing))+360)%360).rename("vaa")

        return vza, vaa
    
    def relative_view_angles(self, utm_zone=33):
        vza, vaa = self.absolute_view_angles(utm_zone=utm_zone)
        vza = np.deg2rad(vza)

        umu = np.cos(vza).rename("umu")
        phi = ((vaa - self.solar_positions.saa.mean())%360).rename("phi")

        return umu, phi
    