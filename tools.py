import numpy as np
import matplotlib.pyplot as plt
import xarray as xr 
from timeit import default_timer as timer

from luti.xarray import invert_data_array
from luti import Alphachecker
from luti import LinearInterpolator

from .retrieval_functions import fast_retrieve
from .conveniences import read_LUT
from .paths import LIBRADTRAN_PATH


class PixelInterpolator(object):
    """The PixelInterpolator is supposed to replace the runmacs BadPixelFixer 
    functionality. Since A(C)³ scenes are typically dark with low sun angle,
    some pixels are unreliable, even though they might work for a different 
    scene. You can choose to interpolate invalid pixels, according to the 
    'valid' variable, or dynamically find unreliable pixel rows. data should
    contain a 'radiance' variable."""

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

    def filtered_radiance(self, with_bpl=True):
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
        # Relative solar azimuth as to be passed to uvspec
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
    
    def overview(self):
        umu = self.umu()
        phi = self.umu()

        sza_mean = self.solar_positions.sza.mean().values
        saa_mean = self.solar_positions.saa.mean().values

        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14,10), sharex=True)

        umu.plot(ax=ax[0,0], x='time')
        umu.plot.contour(ax=ax[0,0], x='time', cmap='coolwarm')
        phi.plot(ax=ax[0,1], x='time')
        phi.plot.contour(ax=ax[0,1], x='time', cmap='coolwarm')
        
        self.solar_positions.sza.plot(ax=ax[1,0])
        ax[1,0].axhline(sza_mean, color='red', linestyle='dotted', 
                      label=f"Mean solar zenith angle = {sza_mean}")
        ax[1,0].legend()
        self.solar_positions.saa.plot(ax=ax[1,1])
        ax[1,1].axhline(saa_mean, color='red', linestyle='dotted', 
                      label=f"Mean solar azimuth angle = {saa_mean}")
        ax[1,1].legend()
        plt.tight_layout()
        plt.show()

        umu_min, umu_max = umu.min().values, umu.max().values
        phi_min, phi_max = phi.min().values, phi.max().values

        fig, ax = plt.subplots(ncols=2, figsize=(14,5))

        umu.plot.hist(bins=150, ax=ax[0], 
                      label=f"umu PDF with min={umu_min:.2f} and max={umu_max:.2f}", 
                      density=True)
        ax[0].legend()
        phi.plot.hist(bins=150, ax=ax[1], 
                      label=f"phi PDF with min={phi_min:.2f} and max={phi_max:.2f}", 
                      density=True)
        ax[1].legend()
        plt.tight_layout()
        plt.show()

        return sza_mean, saa_mean, umu_min, umu_max, phi_min, phi_max
    
    def cloud_pixels(self, wavelenth=1550, threshold=3.):

        return

    def ice_index_Ehrlich2008(measurement, center_wvl, wvl_lower, wvl_upper):
        """Returns spectral ice index following the equation 
        by Ehrlich et al. 2008
        EXPERIMENTAL"""
        
        measurement=measurement.reflectivity
        
        R_diff = measurement.sel(wavelength=wvl_upper, method='nearest')\
        -measurement.sel(wavelength=wvl_lower, method='nearest')
        wvl_diff = wvl_upper-wvl_lower
        
        # To be extended to use linear regression
        I_s = (R_diff/measurement.sel(wavelength=center_wvl, method='nearest'))\
        *(100./wvl_diff)
        
        return I_s.rename('ice_index')
    
    def ice_index_Jaekel2013(self):

        return
    
    def ice_index_Thompson2016(self):

        return 
    
    def merged_data(self):

        merged_data = xr.merge([self.camera_data, self.umu(), self.phi(),
                                self.reflectivity(),
                                self.solar_positions, self.view_angles])
        
        return merged_data
    
    def cloud_properties_fast_BSR(self, invertedLUT, wvl1, wvl2, R1_name, R2_name,
                                  where_cloud=False,
                                  umu_bins=15,
                                  phi_bins=40):
        """Takes an inverted LUT, i.e. with r_eff and tau550 variables, and 
        resamples according to SWIR pixels of approximate equal geometry. 
        Returns r_eff, tau550 as tuple."""
                            # From retrieval_functions.py
        cloud_properties = fast_retrieve(invertedLUT, self.merged_data(),
                                         wvl1, wvl2, R1_name, R2_name,
                                         umu_bins=umu_bins, phi_bins=phi_bins)
        return cloud_properties
    

class BSRLookupTable(object):

    def __init__(self, LUT):
        if LUT.wvl.size != 2:
            raise Exception("Please choose two wavelengths.")
        self.dataset = LUT.copy()
        self.wvl1 = self.dataset.isel(wvl=0).wvl.values.item()
        self.wvl2 = self.dataset.isel(wvl=1).wvl.values.item()

    @classmethod
    def from_path(cls, path):
        LUT = read_LUT(path)

        return cls(LUT)

    def display_nadir(self):
        fig, ax = plt.subplots(figsize=(14,10))

        LUTcut = self.dataset.isel(phi=0, umu=0, sza=0, ic_habit=0)

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
            name1 = f"R({self.wvl1}nm)"
        if name2 is None:
            name2 = f"R({self.wvl2}nm)"

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
        

class PolLookupTable(object):
    """Takes mystic out Stokes parameters and standard deviations in xarray 
    dataset and provides calibration with srfs.
    
    TODO: Make sure use is easy. Define function, 
    pol_cam_reflectivity(sself, calibration_file)"""

    def __init__(self, polLUT):
        self.data = polLUT.copy()
        self.calibrated = False

    def calibrate(self, calibration_file, color='red'):
        # Calibrated refres to simulating pol camera signal
        self.srfs = (calibration_file.srfs.interp(wvl=self.data.wvl)
                     .sel(color=color).mean(dim='angle'))
        self.normalized_srfs = self.srfs/self.srfs.sum(dim='wvl')
        self.scaled_data = self.data * self.normalized_srfs

        self.calibrated_data = self.scaled_data.sum(dim='wvl')
        self.calibrated = True

        return 
    
    def polarized_reflectivity(self, calibrated=False):
        # calibrated=False gives refl. per wavelength
        if calibrated and not self.calibrated:
            raise Exception("Calibrate data and try again")

        solar_flux = (solar_flux_kurudz().rename({'wavelength':'wvl'})
                      .interp(wvl=self.data.wvl.values))
        
        if calibrated: 
            data = self.calibrated_data
            # Scale and integrate solar flux, similar to Stokes params, before
            # computing reflectivity
            scaled_solar_flux = solar_flux * self.normalized_srfs
            solar_flux = scaled_solar_flux.sum(dim='wvl')

        if not calibrated:
            data = self.data
        
        # If calibrated, does not depend on wavelength anymore
        reflectivity = (np.pi * np.sqrt(data.Q**2 + data.U**2)
                        /(solar_flux 
                          *np.cos(2.*np.pi*data.sza/360.)))
        
        return reflectivity
    

        

