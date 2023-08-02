import numpy as np
import matplotlib.pyplot as plt
import xarray as xr 
from .conveniences import load_solar_flux_kurudz
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
        if len(cutoffs) != len(self.data.wavelength):
            raise Exception("""Number of passed cutoffs does not equal 
                            number of wavelengths in data""")
        self.cutoffs = cutoffs
        cutoff_da = xr.DataArray(data=self.cutoffs, dims="wavelength",
                                 coords={"wavelength":self.data\
                                         .wavelength.values})
        self.data = self.data.assign(cutoff=cutoff_da)

    def apply_bpl(self):
        self.data['radiance'] = self.data.radiance.\
                                where(self.data.valid==1).\
                                interpolate_na(dim='x', method='spline', 
                                               use_coordinate=False)
        self.interpolated_from_list = True

    def apply_filter(self):
        if self.cutoffs is None:
            raise Exception("""Enter cutoff list with 'add_cutoffs()'
                            before applying filter""")
        
        selection = self.data.radiance\
                    .where(self.data.int_slopes_mva<self.data.cutoff)
        
        interpolated_radiance = selection.interpolate_na(dim='x', 
                                                         method='spline', 
                                                         use_coordinate=False)
        self.data['radiance'] = interpolated_radiance
        self.applied_filter = True

    def get_filtered_radiance(self, with_bpl=True):
        self.apply_bpl()
        self.apply_filter()
        return self.data.radiance
    
    
def load_solar_flux_kurudz():
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
                            .interp(time=self.camera_data.time.values))
        self.solar_positions = (solar_positions.copy()
                                .interp(time=self.camera_data.time.values))
        
    def get_umu_variable(self):
        # As to be passed to uvspec
        umu = np.cos(2.*np.pi*self.view_angles.vza/360.)
        return umu.rename('umu')

    def get_phi_variable(self):
        # Relative solar azimuth as to be passed to uvspec
        phi = (self.view_angles.vaa - self.solar_positions.saa.mean())%360
        return phi.rename('phi')
    
    def get_reflectivity_variable(self):
        """Returns an additional variable to calibrated SWIR or VNIR data, 
        based on a nas file containing halo data and the radiance measurements, 
        that can be merged with the other datsets. Now also considers Earth-Sun 
        distance."""
        
        print("Load solar flux...")
        solar_flux = (load_solar_flux_kurudz()
                      .interp(wavelength=self.camera_data.wavelength.values))
        
        print("Compute reflectivities...")
        reflectivity = (self.camera_data["radiance"]
                        *(self.solar_positions.d**2)
                        *np.pi
                        /(solar_flux*np.cos(2.*np.pi
                                            *self.solar_positions.sza/360.)))
        
        return reflectivity
    
    def get_scene_overview(self):
        umu = self.get_umu_variable()
        phi = self.get_phi_variable()

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

    
    def get_ice_index(measurement, center_wvl, wvl_lower, wvl_upper):
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


class polLutInterpreter(object):
    """Takes mystic out Stokes parameters and standard deviations in xarray 
    dataset and provides calibration with srfs."""

    def __init__(self, polLUT):
        self.data = polLUT.copy()
        self.calibrated = False

    def calibrate(self, calibration_file, color='red'):

        self.srfs = (calibration_file.srfs.interp(wvl=self.data.wvl)
                     .sel(color=color).mean(dim='angle'))
        self.normalized_srfs = self.srfs/self.srfs.sum(dim='wvl'
                                                       )
        self.scaled_data = self.data * self.normalized_srfs

        self.calibrated_data = self.scaled_data.sum(dim='wvl')
        self.calibrated = True

        return 
    
    def get_polarized_reflectivity(self, calibrated=False):

        if calibrated and not self.calibrated:
            raise Exception("Please calibrate data and try again")

        print("Load solar flux...")
        solar_flux = (load_solar_flux_kurudz().rename({'wavelength':'wvl'})
                      .interp(wvl=self.data.wvl.values))
        
        if calibrated: 
            data = self.calibrated_data

            scaled_solar_flux = solar_flux * self.normalized_srfs
            solar_flux = scaled_solar_flux.sum(dim='wvl')

        if not calibrated:
            data = self.data

        print(data)
        
        reflectivity = (np.pi * np.sqrt(data.Q**2 + data.U**2)
                        /(solar_flux 
                          *np.cos(2.*np.pi*data.sza/360.)))
        
        return reflectivity
    

        

