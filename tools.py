import numpy as np
import matplotlib.pyplot as plt
import xarray as xr 
from .conveniences import load_solar_flux_kurudz


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

    def get_phi_variable(self):
        # Relative solar azimuth as to be passed to uvspec
        return (self.view_angles.vaa - self.solar_positions.saa.mean())%360

    def get_umu_variable(self):
        # As to be passed to uvspec
        return np.cos(2.*np.pi*self.view_angles.vza/360.)
    
    def get_reflectivity_variable(self):
        """Returns an additional variable to calibrated SWIR or VNIR data, 
        based on a nas file containing halo data and the radiance measurements, 
        that can be merged with the other datsets. Now also considers Earth-Sun 
        distance."""
        
        print("Load solar flux...")
        solar_flux = (load_solar_flux_kurudz()
                      .interp(wavelength=measurement.wavelength.values))
        
        print("Compute reflectivities...")
        reflectivity = self.camera_data["radiance"]*(self.solar_positions.d**2)*\
        np.pi/(solar_flux*np.cos(2.*np.pi*self.solar_positions.sza/360.))
        
        return reflectivity
    
    def get_scene_overview(self):
        return

    
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


