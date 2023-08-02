"""Should contain functions that are not crucial to the work flow of iceMACS 
but make handling data files more convenient, such as reading and writing 
netCDF file, plotting data etc."""

import numpy as np
import os
import netCDF4 as nc
import xarray as xr
import datetime
import glob
import macsproc
import macstrace
from macstrace.Shapes import ZPlane
import pvlib.solarposition as sp
import cartopy.crs as ccrs
import cartopy
import matplotlib as plt
from .paths import MOUNTTREE_FILE_PATH


def save_as_netcdf(xr_data, fpath):
    """Avoids permission issues with xarray not overwriting existing files."""
    try: 
        os.remove(fpath)
    except:
        pass
    
    xr_data.to_netcdf(fpath)
    return


def read_LUT(LUTpath):
    LUT = nc.Dataset(LUTpath)
    LUT = xr.open_dataset(xr.backends.NetCDF4DataStore(LUT))
    
    return LUT


def load_AC3_scene(start_time, end_time, swir=True, vnir=True, bahamas=True):
    """Takes specific time in string format and returns nc files as xarray 
    datasets: 
    vnir_scene, swir_scene, bahamas_data

    Camera files with variables:
    'radiance', 'alt', 'act', 'valid'

    All unnecessary variables in camera files are excluded. 
    Your can exclude each by setting the kwarg False."""
    
    day = start_time.strftime("%Y%m%d")
    interval = (start_time.strftime("%Y-%m-%dT%H:%M:%S"), 
                end_time.strftime("%Y-%m-%dT%H:%M:%S"))
    
    if day != end_time.strftime("%Y%m%d"):
        raise Exception("""Interval Error: start and end time have to be on the
                        same day!""")
    
    # Format slightly larger time frame for nas file, in order to avoid overlap
    nas_start_time = start_time - datetime.timedelta(seconds=1)
    nas_end_time = end_time + datetime.timedelta(seconds=1)
    nas_interval = (nas_start_time.strftime("%Y-%m-%dT%H:%M:%S"), 
                    nas_end_time.strftime("%Y-%m-%dT%H:%M:%S"))
    
    source_folder='/archive/meteo/ac3/'+day+'/'

    if bahamas:
        print("Load bahamas data...")
        nas_day_path = (source_folder+"nas/AC3_HALO_BAHAMAS-SPECMACS-100Hz-final_"
                        +day+"a.nc")
        nas_day = read_LUT(nas_day_path)
        nas_scene = nas_day.sel(time=slice(*nas_interval))
        del(nas_day)
    else:
        nas_scene = None
    
    if swir:
        print("Load and calibrate SWIR data...")
        files = sorted(glob.glob(os.path.join(source_folder, 
                                              "raw/specMACS.swir.*.flatds")))
        auxdata = glob.glob(os.path.join(source_folder, 
                                         "auxdata/swir_"+day+"*.log"))
        swir_cal = """/archive/meteo/ac3/specmacs_calibration_data/specMACS_SWIR_cal_AC3.nc"""
        swir_day = macsproc.load_measurement_files(files, auxdata, swir_cal, 
                                                   "swir")
        swir_scene = swir_day.sel(time=slice(*interval))[["radiance", 
                                                          "alt", "act", 
                                                          "valid"]]
        del(swir_day)
    else:
        swir_scene = None

    if vnir:
        print("Load and calibrate VNIR data...")
        files = sorted(glob.glob(os.path.join(source_folder, 
                                              "raw/specMACS.vnir.*.flatds")))
        auxdata = glob.glob(os.path.join(source_folder, 
                                         "auxdata/vnir_"+day+"*.log"))
        # Same calibration as during EUREC4A but dark current LUT necessary
        vnir_cal = """/archive/meteo/eurec4a/specmacs_calibration_data/
        specMACS_VNIR_cal_preEUREC4A_temp.nc"""
        dark_current_LUT_path = """/project/meteo/work/Veronika.Poertge/PhD/data/specmacs/vnir/averaged_dark_current_vnir_20200202.nc"""
        vnir_day = macsproc.load_measurement_files_dark_current_LUT(files,
                                                                    auxdata,
                                                                    vnir_cal,
                                                                    "vnir", 
                                                                    dark_current_LUT_path)
        vnir_scene = vnir_day.sel(time=slice(*interval))[["radiance", 
                                                          "alt", "act", 
                                                          "valid"]]
        del(vnir_day)
    else:
        vnir_scene = None
        
    return nas_scene, swir_scene, vnir_scene


def map_AC3_scene(start_time, end_time):
    
    day = start_time.strftime("%Y%m%d")
    interval = (start_time.strftime("%Y-%m-%dT%H:%M:%S"), 
                end_time.strftime("%Y-%m-%dT%H:%M:%S"))
    
    if day != end_time.strftime("%Y%m%d"):
        print("Error: Start and end time have to be on the same day!")

    print("Load bahamas data...")
    source_folder='/archive/meteo/ac3/'+day+'/'
    nas_day_path = source_folder+"nas/AC3_HALO_BAHAMAS-SPECMACS-100Hz-final_"+day+"a.nc"
    nas_day = read_LUT(nas_day_path)
    nas_scene = nas_day.sel(time=slice(*interval))
    
    nas_scene = nas_day.sel(time=slice(*interval))

    day_trajectory = nas_day.lon.values, nas_day.lat.values
    scene_trajectory = nas_scene.lon.values, nas_scene.lat.values

    fig, ax = plt.subplots(nrows=1,ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            figsize=(16,6))

    # Remove frame 
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False,
                    left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    #for ax in axes:

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, alpha=0.4)
    ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.6)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':',linewidth=0.3)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    ax.plot(*day_trajectory,
             color='tab:orange', linestyle='--', linewidth=0.6,
             transform=ccrs.PlateCarree())

    ax.plot(*scene_trajectory,
            color='red', linewidth=2.5,
            transform=ccrs.PlateCarree())
 
    return 


def get_solar_positions(nas_file, mounttree_file_path=MOUNTTREE_FILE_PATH):
    """Returns all relevant information about Sun-Earth distance and solar 
    position in order to compute cloud reflectivity information and get
    relevant ranges for LUT simulation."""
  
    sun = macstrace.Ephemeris.Sun.from_datasets(mounttree_file_path, nas_file)
    solar_positions = sun.get_sza_az(time=nas_file.time.values)
    
    d = (sp.nrel_earthsun_distance(solar_positions.time).to_xarray()
         .rename({'index':'time'}))
    
    solar_positions['d'] = d
    
    return solar_positions


def correct_sensor_AC3(nas_scene_file_path, 
                       reference_sensor, corrected_sensor,
                       cloud_top_height = 0,
                       mounttree_file_path=MOUNTTREE_FILE_PATH):
    """Maps the specified sensor onto the other camera perspective. A cloud 
    height is assumed to perform the raytracing in macstrace, but not crucial 
    to the tracing accuracy."""
    
    cloud_plane = ZPlane(height=-cloud_top_height)

    sensor_corrected = macstrace.smacs1_to_smacs2(mounttree_file_path, 
                                              nas_scene_file_path, 
                                              corrected_sensor, 
                                              reference_sensor, 
                                              cloud_plane)

    sensor_corrected = sensor_corrected.isel(wvl_2=0).to_dataset()
    
    return sensor_corrected


def format_corrected_data(vnir_scene, swir_scene, swir_corrected):
    """Formats two-camera data after being perspective corrected, such that 
    only pixels are included that both cameras see. This way no nans are 
    generated. Returns a single xarray dataset."""
    
    measurements = xr.merge([vnir_scene, swir_corrected])
    x = swir_scene.x.values
    x = xr.where(abs(x)<abs(vnir_scene.x.max().values), x, np.nan)
    x = x[~np.isnan(x)]
    measurements = measurements.interp(x = x)

    return measurements

"""
class BSRLookupTable(object):

    def __init__(self, LUT):
        self.LUT = LUT.copy()
        self.

    def """