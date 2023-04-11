from .background_tools import *
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from jinja2 import Template, StrictUndefined
import os
import xarray as xr 
import netCDF4 as nc
from timeit import default_timer as timer
import tempfile
import shutil
import itertools as it
from multiprocessing import Pool
import sys
# Luti path
sys.path.insert(0, os.path.abspath('/project/meteo/work/Dennys.Erdtmann/Thesis/Python Packages/luti'))
import luti
from luti.xarray import invert_data_array
from luti import Alphachecker, NeighbourInterpolator
import pvlib

def add_reflectivity_variable(measurement, nas_file):
    """Adds an additional variable to calibrated SWIR or VNIR data, based on a nas file containing halo data and the radiance 
    measurements."""
    
    solar_positions = get_solar_positions(nas_file)
    solar_positions_resampled = solar_positions.interp(time=measurement.time.values)
        
    solar_flux = load_solar_flux_kurudz()
    solar_flux_resampled = solar_flux.interp(wavelength=measurement.wavelength.values)
    
    measurement["reflectivity"] = measurement["radiance"]*np.pi/(solar_flux_resampled* \
                                                                np.cos(2.*np.pi*solar_positions_resampled.zenith/360.))
    
    return


def get_solar_positions(nas_file, assumed_cloud_height=0.):
    solar_positions = pvlib.solarposition.get_solarposition(nas_file.time, nas_file.lon, nas_file.lat,
                                                             altitude=assumed_cloud_height)

    solar_positions = solar_positions.to_xarray()
    solar_positions = solar_positions.rename({'index':'time'})
    
    return solar_positions


def find_nearest_gridpoint(coordinate_array, value):
    """Takes the coordinate array of an xarray dataarray (of the form da.coordinate) and returns the index matching the closest 
    gridpoint to value.""" 
    array = np.asarray(coordinate_array.values)
    index = (np.abs(array - value)).argmin()
    return index

def read_LUT(LUTpath, rename = True):
    
    LUT = nc.Dataset(LUTpath)
    if rename:
        LUT = xr.open_dataset(xr.backends.NetCDF4DataStore(LUT)).rename({"__xarray_dataarray_variable__" : "reflectivity"})
    else:
        LUT = xr.open_dataset(xr.backends.NetCDF4DataStore(LUT))
    
    return LUT


def show_LUT(LUT, wvl1, wvl2, save_under=None):
    """Takes a LUT for one perspective and returns an overview of plots to display the data. Does not retrieve any values."""
    
    fig, ax = plt.subplots(figsize=(16,16))
    LUTcut1 = LUT.sel(wvl=wvl2)
    LUTcut2 = LUT.sel(wvl=wvl1)

    for r_eff in LUT.coords['r_eff'].values:
        ax.plot(LUTcut1.sel(r_eff=r_eff).reflectivity.to_numpy(), LUTcut2.sel(r_eff=r_eff).reflectivity.to_numpy(),
                linewidth=1, label=np.round(r_eff, 2))

    for tau550 in LUT.coords['tau550'].values:
        ax.plot(LUTcut1.sel(tau550=tau550).reflectivity.to_numpy(), LUTcut2.sel(tau550=tau550).reflectivity.to_numpy(),
                "--", color="black",
                linewidth=0.7)
        
        x = np.max(LUTcut1.sel(tau550=tau550).reflectivity.to_numpy())
        y = np.max(LUTcut2.sel(tau550=tau550).reflectivity.to_numpy())
        #if tau550<15 and tau550>0:
         #   plt.text(x,y, r"$\tau=$"+str(tau550), fontsize=11)

    ax.set_xlabel("Reflectivity at "+str(wvl1)+"nm")
    ax.set_ylabel("Reflectivity at "+str(wvl2)+"nm")
    ax.legend(title=r"Effective radius [$\mu$m]", ncols=3)
    
    if save_under is not None:
        plt.savefig(save_under)
        
    plt.show()
    return
        

def get_retrieval_stats(LUT, measuredLUT, wvl1, wvl2, display=True, savefig=None):
    """Takes a reference LUT and measured values and returns the retrieval accuracy.
    Coordinates have to be already cut to perspective, i.e phi and umu as well as sza have to be chosen before handing the 
    files to the function. . """
    
    reflectivity_array, cloud_param_array = get_measurement_arrays(measuredLUT, wvl1, wvl2)
    
    # Format LUT to pass to interpolation function. 
    LUTcut = LUT.sel(wvl=[wvl1,wvl2])
    LUTcut.coords["wvl"]=("wvl",["Rone", "Rtwo"])
    LUTcut.drop(["phi","umu","sza"])
    
    # Pass measured values to interpolator as grid points and get inverted LUT
    measured_R1, measured_R2 = reflectivity_array[:,0], reflectivity_array[:,1]
    inverted = invert_data_array(LUTcut.reflectivity, input_params=["r_eff", "tau550"], output_dim="wvl", 
                                 output_grid={"Rone":measured_R1, "Rtwo":measured_R2})
    inverted.name="inverted parameters"
    
    # Compute differences between retrieved values and measurement
    errors = np.zeros((len(reflectivity_array), 2))
    diffs = np.zeros((len(reflectivity_array), 2))
    for line in range(len(reflectivity_array)):
        diffs[line, 0]=inverted.sel(input_params="r_eff", 
                                    Rone=measured_R1[line], Rtwo=measured_R2[line]).values - cloud_param_array[line,0]
        diffs[line, 1]=inverted.sel(input_params="tau550", 
                                    Rone=measured_R1[line], Rtwo=measured_R2[line]).values - cloud_param_array[line,1]
        errors[line, 0]=diffs[line,0]**2
        errors[line, 1]=diffs[line,1]**2
    r_eff_errors, tau550_errors = [error for error in errors[:,0] if not np.isnan(error)], \
    [error for error in errors[:,1] if not np.isnan(error)]
    r_eff_diffs, tau550_diffs = [diff for diff in diffs[:,0] if not np.isnan(diff)], \
    [diff for diff in diffs[:,1] if not np.isnan(diff)]
    
    nfails = len(reflectivity_array)-len(errors)
    npoints = len(reflectivity_array)
    mean_r_eff_diff = np.mean(r_eff_diffs)
    mean_tau550_diff = np.mean(tau550_diffs)
    r_eff_RMS = np.sqrt(np.mean(r_eff_errors))
    tau550_RMS = np.sqrt(np.mean(tau550_errors))
    
    if display:
        
        # Plot inverted LUT (isert here as well a plot including the positions of the measured values...)
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        
        LUTcut1 = LUT.sel(wvl=wvl2)
        LUTcut2 = LUT.sel(wvl=wvl1)
        
        ax = axis[0]
        for r_eff in LUTcut.coords['r_eff'].values:
            ax.plot(LUTcut1.sel(r_eff=r_eff).reflectivity.to_numpy(), LUTcut2.sel(r_eff=r_eff).reflectivity.to_numpy(),
                      linewidth=0.7, label=np.round(r_eff))

        for itau550 in range(len(LUTcut.coords['tau550'])):
            ax.plot(LUTcut1.isel(tau550=itau550).reflectivity.to_numpy(),
                      LUTcut2.isel(tau550=itau550).reflectivity.to_numpy(),
                      "-.", color="black", linewidth=0.7)
        ax.set_xlabel("Reflectivity at "+str(wvl2)+"nm")
        ax.set_ylabel("Reflectivity at "+str(wvl1)+"nm")
        
        ax = axis[1]
        for r_eff in LUTcut.coords['r_eff'].values:
            ax.plot(LUTcut1.sel(r_eff=r_eff).reflectivity.to_numpy(), LUTcut2.sel(r_eff=r_eff).reflectivity.to_numpy(),
                      linewidth=0.7, label=np.round(r_eff))

        for itau550 in range(len(LUTcut.coords['tau550'])):
            ax.plot(LUTcut1.isel(tau550=itau550).reflectivity.to_numpy(),
                      LUTcut2.isel(tau550=itau550).reflectivity.to_numpy(),
                      "-.", color="black", linewidth=0.7)

        ax.scatter(measured_R2, measured_R1, marker='X', color='red')
        ax.set_xlabel("Reflectivity at "+str(wvl2)+"nm")
        ax.set_ylabel("Reflectivity at "+str(wvl1)+"nm")
        #ax.legend(title=r"Effective radius [$\mu$m]", ncol=2)
        plt.show()
        
        fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        inverted.sel(input_params="r_eff").plot(ax=axis[0])
        axis[0].set_title("Retrieved effective radius")
        inverted.sel(input_params="tau550").plot(ax=axis[1])
        axis[1].set_title("Retrieved optical thickness")
        
        if savefig is not None:
            plt.savefig(savefig, dpi=500)
            print("Image saved under", savefig)
        plt.show()
        
        print("Number of retrieved values:           ", npoints)
        print("Number of failed retrieval attempts: ", nfails)
        print("Mean difference effective radius: ", np.round(mean_r_eff_diff, 4), 
              " | Mean difference optical thickness: ", np.round(mean_tau550_diff, 4))
        print("RMS effective radius: ", np.round(r_eff_RMS, 4), 
              " | RMS optical thickness: ", np.round(tau550_RMS, 4))
        
        return
    
    else:
        
        return nfails, npoints, mean_r_eff_diff, mean_tau550_diff, r_eff_RMS, tau550_RMS
    
    
def sanity_check(LUTcut, wvl1, wvl2):
    get_retrieval_stats(LUTcut, LUTcut, wvl1, wvl2)
    return
