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


def get_ic_reflectivity(args):
    
    """Returns the uvspec output for a single ice cloud. Input params are passed as a single zipped argument and listed below. 
    Function is currently being called by a map function in 'writeLUT', such that 'cloud_property_indices' is iterable. This
    allows for parallel computing of the LUT. wc and ic cloud function so far are not very different, only the height of the 
    cloud is different and the template file is adapted to call the ice parameterizations in libradtran. Optical properties and 
    computation are so far identical.
    """
        
    cloud_property_indices, wvl_array, phi_array, umu_array, isza,\
    sza_array, r_eff_array, tau550_array, phi0, cloud_top_distance, wvl_grid_file_path, ic_habit, ic_properties = args
        
    ir_eff, itau550 = cloud_property_indices
    
    r_eff = r_eff_array[ir_eff]
    tau550 = tau550_array[itau550]
        
    temp_dir_path = tempfile.mkdtemp()
        
    cloud_file_path = temp_dir_path+'temp_cloud_file.dat'
    generated_input_file_path = temp_dir_path+'temp_input.inp'
        
    if len(r_eff_array) is not len(tau550_array):
        print("Cloud property arrays must have the same shape!")
        return
        
    # Format spherical coordinate arrays to make compatible with template
    phi_str = ' '.join([str(p) for p in phi_array])
    umu_str = ' '.join([str(u) for u in umu_array])
    
    # Define cloud structure
    altitude_grid = np.array([7, 8, 9, 10])
    WC_array = np.array([0.1, 0.1, 0.1, 0.1])
    r_eff_array = r_eff*np.array([1, 1, 1, 1])
    
    # Generate ic file with corresponding values
    write_cloud_file(cloud_file_path, altitude_grid, WC_array, r_eff_array)
    cloud_file = np.loadtxt(cloud_file_path)
    cloud_top = np.max(cloud_file[:,0])
    
    # Get uvspec output
    input_file_args = {
        "wvl_LowerLimit"            : "inactive",
        "wvl_UpperLimit"            : "inactive",
        "wavelength_grid_file_path" : wvl_grid_file_path,
        "sza"                       : sza_array[isza],
        "phi0"                      : phi0,
        "umu"                       : umu_str,
        "phi"                       : phi_str,
        "zout"                      : cloud_top + cloud_top_distance, 
        "cloud_file_path"           : cloud_file_path,
        "tau550"                    : tau550,
        "ic_habit"                  : ic_habit,
        "ic_properties"             : ic_properties
    }
    
    input_file_template_path = 'InputFiles/ic_input_file_template.txt'
                            
    write_input_file(input_file_template_path, generated_input_file_path, input_file_args)
    uvspec_result = get_uvspec_output(generated_input_file_path, temp_dir_path+'temp_output.txt')
        
    # Delete tree of temporary dir
    shutil.rmtree(temp_dir_path, ignore_errors=True)
        
    return uvspec_result


def write_icLUT(LUTpath, wvl_array, phi_array, umu_array, sza_array, r_eff_array, tau550_array, ic_habit_array,
              phi0=270, cloud_top_distance=1, ic_properties = "baum_v36 interpolate", delete=True, CPUs=8, description=""):
    
    """Generates lookup table for specified ranges and grid points and writes file as netCDF to specified paths. Cloud file 
    format and positon as well as uvspec input file template are specified in the function. Sun position phi0 is east by 
    default but can be changed. Default distance from cloud is 1km. ic_properties can also be changed as keyword argument, 
    otherwise yang2013 is default.'CPUs' gives number of cores on which the function is run in parallel.
    """
    
    start_time = timer()
    print("Format input...")
    
    # Define specific temporary file name marker. This was implemented to be able to run function multiple times in parallel. 
    extension = ".nc"
    if LUTpath.endswith(extension):
        LUTname = "_"+LUTpath.replace(extension,'')
    else:
        print("Argument LUTpath has to have .nc extension! Program might fail...")
        
    if "/" in LUTname:
        LUTname = LUTname.replace("/",'_')

    wvl_grid_file_path = 'InputFiles/temp/wvlLUT'+LUTname+'.txt'
    write_wavelength_grid_file(wvl_grid_file_path, wvl_array)
    
    # Initialise data array
    reflectivity = np.ones((len(wvl_array), len(phi_array), len(umu_array), 
                             len(sza_array), len(r_eff_array), len(tau550_array), len(ic_habit_array)))
    
    print("Start libRadtran simulation...")
        
    # Compute vector to be passed to the pool.map function. Includes all relevant index combinations for 'tau550' 'r_eff'. 
    # Should actually be the same as get_index_combinations. --> Check later and move to tools.py
    #first_part = np.array(list(it.combinations_with_replacement(np.arange(len(r_eff_array)), 2)))
    #second_part = np.flip(np.array(list(it.combinations(np.arange(len(r_eff_array)), 2))), axis=1)
    #cloud_index_array = np.concatenate((first_part, second_part))
    
    cloud_index_array = get_index_combinations(len(r_eff_array))
    cloud_index_vector = [(line[0], line[1]) for line in cloud_index_array]
        
    # Looping over entries in data array
    for ihabit in range(len(ic_habit_array)):
        ic_habit = ic_habit_array[ihabit]
        print("Computing habit: ", ic_habit)
        for isza in range(len(sza_array)):
            current_sza_start_time = timer()

            with Pool(processes = CPUs) as p:
                uvspec_results = p.map(get_ic_reflectivity, zip(cloud_index_vector,it.repeat(wvl_array),it.repeat(phi_array),
                                                                it.repeat(umu_array), it.repeat(isza), it.repeat(sza_array),
                                                                it.repeat(r_eff_array), it.repeat(tau550_array), 
                                                                it.repeat(phi0), it.repeat(cloud_top_distance),
                                                                it.repeat(wvl_grid_file_path), it.repeat(ic_habit),
                                                                it.repeat(ic_properties)))
            p.close()


            for icloud in range(len(cloud_index_array)):
                ir_eff, itau550 = cloud_index_array[icloud]

                for iwvl in range(len(wvl_array)):
                    for iumu in range(len(umu_array)):
                        for iphi in range(len(phi_array)):
                            # Write entry. uvspec_result[iwvl, iumu*len(phi_array) + iphi + 1] Is adapted to 
                            # specific uvspec output file format specified in input file. See template. 'icloud' iterates 
                            # over r_eff and tau550 combinations returned by pool.map
                            reflectivity[iwvl, iphi, iumu, isza, ir_eff, itau550, ihabit] = \
                            np.array(uvspec_results)[icloud ,iwvl, iumu*len(phi_array) + iphi + 1]
    
            print("Finished computing sza "+str(sza_array[isza]))
            elapsed_time = (timer() - current_sza_start_time)
            remaining_time = (len(sza_array)-(isza+1))*elapsed_time
            print("Remaining time estimate: ", np.round(remaining_time/60., 4), "min")
        
        
    end_time = timer()
    elapsed_time = (end_time - start_time)/60.
    
    print("Done!")
    # Clear path
    try: 
        os.remove(LUTpath)
    except:
        pass
    
    # Format as xr DataArray
    file = open("InputFiles/ic_input_file_template.txt", "r")
    template = file.read()
    file.close()
    
    print("Format results as Xarray DataArray...")
    LUT = xr.DataArray(
        
        data=reflectivity,
        
        dims=["wvl", "phi", "umu", "sza", "r_eff", "tau550", "ic_habit"],
        
        coords=dict(
            wvl = wvl_array,
            phi = phi_array,
            umu = umu_array,
            sza = sza_array,
            r_eff = r_eff_array,
            tau550 = tau550_array,
            ic_habit = ic_habit_array),
        
        attrs=dict(
            measurement="Reflectivity " + str(cloud_top_distance) +" km above cloud top",
            units="",
            descr=description,
            input_template = template,)
    )
    
    LUT.rename("reflectivity")
    
    LUT.wvl.attrs["units"] = r'nm'
    LUT.phi.attrs["units"] = r'degrees'
    LUT.sza.attrs["units"] = r'degrees'
    LUT.r_eff.attrs["units"] = r'$\mu $m'
    LUT.attrs["computation_time[min]"] = elapsed_time
    
    if delete:
        
        print("Cleaning up...")
        os.remove(wvl_grid_file_path)
    
    print("Write LUT to netCDF file...")
    LUT.load().to_netcdf(LUTpath)
    f = open(LUTpath)
    f.close()
    print('--------------------------------------------------------------------------------')
    print("LUT saved under"+LUTpath)    

    file_stats = os.stat(LUTpath)

    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    
    print('Computation took %f6 minutes.' %elapsed_time)
    
    return