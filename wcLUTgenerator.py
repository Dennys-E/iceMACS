import numpy as np
import os
import xarray as xr 
from timeit import default_timer as timer
import tempfile
import shutil
import itertools as it
from multiprocessing import Pool
from .simulation_tools import write_cloud_file, write_input_file, \
get_formatted_uvspec_output, write_wavelength_grid_file, get_index_combinations
from .conveniences import save_as_netcdf
from .paths import *


def get_wc_reflectivity(args):
    
    """Returns the uvspec output for a single water cloud. Input params are 
    passed as a single zipped argument and listed below. Function is currently 
    being called by a map function in 'WIRTE_wclut', such that 
    'cloud_property_indices' is iterable. This allows for parallel computing of
    the LUT. 
    """
        
    cloud_property_indices, wvl_array, phi_array, umu_array, isza,\
    sza_array, r_eff_array, tau550_array, phi0, cloud_top_distance,\
    wvl_grid_file_path = args
        
    ir_eff, itau550 = cloud_property_indices
    
    r_eff = r_eff_array[ir_eff]
    tau550 = tau550_array[itau550]
        
    temp_dir_path = tempfile.mkdtemp()
        
    cloud_file_path = temp_dir_path+'/temp_cloud_file.dat'
    generated_input_file_path = temp_dir_path+'/temp_input.inp'
        
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
    }
    
    input_file_template_path = INPUT_FILES_DIR+'/wc_input_file_template.txt'
                            
    write_input_file(input_file_template_path, generated_input_file_path, input_file_args)
    
    nwvl = len(wvl_array)
    numu = len(umu_array)
    nphi=len(phi_array)
    
    uvspec_result = get_formatted_uvspec_output(generated_input_file_path,
                                                nwvl, numu, nphi,
                                                temp_dir_path+'/temp_output.txt') 
    """
    uvspec_result = get_uvspec_output(generated_input_file_path,
                                                temp_dir_path+'/temp_output.txt')
    """
        
    # Delete tree of temporary dir
    shutil.rmtree(temp_dir_path, ignore_errors=True)
        
    return uvspec_result


def write_wcLUT(LUTpath, wvl_array, phi_array, umu_array, sza_array, 
                r_eff_array, tau550_array, phi0=0, cloud_top_distance=1, 
                CPUs=8, description=""):
    
    start_time = timer()
    temp_dir_path = tempfile.mkdtemp()
    wvl_grid_file_path = temp_dir_path+'/wvl_grid_file.txt'
    write_wavelength_grid_file(wvl_grid_file_path, wvl_array)
    
    # Initialise data array
    reflectivity = np.ones((len(wvl_array), len(phi_array), len(umu_array), 
                            len(sza_array), len(r_eff_array), 
                            len(tau550_array)))
    
        
    # Compute vector to be passed to the pool.map function. Includes all 
    # relevant index combinations for 'tau550' 'r_eff'. 
    cloud_index_array = get_index_combinations(len(r_eff_array))
    cloud_index_vector = [(line[0], line[1]) for line in cloud_index_array]
        
    # Looping over entries in data array
    for isza in range(len(sza_array)):
        ziplock_args = zip(cloud_index_vector,
                           it.repeat(wvl_array), it.repeat(phi_array), 
                           it.repeat(umu_array), it.repeat(isza), 
                           it.repeat(sza_array), it.repeat(r_eff_array),
                           it.repeat(tau550_array), it.repeat(phi0), 
                           it.repeat(cloud_top_distance),
                           it.repeat(wvl_grid_file_path))
            
        print("Open pool...")
        with Pool(processes = CPUs) as p:
            uvspec_results = np.array(p.map(get_wc_reflectivity, ziplock_args))
        p.close()
        end_of_pool_time = timer()
        print("Pool closed")
            # Should ba faster now that some loops are removed...

        print('Rearanging output...')
        for icloud in range(len(cloud_index_array)):
            ir_eff, itau550 = cloud_index_array[icloud]

            reflectivity[:, :, :, isza, 
                         ir_eff, itau550] = \
                         uvspec_results[icloud] 
    
    print("Done!")
    # Clear path
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    
    # Format as xr DataArray
    file = open(INPUT_FILES_DIR+'/wc_input_file_template.txt', "r")
    template = file.read()
    file.close()
    
    print("Format results as Xarray DataArray...")
    LUT = xr.DataArray(
        
        data=reflectivity,
        
        dims=["wvl", "phi", "umu", "sza", "r_eff", "tau550"],
        
        coords=dict(
            wvl = wvl_array,
            phi = phi_array,
            umu = umu_array,
            sza = sza_array,
            r_eff = r_eff_array,
            tau550 = tau550_array
            ),
        
        attrs=dict(
            measurement="Reflectivity " + str(cloud_top_distance) +" km above cloud top",
            units="",
            descr=description,
            input_template = template,)
    )
    
    LUT = LUT.rename("reflectivity")
    LUT.wvl.attrs["units"] = r'nm'
    LUT.phi.attrs["units"] = r'degrees'
    LUT.sza.attrs["units"] = r'degrees'
    LUT.r_eff.attrs["units"] = r'$\mu $m'
    end_time = timer()
    elapsed_time = (end_time - start_time)/60.
    pool_time = (end_of_pool_time - start_time)/60.
    LUT.attrs["computation_time[min]"] = elapsed_time
    print("Write LUT to netCDF file...")
    save_as_netcdf(LUT, LUTpath)
    print('--------------------------------------------------------------------------------')
    print("LUT saved under"+LUTpath)    
    file_stats = os.stat(LUTpath)
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    print('Simulation took %f6 minutes.' %pool_time)
    print('Computation took %f6 minutes.' %elapsed_time)
    
    return