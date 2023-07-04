import numpy as np
import os
import xarray as xr 
from timeit import default_timer as timer
import tempfile
import shutil
import itertools as it
from multiprocessing import Pool
from .simulation_tools import write_cloud_file, write_input_file, get_formatted_mystic_output,\
get_formatted_uvspec_output, write_wavelength_grid_file, get_index_combinations
from .tools import save_as_netcdf
from .paths import *


def get_pol_ic_stokes_params(args):
        
    cloud_property_indices, wvl_array, phi, umu, isza,\
    sza_array, r_eff_array, tau550_array, phi0, cloud_top_distance,\
    wvl_grid_file_path, ic_habit, surface_roughness, ic_properties = args
        
    ir_eff, itau550 = cloud_property_indices
    
    r_eff = r_eff_array[ir_eff]
    tau550 = tau550_array[itau550]
    sza = sza_array[isza]
        
    temp_dir_path = tempfile.mkdtemp()
        
    cloud_file_path = temp_dir_path+'/temp_cloud_file.dat'
    generated_input_file_path = temp_dir_path+'/temp_input.inp'
        
    if len(r_eff_array) is not len(tau550_array):
        print("Cloud property arrays must have the same shape!")
        return
    
    # Define cloud structure and generate ic file with corresponding values
    altitude_grid = np.array([7, 8, 9, 10])
    WC_array = np.array([0.1, 0.1, 0.1, 0.1])
    r_eff_array = r_eff*np.array([1, 1, 1, 1])
    write_cloud_file(cloud_file_path, altitude_grid, WC_array, r_eff_array)
    cloud_file = np.loadtxt(cloud_file_path)
    cloud_top = np.max(cloud_file[:,0])
    
    if ic_properties == "yang2013":
        habit_mode = "ic_habit_yang2013"
    else:
        habit_mode = "ic_habit"
        surface_roughness = ""
    
    # Get uvspec output
    input_file_args = {
        "wavelength_grid_file_path" : wvl_grid_file_path,
        "sza"                       : sza,
        "phi0"                      : phi0,
        "umu"                       : umu,
        "phi"                       : phi,
        "zout"                      : cloud_top + cloud_top_distance, 
        "cloud_file_path"           : cloud_file_path,
        "tau550"                    : tau550,
        "habit_mode"                : habit_mode, 
        "ic_habit"                  : ic_habit,
        "surface_roughness"         : surface_roughness,
        "ic_properties"             : ic_properties
    }
    
    input_file_template_path = INPUT_FILES_DIR+'/pol_ic_input_file_template.txt'
                           
    write_input_file(input_file_template_path, generated_input_file_path, 
                     input_file_args)
    
    uvspec_result = get_formatted_mystic_output(generated_input_file_path,
                                                temp_dir_path) 
    
        
    # Delete tree of temporary dir
    shutil.rmtree(temp_dir_path, ignore_errors=True)
        
    return uvspec_result


def write_pol_icLUT(LUTpath, wvl_array, phi_array, umu_array, sza_array, 
                    r_eff_array, tau550_array, ic_habit_array, phi0=0, 
                    cloud_top_distance=1, ic_properties="baum_v36", 
                    surface_roughness="severe", CPUs=8, description=""):
    
    start_time = timer()
    #temp_dir_path = tempfile.mkdtemp()
    wvl_grid_file_path = os.getcwd()+'wvl_grid_file.txt'
    write_wavelength_grid_file(wvl_grid_file_path, wvl_array)
    
    # Initialise data array. Last indicated dimension of size four are the
    # four Stokes parameters.
    stokes_params = np.ones((len(wvl_array), len(phi_array), len(umu_array), 
                             len(sza_array), len(r_eff_array), 
                             len(tau550_array), len(ic_habit_array), 4))
    
        
    # Compute vector to be passed to the pool.map function. Includes all
    # relevant index combinations for 'tau550' 'r_eff'. 
    cloud_index_array = get_index_combinations(len(r_eff_array))
    cloud_index_vector = [(line[0], line[1]) for line in cloud_index_array]
        
    # Looping over entries in data array
    for ihabit in range(len(ic_habit_array)):
        ic_habit = ic_habit_array[ihabit]
        print("Computing habit: ", ic_habit)
        for isza in range(len(sza_array)):
            for iumu in range(len(umu_array)):
                for iphi in range(len(phi_array)):
                    
                    umu = umu_array[iumu]
                    phi = phi_array[iphi]
                    
                    ziplock_args = zip(cloud_index_vector,
                                       it.repeat(wvl_array),
                                       it.repeat(phi), 
                                       it.repeat(umu), 
                                       it.repeat(isza), 
                                       it.repeat(sza_array), 
                                       it.repeat(r_eff_array),
                                       it.repeat(tau550_array),
                                       it.repeat(phi0), 
                                       it.repeat(cloud_top_distance),
                                       it.repeat(wvl_grid_file_path),
                                       it.repeat(ic_habit), 
                                       it.repeat(surface_roughness),
                                       it.repeat(ic_properties))
            
                    print("Open pool...")
                    with Pool(processes = CPUs) as p:
                        mystic_results = np.array(p.map(get_pol_ic_stokes_params, 
                                                       ziplock_args))
                    p.close()
                    end_of_pool_time = timer()
                    print("Pool closed")
                    print('Rearanging output...')
                    for icloud in range(len(cloud_index_array)):
                        ir_eff, itau550 = cloud_index_array[icloud]
        
                        stokes_params[:, iphi, iumu, isza, 
                                      ir_eff, itau550, ihabit, :] = \
                                      mystic_results[icloud, :] 
    
    print("Done!")
    # Clear temporary path
    
    # Format as xr DataArray
    file = open(INPUT_FILES_DIR+"/pol_ic_input_file_template.txt", "r")
    template = file.read()
    file.close()
    
    print("Format results as Xarray DataArray...")
    
    I = xr.DataArray(
        
        data=stokes_params[:,:,:,:,:,:,:,0],
        
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
    ).rename('I')

    Q = xr.DataArray(
        
        data=stokes_params[:,:,:,:,:,:,:,1],
        
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
    ).rename('Q')
    
    U = xr.DataArray(
        
        data=stokes_params[:,:,:,:,:,:,:,2],
        
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
    ).rename('U')
   
    V = xr.DataArray(
        
        data=stokes_params[:,:,:,:,:,:,:,3],
        
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
    ).rename('V')
        
    LUT = xr.merge([I, Q, U, V])
    
    LUT.wvl.attrs["units"] = r'nm'
    LUT.phi.attrs["units"] = r'degrees'
    LUT.sza.attrs["units"] = r'degrees'
    LUT.r_eff.attrs["units"] = r'$\mu $m'
    end_time = timer()
    elapsed_time = (end_time - start_time)/60.
    pool_time = (end_of_pool_time - start_time)/60.
    print("Write LUT to netCDF file...")
    LUT.attrs["computation_time[min]"] = elapsed_time
    
    save_as_netcdf(LUT, LUTpath)
    
    print('--------------------------------------------------------------------------------')
    print("LUT saved under"+LUTpath)    
    file_stats = os.stat(LUTpath)
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    print('Simulation took %f6 minutes.' %pool_time)
    print('Computation took %f6 minutes.' %elapsed_time)
    
    return