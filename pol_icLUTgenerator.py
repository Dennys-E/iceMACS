import numpy as np
import os
import xarray as xr 
from timeit import default_timer as timer
import tempfile
import shutil
import itertools as it
from multiprocessing import Pool
from .simulation_tools import (write_cloud_file_from_heights, 
                               write_input_file_from_RAM, 
                               get_formatted_mystic_output,
                               write_wavelength_grid_file)
from .conveniences import save_as_netcdf
from .paths import *


def get_pol_ic_stokes_params(args):
    """This function is bein called by write_pol_icLUT in order to allow for 
    parallel computing on multiple CPUs."""
        
    (umu, 
     input_file_template, 
     wvl_grid_file_path,
     phi,
     tau550, 
     sza,
     r_eff,
     ic_habit,
     cloud_depth, 
     cloud_top,
     zout,
     ic_properties,
     surface_roughness) = args
        
    temp_dir_path = tempfile.mkdtemp()
        
    cloud_file_path = temp_dir_path+'/temp_cloud_file.dat'
    generated_input_file_path = temp_dir_path+'/temp_input.inp'

    # cloud base is defined relative to depth and cloud top in order to avoid
    # unrealistic or unnecessary combinations of cloud top and cloud base 
    # e.g. "negative" thickness etc.
    cloud_base = cloud_top - cloud_depth
    write_cloud_file_from_heights(cloud_file_path, cloud_base, cloud_top, r_eff)
    
    if ic_properties == "yang2013":
        habit_mode = "ic_habit_yang2013"
    else:
        habit_mode = "ic_habit"
        surface_roughness = ""
    
    # Get uvspec output
    input_file_args = {
        "wavelength_grid_file_path" : wvl_grid_file_path,
        "sza"                       : sza,
        "umu"                       : umu,
        "phi"                       : phi,
        "cloud_file_path"           : cloud_file_path,
        "tau550"                    : tau550,
        "habit_mode"                : habit_mode, 
        "ic_habit"                  : ic_habit,
        "surface_roughness"         : surface_roughness,
        "ic_properties"             : ic_properties,
        "zout"                      : zout
    }
    
    #input_file_template_path = INPUT_FILES_DIR+'/pol_ic_input_file_template.txt'
                           
    write_input_file_from_RAM(input_file_template, generated_input_file_path, 
                              input_file_args)
    
    # with open(generated_input_file_path) as inp:
    #     print(inp.read())

    # with open(cloud_file_path) as cloud:
    #     print(cloud.read())

    uvspec_result = get_formatted_mystic_output(generated_input_file_path,
                                                temp_dir_path) 
    
    
        
    # Delete tree of temporary dir
    shutil.rmtree(temp_dir_path, ignore_errors=True)
        
    return uvspec_result


def write_pol_icLUT(LUTpath, input_file_template, wvl_array, 
                    phi_array, umu_array, sza_array, 
                    r_eff_array, tau550_array, ic_habit_array, 
                    cloud_depth_array, cloud_top_array, zout_array,
                    ic_properties="baum_v36", surface_roughness="severe", 
                    CPUs=8, description=""):
    
    start_time = timer()
    temp_dir_path = tempfile.mkdtemp()
    wvl_grid_file_path = temp_dir_path+'/wvl_grid_file.txt'
    write_wavelength_grid_file(wvl_grid_file_path, wvl_array)
    
    # Initialise data array. Last indicated dimension of size four are the
    # four Stokes parameters.
    stokes_params = np.ones((len(wvl_array), len(phi_array), len(umu_array), 
                             len(sza_array), len(r_eff_array), 
                             len(tau550_array), len(ic_habit_array), 
                             len(cloud_depth_array), len(cloud_top_array), 
                             len(zout_array), 4))
    
    stokes_std = np.copy(stokes_params)
        
    # Looping over entries in data array. MYSTIC does not take multiple umu and phi values at once...
    ntasks = (len(ic_habit_array)*len(sza_array)*len(tau550_array)
              *len(phi_array)*len(r_eff_array)*len(cloud_depth_array)
              *len(cloud_top_array))*len(zout_array)
    print('Total calls to compute: ', ntasks)
    current_call = 1
    for ihabit, ic_habit in enumerate(ic_habit_array):
        print("Computing habit: ", ic_habit)
        for isza, sza in enumerate(sza_array):
            for itau550, tau550 in enumerate(tau550_array):
                for iphi, phi in enumerate(phi_array):
                    for ir_eff, r_eff in enumerate(r_eff_array):
                        for icloud_depth, cloud_depth in enumerate(cloud_depth_array):
                            for icloud_top, cloud_top in enumerate(cloud_top_array):
                                for izout, zout in enumerate(zout_array):
                                    print('Call: ', current_call, '/', ntasks)

                                    
                                    ziplock_args = zip(umu_array,
                                                    it.repeat(input_file_template),
                                                    it.repeat(wvl_grid_file_path),
                                                    it.repeat(phi),
                                                    it.repeat(tau550), 
                                                    it.repeat(sza), 
                                                    it.repeat(r_eff), 
                                                    it.repeat(ic_habit), 
                                                    it.repeat(cloud_depth),
                                                    it.repeat(cloud_top),
                                                    it.repeat(zout),
                                                    it.repeat(ic_properties),
                                                    it.repeat(surface_roughness))

                                    print("Open pool for ", "r_eff=", r_eff, "phi=", phi, 
                                        "tau=", tau550, "sza=", sza, "cloud_depth=", cloud_depth,
                                        "cloud_top=", cloud_top, "zout=", zout)
                                    
                                    start_of_pool_time = timer()
                                    with Pool(processes = CPUs) as p:
                                        mystic_results = np.array(p.map(get_pol_ic_stokes_params, 
                                                                        ziplock_args))
                                    p.close()
                                    end_of_pool_time = timer()
                                    last_call_time = end_of_pool_time-start_of_pool_time
                                    print("Pool closed, took: ", last_call_time, "s")
                                    time_estimate = (ntasks-current_call)*last_call_time
                                    print("Estimated remaining time: ", time_estimate/60., "min")
                                    current_call += 1                  
                                    print('Rearanging output...')
                                    for iumu in range(len(umu_array)):
                        
                                        stokes_params[:, iphi, iumu, isza, 
                                                    ir_eff, itau550, ihabit,
                                                    icloud_depth, icloud_top, 
                                                    izout, :] = \
                                                    mystic_results[iumu, 0, :] 
                                                    
                                        stokes_std[:, iphi, iumu, isza, 
                                                ir_eff, itau550, ihabit, 
                                                icloud_depth, icloud_top, 
                                                izout, :] = \
                                                mystic_results[iumu, 1, :] 
            
            
    print("Done!")
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    # Format as xr DataArray
    
    print("Format results as Xarray DataArray...")
    
    dims = ["wvl", "phi", "umu", "sza", "r_eff", "tau550", "ic_habit", "cloud_depth", "cloud_top", "zout"]
    coords = dict(
            wvl = wvl_array,
            phi = phi_array,
            umu = umu_array,
            sza = sza_array,
            r_eff = r_eff_array,
            tau550 = tau550_array,
            ic_habit = ic_habit_array,
            cloud_depth = cloud_depth_array,
            cloud_top = cloud_top_array,
            zout = zout_array)
    
    I = xr.DataArray(data=stokes_params[:,:,:,:,:,:,:,:,:,:,0], 
                     dims=dims,
                     coords=coords).rename('I')
    
    Q = xr.DataArray(data=stokes_params[:,:,:,:,:,:,:,:,:,:,1], 
                     dims=dims,
                     coords=coords).rename('Q')
    
    U = xr.DataArray(data=stokes_params[:,:,:,:,:,:,:,:,:,:,2], 
                     dims=dims,
                     coords=coords).rename('U')
    
    V = xr.DataArray(data=stokes_params[:,:,:,:,:,:,:,:,:,:,3], 
                     dims=dims,
                     coords=coords).rename('V')
    
    I_std = xr.DataArray(data=stokes_std[:,:,:,:,:,:,:,:,:,:,0], 
                     dims=dims,
                     coords=coords).rename('I_std')
    
    Q_std = xr.DataArray(data=stokes_std[:,:,:,:,:,:,:,:,:,:,1], 
                     dims=dims,
                     coords=coords).rename('Q_std')
    
    U_std = xr.DataArray(data=stokes_std[:,:,:,:,:,:,:,:,:,:,2], 
                     dims=dims,
                     coords=coords).rename('U_std')
    
    V_std = xr.DataArray(data=stokes_std[:,:,:,:,:,:,:,:,:,:,3], 
                     dims=dims,
                     coords=coords).rename('V_std')

        
    LUT = xr.merge([I, Q, U, V, I_std, Q_std, U_std, V_std])
    
    LUT.wvl.attrs["units"] = r'nm'
    LUT.phi.attrs["units"] = r'degrees'
    LUT.sza.attrs["units"] = r'degrees'
    LUT.r_eff.attrs["units"] = r'$\mu $m'
    LUT.cloud_depth.attrs["units"] = r'km'
    LUT.cloud_top.attrs["units"] = r'km'
    LUT.zout.attrs["units"] = r"km"
    end_time = timer()
    elapsed_time = (end_time - start_time)/60.
    print("Write LUT to netCDF file...")
    LUT.attrs=dict(descr=description,
                   input_template = input_file_template,
                   computation_time = str(elapsed_time)+'[min]')

    save_as_netcdf(LUT, LUTpath)
    
    print('--------------------------------------------------------------------------------')
    print("LUT saved under "+LUTpath)    
    file_stats = os.stat(LUTpath)
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    print('Computation took %f6 minutes.' %elapsed_time)
    
    return