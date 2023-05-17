import numpy as np
import subprocess
from jinja2 import Template, StrictUndefined
import os
import xarray as xr 
from timeit import default_timer as timer
import tempfile
import shutil
import itertools as it
from multiprocessing import Pool
from tqdm import tqdm
from .tools import *

LIBRADTRAN_PATH = "/project/meteo/work/Dennys.Erdtmann/Thesis/libRadtran-2.0.4"
UVSPEC_PATH = LIBRADTRAN_PATH+"/bin/uvspec"
INPUT_FILES_DIR = "/project/meteo/work/Dennys.Erdtmann/Thesis/iceMACS/templates"


def write_wavelength_grid_file(fpath, wvl_array):                             
    """Saves array as formated txt to be passed to uvspec"""
    np.savetxt(fpath, wvl_array, delimiter=' ')
    
    return


def write_cloud_file(fpath, altitude_grid, WC_array, r_eff_array):
    
    if any([len(altitude_grid)!=len(WC_array), len(altitude_grid)!=len(WC_array)]):
        print('WARNING: Cloud properties do not fit altitude grid!')
        
    cloud_array = np.transpose(np.vstack((np.flip(altitude_grid), np.flip(WC_array), np.flip(r_eff_array))))
    np.savetxt(fpath, cloud_array, delimiter=' ')
    
    return


def write_input_file(input_file_template_path, generated_input_file_path, input_file_args):
    
    f = open(input_file_template_path, 'r')
    input_file_template = f.read()
    f.close()

    j2_template = Template(input_file_template, undefined=StrictUndefined)

    generated_input_file = j2_template.render(input_file_args)

    f = open(generated_input_file_path, 'w')
    f.write(generated_input_file)
    f.close()
    
    return


def save_uvspec_output_under(input_file_path, output_file_path):
    
    f = open(input_file_path, 'r')
    input_file = f.read()
    f.close()

    result = subprocess.run([UVSPEC_PATH], input = input_file, capture_output=True, encoding='ascii')

    output_temp = result.stdout
    
    f = open(output_file_path, 'w')
    f.write(output_temp)
    f.close()
    
    return


def get_index_combinations(length):
    
    """Returns all relevant index combinations for two arrays (e.g. of cloud parameters) with the same length, while exluding 
    redundant combinations."""
    
    first_part = np.array(list(it.combinations_with_replacement(np.arange(length), 2)))
    second_part = np.flip(np.array(list(it.combinations(np.arange(length), 2))), axis=1)
    cloud_index_array = np.concatenate((first_part, second_part))
    
    return cloud_index_array


def get_measurement_arrays(measurementLUT, wvl1, wvl2):
    """Takes a LUT containing measurements and knowledge about the corresponsing "correct" values for r_eff and tau550. Returns 
    arrays containing all relevant combinations to be passed to the luti invert_data_array function."""
    
    LUTcut = measurementLUT
    r_eff_array = LUTcut.coords["r_eff"]
    cloud_index_array = get_index_combinations(len(r_eff_array))

    cloud_param_array = np.zeros(np.shape(cloud_index_array))
    reflectivity_array = np.zeros(np.shape(cloud_index_array))

    for line in range(len(cloud_index_array)):
        ir_eff = cloud_index_array[line,0]
        itau550 = cloud_index_array[line,1]

        cloud_param_array[line,0]=LUTcut.coords["r_eff"].values[ir_eff]
        cloud_param_array[line,1]=LUTcut.coords["tau550"].values[itau550]

        reflectivity_array[line,0]=LUTcut.isel(r_eff = ir_eff, tau550CPUs=itau550).sel(wvl=wvl1).reflectivity.values
        reflectivity_array[line,1]=LUTcut.isel(r_eff = ir_eff, tau550=itau550).sel(wvl=wvl2).reflectivity.values
    
    return reflectivity_array, cloud_param_array


def get_uvspec_output(input_file_path, temp_path):
    
    """To ensure correct format of returned file, uvspec output is temporarily saved in specified directory. If argument delete 
    is True, this file is cleaned before function closes. Argument uniqueness was implemented to protect the temporary file if 
    function is run multiple times in parallel.
    """
    
    f = open(input_file_path, 'r')
    input_file = f.read()
    f.close()

    result = subprocess.run([UVSPEC_PATH], input = input_file, 
                            capture_output=True, encoding='ascii')

    output_temp = result.stdout
    
    f = open(temp_path, 'w')
    f.write(output_temp)
    f.close()
    
    f = open(temp_path, 'r')
    return_value = np.loadtxt(f)
    f.close()
    
    return return_value


def get_formatted_uvspec_output(input_file_path, nwvl, numu, nphi, temp_path):
    
    """To ensure correct format of returned file, uvspec output is temporarily saved in specified directory. If argument delete 
    is True, this file is cleaned before function closes. Argument uniqueness was implemented to protect the temporary file if 
    function is run multiple times in parallel.
    """
    
    f = open(input_file_path, 'r')
    input_file = f.read()
    f.close()

    result = subprocess.run([UVSPEC_PATH], input = input_file, capture_output=True, encoding='ascii')

    output_temp = result.stdout
    
    f = open(temp_path, 'w')
    f.write(output_temp)
    f.close()
    
    f = open(temp_path, 'r')
    return_value = np.loadtxt(f)
    f.close()
    
    result = np.zeros((nwvl, nphi, numu))
    
    for iwvl in range(nwvl):
        for iumu in range(numu):
            for iphi in range(nphi):
                result[iwvl, iphi, iumu] = \
                return_value[iwvl, iumu*nphi + iphi + 1] 
    
    return result



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
        "ic_habit"                  : ic_habit,
        "ic_properties"             : ic_properties
    }
    
    input_file_template_path = INPUT_FILES_DIR+'/ic_input_file_template.txt'
                            
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


def new_write_icLUT(LUTpath, wvl_array, phi_array, umu_array, sza_array, 
                    r_eff_array, tau550_array, ic_habit_array, phi0=0, 
                    cloud_top_distance=1, ic_properties = "baum_v36", 
                    CPUs=8, description=""):
    
    start_time = timer()
    temp_dir_path = tempfile.mkdtemp()
    wvl_grid_file_path = temp_dir_path+'/wvl_grid_file.txt'
    write_wavelength_grid_file(wvl_grid_file_path, wvl_array)
    
    # Initialise data array
    reflectivity = np.ones((len(wvl_array), len(phi_array), len(umu_array), 
                            len(sza_array), len(r_eff_array), 
                            len(tau550_array), len(ic_habit_array)))
    
        
    # Compute vector to be passed to the pool.map function. Includes all 
    # relevant index combinations for 'tau550' 'r_eff'. 
    cloud_index_array = get_index_combinations(len(r_eff_array))
    cloud_index_vector = [(line[0], line[1]) for line in cloud_index_array]
    
    r_eff_v, tau550_v = np.meshgrid(r_eff_array, tau550_array)
    flat_r_eff, flat_tau550 = r_eff_v.flatten(), tau550_v.flatten()
        
    # Looping over entries in data array
    LUTs=[]
    for ihabit in range(len(ic_habit_array)):
        ic_habit = ic_habit_array[ihabit]
        print("Computing habit: ", ic_habit)
        for isza in range(len(sza_array)):
            ziplock_args = zip(cloud_index_vector,
                               it.repeat(wvl_array), it.repeat(phi_array), 
                               it.repeat(umu_array), it.repeat(isza), 
                               it.repeat(sza_array), it.repeat(r_eff_array),
                               it.repeat(tau550_array), it.repeat(phi0), 
                               it.repeat(cloud_top_distance),
                               it.repeat(wvl_grid_file_path),
                               it.repeat(ic_habit), it.repeat(ic_properties))
            
            print("Open pool...")
            with Pool(processes = CPUs) as p:
                # When optimizing this part: maybe you can define the zpped 
                # arguments as args = tqdm(zip(...)) and see if progress bar 
                # will show up. Do the same when including the map() function
                # below to increase writing speed
                uvspec_results = np.array(p.map(get_ic_reflectivity, ziplock_args))
            p.close()
            end_of_pool_time = timer()
            print("Pool closed")
            # Slows down process dramatically. Replace for loops with faster 
            # method, e.g. list comprehension? Would remove outer loops...
            
            single_results=[]
            for icloud in range(len(flat_r_eff)):
                print(icloud)
                
                single_result = xr.DataArray(
                        data = uvspec_results[icloud,:,:,:],
                        dims=["wvl", "umu", "phi"],
                        coords=dict(
                                wvl=wvl_array,
                                umu=umu_array,
                                phi=phi_array)
                        ).expand_dims({
                                "r_eff":[flat_r_eff[icloud]],
                                "tau550":[flat_tau550[icloud]]}).rename('reflectivity')
                single_results.append(single_result)
                
            merged_results=xr.merge(single_results)
            
            merged_results = merged_results.expand_dims({
                            "ic_habit":[ic_habit_array[ihabit]],
                            "sza":[sza_array[isza]]})
            
            LUTs.append(merged_results)
            
            
            print('Rearanging output...')
        """
            for icloud in range(len(r_eff_v.flatten())):
                ir_eff, itau550 = cloud_index_array[icloud]

                for iwvl in range(len(wvl_array)):
                    for iumu in range(len(umu_array)):
                        for iphi in range(len(phi_array)):

                            reflectivity[iwvl, iphi, iumu, isza, 
                                         ir_eff, itau550, ihabit] = \
                            uvspec_results[icloud ,iwvl, 
                                    iumu*len(phi_array) + iphi + 1] """
    
    LUT=xr.merge(LUTs)
    
    print("Done!")
    # Clear path
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    
    # Format as xr DataArray
    file = open("InputFiles/ic_input_file_template.txt", "r")
    template = file.read()
    file.close()
    
    print("Format results as Xarray DataArray...")
        
    LUT.attrs=dict(    
            measurement="Reflectivity " + str(cloud_top_distance) +" km above cloud top",
            units="",
            descr=description,
            input_template = template)
    
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
    
    print(LUT)
    return

def write_icLUT(LUTpath, wvl_array, phi_array, umu_array, sza_array, 
                r_eff_array, tau550_array, ic_habit_array, phi0=0, 
                cloud_top_distance=1, ic_properties = "baum_v36", 
                CPUs=8, description=""):
    
    start_time = timer()
    temp_dir_path = tempfile.mkdtemp()
    wvl_grid_file_path = temp_dir_path+'/wvl_grid_file.txt'
    write_wavelength_grid_file(wvl_grid_file_path, wvl_array)
    
    # Initialise data array
    reflectivity = np.ones((len(wvl_array), len(phi_array), len(umu_array), 
                            len(sza_array), len(r_eff_array), 
                            len(tau550_array), len(ic_habit_array)))
    
        
    # Compute vector to be passed to the pool.map function. Includes all relevant index combinations for 'tau550' 'r_eff'. 
    cloud_index_array = get_index_combinations(len(r_eff_array))
    cloud_index_vector = [(line[0], line[1]) for line in cloud_index_array]
        
    # Looping over entries in data array
    for ihabit in range(len(ic_habit_array)):
        ic_habit = ic_habit_array[ihabit]
        print("Computing habit: ", ic_habit)
        for isza in range(len(sza_array)):
            ziplock_args = zip(cloud_index_vector,
                               it.repeat(wvl_array), it.repeat(phi_array), 
                               it.repeat(umu_array), it.repeat(isza), 
                               it.repeat(sza_array), it.repeat(r_eff_array),
                               it.repeat(tau550_array), it.repeat(phi0), 
                               it.repeat(cloud_top_distance),
                               it.repeat(wvl_grid_file_path),
                               it.repeat(ic_habit), it.repeat(ic_properties))
            
            print("Open pool...")
            with Pool(processes = CPUs) as p:
                # When optimizing this part: maybe you can define the zpped arguments as args = tqdm(zip(...)) and see if progress bar will show up. Do the same when including the map() function below to increase writing speed
                uvspec_results = np.array(p.map(get_ic_reflectivity, ziplock_args))
            p.close()
            end_of_pool_time = timer()
            print("Pool closed")
            # Should ba faster now that some loops are removed...

            print('Rearanging output...')
            for icloud in range(len(cloud_index_array)):
                ir_eff, itau550 = cloud_index_array[icloud]

                reflectivity[:, :, :, isza, 
                             ir_eff, itau550, ihabit] = \
                             uvspec_results[icloud] 
    
    print("Done!")
    # Clear path
    shutil.rmtree(temp_dir_path, ignore_errors=True)
    
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