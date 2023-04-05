import numpy as np
import xarray as xr 
import netCDF4 as nc
import tempfile
import shutil
import itertools as it
import subprocess
from jinja2 import Template, StrictUndefined

UVSPEC_PATH = "/project/meteo/work/Dennys.Erdtmann/Thesis/libRadtran-2.0.4/bin/uvspec"

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


def get_uvspec_output(input_file_path, temp_path):
    
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
    
    return return_value


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

        reflectivity_array[line,0]=LUTcut.isel(r_eff = ir_eff, tau550=itau550).sel(wvl=wvl1).reflectivity.values
        reflectivity_array[line,1]=LUTcut.isel(r_eff = ir_eff, tau550=itau550).sel(wvl=wvl2).reflectivity.values
    
    return reflectivity_array, cloud_param_array