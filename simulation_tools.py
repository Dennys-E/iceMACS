# This file is supposed to contain function being used by both ic and 
# wcLUTgenerator functions

import numpy as np
from jinja2 import Template, StrictUndefined
import subprocess
import itertools as it
from .paths import *


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


def get_uvspec_output(input_file_path, temp_path):
    """Returns standard 2D uvspec output. File is temporarily saved in order to
    ensure correct format.
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


def get_formatted_mystic_output(input_file_path, temp_path):
    """Returns Stokes vectors for each wavelengths, as returned by mystic.
    """
    
    f = open(input_file_path, 'r')
    input_file = f.read()
    f.close()
    
    result = subprocess.run([UVSPEC_PATH], input = input_file, 
                            capture_output=False, encoding='ascii')
    
    f = open(temp_path+'/mc.rad.spc', 'r')
    mystic_output = np.loadtxt(f)
    f.close()
    
    n_wvl = np.int(np.shape(mystic_output)[0]/4.)
    
    stokes_params = mystic_output[:,-1].reshape(n_wvl, 4)
    
    return stokes_params


def get_formatted_uvspec_output(input_file_path, nwvl, numu, nphi, temp_path):
    
    """Returns uvspec output as an array of the format (wvl, umu, phi) instead 
    of 2D standard output. To ensure correct format of returned file, uvspec 
    output is temporarily saved in specified directory.
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