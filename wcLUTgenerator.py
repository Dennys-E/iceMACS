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
import luti
from luti.xarray import invert_data_array
from luti import Alphachecker, NeighbourInterpolator


UVSPEC_PATH = "/project/meteo/work/Dennys.Erdtmann/Thesis/libRadtran-2.0.4/bin/uvspec"
# Luti path
sys.path.insert(0, os.path.abspath('/project/meteo/work/Dennys.Erdtmann/Thesis/Python Packages/luti'))

def spec_plot(output_file, phi_array, umu_array):
    
    num_phi = len(phi_array)
    num_umu = len(umu_array)
    
    plt.figure(figsize=(15,10))
    for i_umu in range (num_umu):
        for i_phi in range (num_phi):
            plt.plot(output_file[:,0], output_file[:,i_umu*num_phi + i_phi+1], 
                     linewidth=1, label="Phi = " + str(phi_array[i_phi]) + " / Umu = " + str(umu_array[i_umu]))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.show()
    
    return


def write_wavelength_grid_file(fpath, wvl_array):
    
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
    
    """Returns all relevant index combinations for two arrays (e.g. of cloud parameters) with the same length, while exluding redundant combinations."""
    
    first_part = np.array(list(it.combinations_with_replacement(np.arange(length), 2)))
    second_part = np.flip(np.array(list(it.combinations(np.arange(length), 2))), axis=1)
    cloud_index_array = np.concatenate((first_part, second_part))
    
    return cloud_index_array

def get_wc_reflectivity(args):
    
    """Returns the uvspec output for a single cloud with specified parameters. Input params are passed as a single zipped
    argument and listed below. Function is currently being called by a map function in 'writeLUT', such that 
    'wc_property_indices' is iterable. This
    allows for parallel computing of the LUT. For technical reasons (in libRadtran) 550nm has to be included in the wvl sample
    points, so that tau550 can be explicitly varied.
    """
        
    wc_property_indices, wvl_array, phi_array, umu_array, isza,\
    sza_array, r_eff_array, tau550_array, phi0, cloud_top_distance, wvl_grid_file_path = args
        
    ir_eff, itau550 = wc_property_indices
    
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
    altitude_grid = np.array([2, 3, 4, 5])
    WC_array = np.array([0.1, 0.1, 0.1, 0])
    r_eff_array = r_eff*np.array([1, 1, 1, 0])
    
    # Generate wc file with corresponding values
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
        "tau550"                    : tau550
    }
    
    input_file_template_path = 'InputFiles/wc_input_file_template.txt'
                            
    write_input_file(input_file_template_path, generated_input_file_path, input_file_args)
    uvspec_result = get_uvspec_output(generated_input_file_path, temp_dir_path+'temp_output.txt')
        
    # Delete tree of temporary dir
    shutil.rmtree(temp_dir_path, ignore_errors=True)
        
    return uvspec_result

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


def write_wcLUT(LUTpath, wvl_array, phi_array, umu_array, sza_array, r_eff_array, tau550_array,
              phi0=270, cloud_top_distance=1, delete=True, CPUs=8):
    
    """Generates water cloud lookup table for specified ranges and grid points and writes file as netCDF to specified paths.
    Cloud file format and positon as well as uvspec input file template are specified in the function. Sun position phi0 is 
    east by default but can be changed. Default distance from cloud is 1km. 'CPUs' gives number of cores on which the function 
    is run in parallel.
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
                             len(sza_array), len(r_eff_array), len(tau550_array)))
    
    print("Start libRadtran simulation...")
        
    # Compute vector to be passed to the pool.map function. Includes all relevant index combinations for 'tau550' 'r_eff'.
    first_part = np.array(list(it.combinations_with_replacement(np.arange(len(r_eff_array)), 2)))
    second_part = np.flip(np.array(list(it.combinations(np.arange(len(r_eff_array)), 2))), axis=1)
    wc_index_array = np.concatenate((first_part, second_part))
    wc_index_vector = [(line[0], line[1]) for line in wc_index_array]
        
    # Looping over entries in data array
    for isza in range(len(sza_array)):
        
            # Depending on wether a crystal habit has been specified as keyword argument, one of the two get_reflectivity
            # functions, water or ice, is called
        if ic_habit == None:
        
            with Pool(processes = CPUs) as p:
                uvspec_results = p.map(get_wc_reflectivity, zip(wc_index_vector,it.repeat(wvl_array),it.repeat(phi_array),
                                                                it.repeat(umu_array), it.repeat(isza), it.repeat(sza_array),
                                                                it.repeat(r_eff_array), it.repeat(tau550_array), 
                                                                it.repeat(phi0), 
                                                                it.repeat(cloud_top_distance), it.repeat(wvl_grid_file_path)))
            p.close()
            
        else:
            
            with Pool(processes = CPUs) as p:
                uvspec_results = p.map(get_ic_reflectivity, zip(wc_index_vector,it.repeat(wvl_array),it.repeat(phi_array),
                                                                it.repeat(umu_array), it.repeat(isza), it.repeat(sza_array),
                                                                it.repeat(r_eff_array), it.repeat(tau550_array), 
                                                                it.repeat(phi0), it.repeat(cloud_top_distance),
                                                                it.repeat(wvl_grid_file_path), it.repeat(ic_habit)))
            p.close()
            
       
        for icloud in range(len(wc_index_array)):
            ir_eff, itau550 = wc_index_array[icloud]
            
            for iwvl in range(len(wvl_array)):
                for iumu in range(len(umu_array)):
                    for iphi in range(len(phi_array)):
                        # Write entry. uvspec_result[iwvl, iumu*len(phi_array) + iphi + 1] Is adapted to 
                        # specific uvspec output file format specified in input file. See template. 'icloud' iterates over
                        # r_eff and tau550 combinations returned by pool.map
                        reflectivity[iwvl, iphi, iumu, isza, ir_eff, itau550] = \
                        np.array(uvspec_results)[icloud ,iwvl, iumu*len(phi_array) + iphi + 1]
    
        print("Finished computing sza"+str(isza))
        
    end_time = timer()
    elapsed_time = (end_time - start_time)/60.
    
    print("Done!")
    # Clear path
    try: 
        os.remove(LUTpath)
    except:
        pass
    
    # Format as xr DataArray
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
            tau550 = tau550_array),
        
        attrs=dict(
            description="Reflectivity " + str(cloud_top_distance) +" km above cloud top",
            units="",)
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
    first_part = np.array(list(it.combinations_with_replacement(np.arange(len(r_eff_array)), 2)))
    second_part = np.flip(np.array(list(it.combinations(np.arange(len(r_eff_array)), 2))), axis=1)
    wc_index_array = np.concatenate((first_part, second_part))
    wc_index_vector = [(line[0], line[1]) for line in wc_index_array]
        
    # Looping over entries in data array
    for ihabit in range(len(ic_habit_array)):
        ic_habit = ic_habit_array[ihabit]
        print("Computing habit: ", ic_habit)
        for isza in range(len(sza_array)):
            current_sza_start_time = timer()

            with Pool(processes = CPUs) as p:
                uvspec_results = p.map(get_ic_reflectivity, zip(wc_index_vector,it.repeat(wvl_array),it.repeat(phi_array),
                                                                it.repeat(umu_array), it.repeat(isza), it.repeat(sza_array),
                                                                it.repeat(r_eff_array), it.repeat(tau550_array), 
                                                                it.repeat(phi0), it.repeat(cloud_top_distance),
                                                                it.repeat(wvl_grid_file_path), it.repeat(ic_habit),
                                                                it.repeat(ic_properties)))
            p.close()


            for icloud in range(len(wc_index_array)):
                ir_eff, itau550 = wc_index_array[icloud]

                for iwvl in range(len(wvl_array)):
                    for iumu in range(len(umu_array)):
                        for iphi in range(len(phi_array)):
                            # Write entry. uvspec_result[iwvl, iumu*len(phi_array) + iphi + 1] Is adapted to 
                            # specific uvspec output file format specified in input file. See template. 'icloud' iterates 
                            # over
                            # r_eff and tau550 combinations returned by pool.map
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


def read_LUT(LUTpath):
    
    LUT = nc.Dataset(LUTpath)
    LUT = xr.open_dataset(xr.backends.NetCDF4DataStore(LUT))
    
    return LUT

def get_measurement_arrays(measurementLUT, wvl1, wvl2):
    """Takes a LUT containing measurements and knowledge about the corresponsing "correct" values for r_eff and tau550. Returns arrays containing all relevant combinations to be passed to the luti invert_data_array function."""
    
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
        diffs[line, 0]=inverted.sel(input_params="r_eff", Rone=measured_R1[line], Rtwo=measured_R2[line]).values - cloud_param_array[line,0]
        diffs[line, 1]=inverted.sel(input_params="tau550", Rone=measured_R1[line], Rtwo=measured_R2[line]).values - cloud_param_array[line,1]
        errors[line, 0]=diffs[line,0]**2
        errors[line, 1]=diffs[line,1]**2
    r_eff_errors, tau550_errors = [error for error in errors[:,0] if not np.isnan(error)], [error for error in errors[:,1] if not np.isnan(error)]
    r_eff_diffs, tau550_diffs = [diff for diff in diffs[:,0] if not np.isnan(diff)], [diff for diff in diffs[:,1] if not np.isnan(diff)]
    
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
    
def display_LUT(LUT, wvl1, wvl2):
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

        ax.scatter(measured_R2, measured_R1, marker='X')
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
