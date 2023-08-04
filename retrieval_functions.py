"""Should contain all function that take a LUT, formatted data and return 
cloud optical porperties."""

import numpy as np
from tqdm import tqdm
import xarray as xr
import itertools as it
from multiprocessing import Pool
from luti import Alphachecker
from luti.xarray import invert_data_array
from .conveniences import read_LUT
import matplotlib.pyplot as plt

def fast_retrieve(inverted, merged_data, wvl1, wvl2, R1_name, R2_name, 
                  umu_bins=10, phi_bins=10):
    
    umu_counts, umu_bin_edges = np.histogram(merged_data.umu.to_numpy()\
                                             .flatten(), 
                                             bins=umu_bins)
    phi_counts, phi_bin_edges = np.histogram(merged_data.phi.to_numpy()\
                                             .flatten(), 
                                             bins=phi_bins)
    
    print(umu_bin_edges)
    
    inverted = inverted.sel(sza=merged_data.sza.mean(), method='nearest')
    
    print('Start loop...')
    last_result = None 

    for i_phi_bin in tqdm(range(len(phi_bin_edges)-1)):
        
        phi_mean = (phi_bin_edges[i_phi_bin+1]+phi_bin_edges[i_phi_bin])/2.
             
        for i_umu_bin in range(len(umu_bin_edges)-1):
            
            umu_mean = (umu_bin_edges[i_umu_bin+1]+umu_bin_edges[i_umu_bin])/2.
            #umu_mean = umu_bin_edges[i_umu_bin+1]

            data_cut = merged_data.reflectivity\
            .where(umu_bin_edges[i_umu_bin]<=merged_data.umu)\
            .where(merged_data.umu<umu_bin_edges[i_umu_bin+1])\
            .where(phi_bin_edges[i_phi_bin]<=merged_data.phi)\
            .where(merged_data.phi<phi_bin_edges[i_phi_bin+1]).dropna(dim='x', 
                  how='all')


            if data_cut.x.size == 0:
                continue

            LUT = inverted.sel(phi=phi_mean, umu=umu_mean, 
                               method='nearest').rename({R1_name:'Rone',
                                                         R2_name:'Rtwo'})

            result = LUT.interp(Rone=data_cut.sel(wavelength=wvl1, 
                                                  method='nearest'), 
                                Rtwo=data_cut.sel(wavelength=wvl2, 
                                                  method='nearest'))
                
            result['r_eff'] = (result.sel(input_params='r_eff')
                                     .reflectivity)
            result['tau550'] = (result.sel(input_params='tau550')
                                      .reflectivity)

            if last_result is None:
                last_result = (result.drop_vars(['phi', 'umu', 'reflectivity', 
                                                 'input_params']))
               
            else: 
                current_result = (result.drop_vars(['phi', 'umu', 'reflectivity', 
                                                    'input_params']))
                    
            last_result = xr.merge([last_result, current_result])



    return last_result


def retrieve_image(LUTpath, wvl1, wvl2, merged_data, habit):
    
    time_array = merged_data.time.values
    time_array_counter = tqdm(time_array)
    
    
    result = np.array(list(map(retrieve_line, it.repeat(LUTpath), 
                               it.repeat(wvl1), it.repeat(wvl2), 
                               it.repeat(merged_data), 
                               time_array_counter, it.repeat(habit))))
    
    r_eff = xr.DataArray(
            data=result[:,:,0],
            dims=["time", "x"],
            coords=dict(
                time = time_array,
                x = merged_data.x.values),
            attrs=dict(
                measurement="Effective radius",
                units=r'$\mu $m',)
        )
    r_eff = r_eff.rename('r_eff')
    
    tau550 = xr.DataArray(
            data=result[:,:,1],
            dims=["time", "x"],
            coords=dict(
                time = time_array,
                x = merged_data.x.values),
            attrs=dict(
                measurement="Optical thickness at 550 nm",
                units=r'',)
        )
    tau550 = tau550.rename('tau550')
    
    return r_eff, tau550


def retrieve_line(LUTpath, wvl1, wvl2, merged_data, time, habit, CPUs=10):
    
    line = merged_data.sel(time=time)
    geometry = line.sel(wavelength = wvl1, method='nearest')
        
    Rone = np.array(line.sel(wavelength=wvl1, 
                             method='nearest').reflectivity.values)
    Rtwo = np.array(line.sel(wavelength=wvl2, 
                             method='nearest').reflectivity.values)
        
    umu = np.array(geometry.umu.values)
    phi = np.array(geometry.phi.values)
    
    with Pool(processes = CPUs) as p:
        cloud_params = p.map(retrieve_pixel, 
                             zip(it.repeat(LUTpath),                                        
                                 it.repeat(wvl1), it.repeat(wvl2), 
                                 Rone, Rtwo, umu, phi, it.repeat(habit)))
    p.close()
    
    cloud_params = np.array(list(cloud_params))

    return cloud_params


def retrieve_pixel(args):
    """Takes LUT and the two reflectivity positions to retrieve cloud parameter
    tuple for specific viewing angle."""
    LUTpath, wvl1, wvl2, Rone, Rtwo, umu, phi, habit = args
    
    LUT = read_LUT(LUTpath)
    LUT = LUT.sel(ic_habit=habit)
    
    Rone, Rtwo = [Rone], [Rtwo]
    LUTcut = LUT.sel(wvl=[wvl1,wvl2], phi=phi, umu=umu, method='nearest')
    LUTcut.coords["wvl"]=("wvl",["Rone", "Rtwo"])
    inverted = invert_data_array(LUTcut.reflectivity, 
                                 input_params=["r_eff", "tau550"], 
                                 output_dim="wvl",                       
                                 output_grid={"Rone":Rone, "Rtwo":Rtwo}, 
                                 checker=Alphachecker(alpha=0.5))
    
    r_eff = np.double(inverted.sel(input_params='r_eff').values)
    tau550 = np.double(inverted.sel(input_params='tau550').values)
    
    return r_eff, tau550