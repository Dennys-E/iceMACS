"""Generation of LUTs for RF03 bispectral retrieval. Loops over habits in 
order to not overwhelm the inversion."""

import iceMACS
import numpy as np
from iceMACS.tools import BSRLookupTable

def main():
    # Scene specifications:

    phi= np.linspace(99, 270, num=171)
    vza = np.linspace(0, np.deg2rad(20), num=40)
    umu = sorted(np.cos(vza))
    sza = [78.8]
    scene_name = "RF03"
    # Wavelengths
    wvl = [1252.17, 1598.57]
    #Cloud properties:    
    steps = 50
    r_eff_min = 5 #<-- cannot be below 5
    r_eff_max = 60 #<-- Cannot be larger than 60
    r_eff = np.linspace(r_eff_min, r_eff_max, num=steps)

    tau550_min = 0.01
    tau550_cut = 2
    tau550_max = 8

    tau550_low = np.linspace(tau550_min, tau550_cut, num=25, endpoint=False)
    tau550_high = np.linspace(tau550_cut, tau550_max, num=25)

    tau550 = np.append(tau550_low, tau550_high) 
	
   	
    cloud_altitude_grid = np.array([6, 7]) #km	Corresponds to first column in libradtran cloud file 

    template = r"""
    data_files_path /project/meteo/work/Dennys.Erdtmann/Thesis/libRadtran-2.0.4/data
    atmosphere_file /project/meteo/work/Dennys.Erdtmann/Thesis/libRadtran-2.0.4/data/atmmod/afglsw.dat

    source solar

    mol_abs_param reptran coarse

    wavelength_grid_file {{ wavelength_grid_file_path }}

    sza {{ sza }}
    phi0 {{ phi0 }}

    umu {{ umu }}
    phi {{ phi }}

    zout {{ zout }}

    brdf_cam u10 7

    ic_file 1D {{ cloud_file_path }}
    ic_modify tau550 set {{tau550}}

    ic_properties {{ ic_properties }} interpolate
    {{habit_mode}} {{ ic_habit }} {{ surface_roughness }}

    output_user wavelength uu
    output_quantity reflectivity

    verbose"""

    CPUs=8 # For parallel computing

    description = ""
    LUTpath = f"{scene_name}_ghm_LUT.nc"
    iceMACS.write_icLUT(LUTpath, template, wvl, phi, umu, sza, r_eff, tau550,
                        ['ghm'], cloud_altitude_grid, 
                        CPUs=CPUs,
                        description=description)

    print("Invert LUT...")

    sim_LUT_obj = BSRLookupTable.from_path(LUTpath)
    inverted_sim_LUT = sim_LUT_obj.inverted()
    iceMACS.save_as_netcdf(inverted_sim_LUT, f"{scene_name}_ghm_LUT_inverted.nc")

    # Define yang habits 
    column_habits = ["solid_column", "hollow_column", "column_8elements"]
    plate_habits = ["plate", "plate_5elements", "plate_10elements"]
    bullet_rosette_habits = ["solid_bullet_rosette", "hollow_bullet_rosette"]
    droxtal = ["droxtal"]

    habits = np.concatenate((column_habits, plate_habits, bullet_rosette_habits, droxtal))
    
     for habit in habits:
         LUTpath = f"{scene_name}_{habit}_rough_LUT.nc"
         iceMACS.write_icLUT(LUTpath, template, wvl, phi, umu, sza, r_eff, tau550,
                         [habit], cloud_altitude_grid, 
                         CPUs=CPUs,
                         description=description,
                         ic_properties="yang2013",
                         surface_roughness="severe")
        
    
     for habit in habits:
         LUTpath = f"{scene_name}_{habit}_smooth_LUT.nc"
         iceMACS.write_icLUT(LUTpath, template, wvl, phi, umu, sza, r_eff, tau550,
                         [habit], cloud_altitude_grid, 
                         CPUs=CPUs,
                         description=description,
                         ic_properties="yang2013",
                         surface_roughness="smooth")


    print("Produced all single habit LUTs and start inverting them...")
    for habit in habits:
        LUTpath = f"{scene_name}_{habit}_rough_LUT.nc"
        LUT = iceMACS.read_LUT(LUTpath)
        LUTds = LUT.sel(umu=np.append(LUT.umu[::2],1), phi=LUT.phi[::2])
        LUT_obj = BSRLookupTable(LUTds)
        inverted_LUT = LUT_obj.inverted()
        iceMACS.save_as_netcdf(inverted_LUT, f"{scene_name}_{habit}_rough_LUT_inverted.nc")

    for habit in habits:
        LUTpath = f"{scene_name}_{habit}_smooth_LUT.nc"
        LUT = iceMACS.read_LUT(LUTpath)
        LUTds = LUT.sel(umu=np.append(LUT.umu[::2],1), phi=LUT.phi[::2])
        LUT_obj = BSRLookupTable(LUTds)
        inverted_LUT = LUT_obj.inverted()
        iceMACS.save_as_netcdf(inverted_LUT, f"{scene_name}_{habit}_smooth_LUT_inverted.nc")


if __name__ == "__main__":
    main()
