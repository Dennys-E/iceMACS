# iceMACS
Collection of tools to calibrate and manage SWIR, VNIR and pol cam data from specMACS, as well as retrieve ice cloud optical properties and habit estimates using a bispectral Nakajima-King retrieval and angular retrieval. 

## Todos
* Unify LUT generators, preferably into one single function.
* Add a git submodules functionality
* Include functions for polarized retrieval
* Make the allocation of cloud properties more stable
* Add halo_height as a free coordinate

- [Structure](#-structure)
- [Usage](#-usage)

## Structure
The submodules in the `iceMACS` package are organized as follows:

* The `paths` submodule defines global paths specific to you system. Adapt before usage.
* `conveniences` contains functions that are non-essential to the retrieval but
are compatible with other functions and sometimes called by the `tools` 
submodule. For example, loading data from the A(C)³ archive directory, plotting, reading and writing NetCDF files etc.
* `tools` contains functions to interpret camera data and LUT data and add new variables, 
such as reflectivities, ice index and relative view angles. The updated `PixelInterpolator` class is defined here.
* Rest to be determined...


## Usage

This readme contains the documentation of the most important functions contained in iceMACS but is not exhaustive. There are additional functions that might be useful for you. 

### SWIR bad pixel interpolation
Many (AC)³ scenes are relatively dark, with a high solar zenith angle and low cirrus radiance values. Some pixels are shown to be unreliable under these conditions. The `PixelInterpolator` class finds these pixels and interpolates for the entire scene. Additionally, interpolation over invalid pixel from the bad pixel list is performed, analogous to the `runmacs` `BadPixelFixer`.
Initiate with loaded SWIR dataset, containing the variables `radiance`and `valid` access "badness" signal with

```python
from iceMACS.tools import PixelInterpolator
interp = PixelInterpolator(swir_ds, window=3)
interp.show_signals()
```

The `window` variable sets the moving average frame size. Choose a fitting cutoff value for each plotted wavelength and pass as `list`, e.g.

```python
interp.add_cutoffs([4, 1.2])
```

Adjust cutoff as needed and apply filter with 

```python
filtered_radiance = interp.filtered_radiance(with_bpl=True)
```

or

```python
filtered_radiance = interp.interpolated_radiance(with_bpl=True)
```

where also interpolating pixels from bad pixel list is the default. 

### Data formatting
The `SceneInterpreter` class takes calibrated loaded SWIR and VNIR datasets, as view angles and solar position datasets and facilitates computation of variables that need to be passed to the `LUTGenerator` functions. Initiate with

```python
from iceMACS.tools import SceneInterpreter
scene = SceneInterpreter(swir_scene, view_angles, solar_positions)
```
and get summarized scene geometry for LUT reference with 
``` python
scene.overview()
```
Get relative view angles, reflectivity, ice index etc. variables with

```python
scene.reflectivity()
scene.umu()
scene.phi()
```
or get summarized scene information with 

```python
scene.merged_data()
```

The effective radius and optical thickness are returned by the SceneInterpreter as an xarray DataSet through

```python
scene.cloud_properties_fast_BSR(...)
```
More on that in the section about the bispectral retrieval. 

### LUT generation

The simulation results for various viewing geometries, ice crystal habits and bulk optical properties are obtained from calling uvspec and storing the results in a `.nc` file. Call with 

```python
iceMACS.write_icLUT(LUTpath, input_file_template, wvl_array, phi_array, umu_array, sza_array, r_eff_array, tau550_array, ic_habit_array, cloud_altitude_grid, phi0=0, cloud_top_distance=1, ic_properties="baum_v36", surface_roughness="severe", CPUs=8, description="")
```

with the desired coordinate arrays. The function is defined in the `icLUTgenerator.py` module. The `CPUs` keyword specifies the number of processor units that should be used during the parallel calling of uvspec. 

### LUT handling and inversion

The LUT dataset containing the simulation results can be passed to the `BSRLookupTable` class. To initiate call

```python
from iceMACS.tools import BSRLookupTable
LUT = BSRLookupTable(LUT_ds)
```

or from path

```python
LUT = BSRLookupTable.from_path('LUT_ds_path')
```

The dataset has to contain the two wavelengths intended to be used in the retrieval. Original data is saved in 'LUT.dataset' You can visualize the splitting of reflectivities with 

```python
LUT.display_nadir()
```

Similarly, a different viewing geometry and solar position can be selected with 

```python
LUT.display_at(self, sza, umu, phi, ic_habit)
```

The `BSRLookupTable` class provides an automated lookup table inversion based on Paul's [`luti`](https://github.com/Ockenfuss/luti) package. Call

```python
invertedLUT = LUT.inverted(num=200, alpha=4)
```
where `num` is the sample number within the relevant reflectivity range and `alpha` is a parameter used to define the convex hull. `alpha=4.` has been found to work well for bispectral cloud lookup tables.

### Bispectral retrieval

The inverted lookup table contains reflectivities as coordinates and the cloud parameters in the variable `input_params`. Without any further formatting, you can pass the inverted dataset to the `SceneInterpreter` instance you want to retrieve by calling

```python
scene.cloud_properties_fast_BSR(invertedLUT, LUT.wvl1, LUT.wvl2, 
                                LUT.Rone_name, LUT.Rtwo_name, 
                                umu_bins=20, phi_bins=50, interpolate=True)
```

Here, `LUT` is the `BSRLookupTable` instance that produced the inverted dataset. `interpolate` chooses the method by which the simulations are cut to pixel geometries. `True` interpolates between simulated viewing geometries while `False` chooses the closest existing coordinate.

### Handling of polarized LUT data

Similar to the `BSRLookupTable` class, Stokes parameter data stored in LUT format can be handled with the `PolLookupTable` class. Functionalities include computation of polarization-specific quantities such as the polarized reflectivity and DOLP. Additionally, color channels of pol cameras can be simulated with 

```python
polLUT.polarized_reflectivity_with_srfs(calibration_file, color='red')
```

or similarly named functions. Inspection of scattering behavior along the principal plane for LUTs containing relative azimuth coordinates 0° and 180° is facilitated by

```python
polLUT.over_theta_in_pp()
```

which automatically transforms `umu` to scattering angle `theta`.

### Angular habit retrieval

### Additional functionalities
