# iceMACS
Collection of tools to calibrate and manage SWIR and VNIR data from the specMACS system, as well as retrieve ice cloud optical properties using a bispectral Nakajima-King retrieval. 

## Todos
* Change string logs to modern f-string syntax
* Change <code>open()</code> calls when reading files to <code>with open()</code> in order to ensure files are closed when exception occurs.
* Instead of 
* Find a way to avoid * imports in <code>__init__</code> file
* Unify LUT generators, preferably into one single function.
* Restructure submodules to avoid confusion. Add classes.
* Add examples
* Complete documentation under usage
* Add a git submodules functionality
* Find better way to organize paths
* Try to replace \ line continuation with brackets
* Add documentation for SceneInterpreter

## Structure
The submodules in the `iceMACS` package are organized as follows:

* The `paths` submodule defines global paths specific to you system. Adapt before usage.
* `conveniences` contains functions that are non-essential to the retrieval but
are compatible with other functions and sometimes called by the `tools` 
submodule.
* `tools` contains functions to interpret camera data and add new variables, 
such as reflectivities, ice index and relative view angles. The updated `PixelInterpolator` class is defined here.
* Rest to be determined...


## Usage

### SWIR bad pixel interpolation
Many A(C)³ scenes are relatively dark, with a high solar zenith angle and low cirrus rediance values. Some pixels are shown to be unreliable under these conditions. The `PixelInterpolator` class finds these pixels and interpolates for the entire scene. Additionally, interpolation over invalid pixel from the bad pixel list is performed, analogous to the `runmacs` `BadPixelFixer`.
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
filtered_radiance = interp.get_filtered_radiance(with_bpl=True)
```

where also interpolating pixels from bad pixel list is the default. 

### Bispectral retrieval (BSR)

### Habit detection

### Additional functionalities
