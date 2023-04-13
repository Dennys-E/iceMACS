import cartopy.crs as ccrs
import cartopy
import pvlib
import matplotlib.pyplot as plt
import xarray as xr

def map_scene(nas_day, interval):
    
    nas_scene = nas_day.sel(time=slice(*interval))

    day_trajectory = nas_day.lon.values, nas_day.lat.values
    scene_trajectory = nas_scene.lon.values, nas_scene.lat.values

    fig, ax = plt.subplots(nrows=1,ncols=1,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            figsize=(16,6))

    # Remove frame 
    plt.tick_params(axis='x', which='both', bottom=False,
                    top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False,
                    left=False, labelleft=False)
    for pos in ['right', 'top', 'bottom', 'left']:
        plt.gca().spines[pos].set_visible(False)

    #for ax in axes:

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, alpha=0.4)
    ax.add_feature(cartopy.feature.COASTLINE,linewidth=0.6)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':',linewidth=0.3)
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    ax.plot(*day_trajectory,
             color='tab:orange', linestyle='--', linewidth=0.6,
             transform=ccrs.PlateCarree())

    ax.plot(*scene_trajectory,
            color='red', linewidth=2.5,
            transform=ccrs.PlateCarree())

    plt.show()
    
    return