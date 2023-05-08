import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import numpy as np
from .tools import *



def show_scene_geometry(view_angles, solar_positions):
    
    fig, ax = plt.subplots(figsize=(8,6))
    view_angles.vaa.plot.hist(ax=ax, bins=200, label="Viewing azimuth angle")
    view_angles.phi.plot.hist(ax=ax, bins=200, label="Relative azimuth angle")
    plt.legend()
    fig.tight_layout()
    plt.show()

    phi_min, phi_max = view_angles.phi.values.min(), view_angles.phi.values.max()
    umu_min, umu_max = view_angles.umu.values.min(), view_angles.umu.values.max()

    sza = np.mean(solar_positions.sza.values)
    print("Mean sza: ", sza)
    print("(Relative) phi range: ", phi_min, phi_max)
    print("Umu range: ", umu_min, umu_max)

    fig, ax = plt.subplots(ncols=2, figsize=(16,6))
    view_angles.umu.plot(ax=ax[0], x='time')
    umu_contour = view_angles.umu.plot.contour(ax=ax[0], x='time', cmap='coolwarm')
    view_angles.phi.plot(ax=ax[1], x='time')
    phi_contour = view_angles.phi.plot.contour(ax=ax[1], x='time', cmap='coolwarm')
    ax[0].clabel(umu_contour, umu_contour.levels, inline=True, fontsize=14)
    ax[1].clabel(phi_contour, phi_contour.levels, inline=True, fontsize=14)
    fig.tight_layout()
    plt.show()
    
    return sza, phi_min, phi_max, umu_min, umu_max


def map_AC3_scene(start_time, end_time):
    
    day = start_time.strftime("%Y%m%d")
    interval = (start_time.strftime("%Y-%m-%dT%H:%M:%S"), 
                end_time.strftime("%Y-%m-%dT%H:%M:%S"))
    
    if day != end_time.strftime("%Y%m%d"):
        print("Error: Start and end time have to be on the same day!")

    print("Load bahamas data...")
    source_folder='/archive/meteo/ac3/'+day+'/'
    nas_day_path = source_folder+"nas/AC3_HALO_BAHAMAS-SPECMACS-100Hz-final_"+day+"a.nc"
    nas_day = read_LUT(nas_day_path)
    nas_scene = nas_day.sel(time=slice(*interval))
    
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

    
    return 

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


def show_LUT(LUT, wvl1, wvl2, save_under=None):
    """Takes a LUT for one perspective and returns an overview of plots to display the data. Does not retrieve any values."""
    
    fig, ax = plt.subplots(figsize=(16,16))
    LUTcut1 = LUT.sel(wvl=wvl1)
    LUTcut2 = LUT.sel(wvl=wvl2)

    for r_eff in LUT.coords['r_eff'].values:
        ax.plot(LUTcut1.sel(r_eff=r_eff).reflectivity.to_numpy(), LUTcut2.sel(r_eff=r_eff).reflectivity.to_numpy(),
                linewidth=1, label=np.round(r_eff, 2))

    for tau550 in LUT.coords['tau550'].values:
        ax.plot(LUTcut1.sel(tau550=tau550).reflectivity.to_numpy(), LUTcut2.sel(tau550=tau550).reflectivity.to_numpy(),
                "--", color="black",
                linewidth=0.7)
        
        x = np.max(LUTcut1.sel(tau550=tau550).reflectivity.to_numpy())
        y = np.max(LUTcut2.sel(tau550=tau550).reflectivity.to_numpy())
        if tau550<15 and tau550>0:
            plt.text(x,y, r"$\tau=$"+str(tau550), fontsize=11)

    ax.set_xlabel("Reflectivity at "+str(wvl1)+"nm")
    ax.set_ylabel("Reflectivity at "+str(wvl2)+"nm")
    ax.legend(title=r"Effective radius [$\mu$m]", ncols=3)
    
    if save_under is not None:
        plt.savefig(save_under)
        
    plt.show()
    return