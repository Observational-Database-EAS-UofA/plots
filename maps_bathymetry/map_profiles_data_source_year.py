import cartopy.crs as ccrs
import cartopy.feature as cfeature

# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.path as mpath
import os
import xarray as xr
import pandas as pd


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return np.nan


def plot_points_by_data_source(lon_list, lat_list, datasets_str):
    # convert string to numeric values
    unique_values = np.unique(datasets_str)
    value_map = {val: i for i, val in enumerate(unique_values)}
    datasets = np.array([value_map[val] for val in datasets_str])

    projection = ccrs.NorthPolarStereo()
    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes([0.1, 0.1, 0.8, 0.8], projection=projection)
    ax.set_extent(
        [-180, 180, 50, 90],
        crs=ccrs.PlateCarree(),
    )
    # ax.set_global()

    bathym = cfeature.NaturalEarthFeature(
        name="bathymetry_J_1000", scale="10m", category="physical"
    )

    # features
    ax.add_feature(
        bathym,
        facecolor="none",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.4,
        zorder=0,
        # linestyle="dashed",
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, edgecolor="red")

    # create the map
    scatter = ax.scatter(
        lon_list,
        lat_list,
        c=datasets,
        transform=ccrs.PlateCarree(),
        s=1,
        cmap="viridis",
    )
    # ax.contour(X, Y, Z, transform=ccrs.NorthPolarStereo(), colors='gray', alpha=0.5,)

    # legend
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=val,
            markerfacecolor=scatter.cmap(scatter.norm(value_map[val])),
        )
        for val in unique_values
    ]
    ax.legend(handles=legend_elements, loc="upper right", title="Dataset Name")

    ax.coastlines()
    plt.savefig(f"imgs/points_data_source_{time.time()}.png")


def plot_points_by_year(lon_list, lat_list, years):
    projection = ccrs.NorthPolarStereo()
    fig = plt.figure(figsize=(10, 10))

    ax = plt.axes([0.1, 0.1, 0.8, 0.8], projection=projection)
    ax.set_extent(
        [-180, 180, 50, 90],
        crs=ccrs.PlateCarree(),
    )
    # create the map
    scatter = ax.scatter(
        lon_list, lat_list, c=years, transform=ccrs.PlateCarree(), s=1, cmap="viridis"
    )

    # add bathymetry
    bathym = cfeature.NaturalEarthFeature(
        name="bathymetry_J_1000", scale="10m", category="physical"
    )

    # features
    ax.add_feature(
        bathym,
        facecolor="none",
        edgecolor="black",
        linewidth=0.5,
        alpha=0.4,
        zorder=0,
        # linestyle="dashed",
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.add_feature(cfeature.COASTLINE, edgecolor="red")

    # display map
    ax.coastlines()

    # color bar
    cbar = plt.colorbar(scatter, ax=ax, orientation="vertical", label="Year")

    plt.savefig(f"imgs/points_by_year_{time.time()}.png")
    # plt.show()


def plot_points_by_data_source_good_data(ncfiles_path):
    projection = ccrs.NorthPolarStereo()
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(20, 10),
        sharex=True,
        sharey=True,
        subplot_kw={"projection": projection},
        gridspec_kw={"width_ratios": [1, 1]}
    )

    for ax in axs:
        # ax = plt.axes([0.1, 0.1, 0.8, 0.8], projection=projection)
        ax.set_extent(
            [-180, 180, 50, 90],
            crs=ccrs.PlateCarree(),
        )
        # add bathymetry
        bathym = cfeature.NaturalEarthFeature(
            name="bathymetry_J_1000", scale="10m", category="physical"
        )
        # features
        ax.add_feature(
            bathym,
            facecolor="none",
            edgecolor="black",
            linewidth=0.5,
            alpha=0.4,
            zorder=0,
            # linestyle="dashed",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray")
        ax.add_feature(cfeature.COASTLINE, edgecolor="red")

        # display map
        ax.coastlines()
    # axs = axs.ravel()
    data_list = []

    for file in os.listdir(ncfiles_path):
        ds = xr.open_dataset(ncfiles_path + file)
        data = {
            "lon": ds["lon"].values,
            "lat": ds["lat"].values,
            "lonlat_flag": ds["lonlat_flag"].values,
            "year": np.array(
                [convert_to_int(date[:4]) for date in ds["datestr"].values]
            ),
        }

        data_list.append(data)
    df = pd.DataFrame()
    for data in data_list:
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    # create the map
    scatter = axs[0].scatter(
        df["lon"],
        df["lat"],
        c=df["year"],
        transform=ccrs.PlateCarree(),
        s=1,
        cmap="viridis",
    )
    filtered_df = df[(df["lonlat_flag"] == 1)]
    scatter = axs[1].scatter(
        filtered_df["lon"],
        filtered_df["lat"],
        c=filtered_df["year"],
        transform=ccrs.PlateCarree(),
        s=1,
        cmap="viridis",
    )

    # color bar
    cbar = plt.colorbar(scatter, ax=axs, orientation="vertical", label="Year")
    axs[0].set_title("Artic region - bad data")
    axs[1].set_title("Artic region - good data")

    plt.savefig(f"imgs/points_by_year_bad_good_data.png")
    # plt.show()
    plt.clf()


# main
def main():
    # plotting from all datasets
    merged_file = xr.open_dataset("../../merged_file.nc")
    datestr = merged_file["datestr"].values
    lon_list = merged_file["lon"].values
    lat_list = merged_file["lat"].values
    datasets_str = merged_file["dataset_name"].values
    years = np.array([convert_to_int(year[:4]) for year in datestr])
    # plot_points_by_data_source(lon_list, lat_list, datasets_str)
    # plot_points_by_year(lon_list, lat_list, years)

    # plotting specific datasets
    meds_datasets = "/mnt/storage6/caio/AW_CAA/CTD_DATA/MEDS_2021/ncfiles_id/"
    plot_points_by_data_source_good_data(meds_datasets)


main()
