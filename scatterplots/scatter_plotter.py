import xarray as xr
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return np.nan


def plot_var_depth_one_color(datasets, var):
    fig, ax = plt.subplots()

    data = {
        var: [],
        "depth": [],
    }
    for ds in datasets:
        data[var].extend(ds[var].values)
        data["depth"].extend(ds["depth"].values)

    df = pd.DataFrame(data)
    df = df[df[var] > -20]

    ax.scatter(df[var], df["depth"], s=0.5)
    if var == "temp":
        plt.xlabel("Temperature (ᵒC)")
        plt.title("Temperature profiles in artic region")
    elif var == "psal":
        plt.xlabel("Salinity (PSU)")
        plt.title("Salinity profiles in artic region")

    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    plt.savefig(f"imgs/{var}_depth.png")
    plt.clf()


def plot_var_depth_year(datasets, var):
    fig, ax = plt.subplots()

    data = {
        var: [],
        "depth": [],
        "year": [],
    }

    for ds in datasets:
        data[var].extend(ds[var].values)
        data["depth"].extend(ds["depth"].values)

        ### transform date into the same dimension as 'obs' dimension
        _, count = np.unique(ds["parent_index"].values, return_counts=True)
        for c, date in zip(count, ds["datestr"].values):
            data["year"].extend([convert_to_int(date[:4])] * c)

    df = pd.DataFrame(data)
    df = df[df[var] > -20]
    df = df[df["depth"] < 500]

    sc = ax.scatter(df[var], df["depth"], s=0.5, c=df["year"], cmap="viridis")

    plt.colorbar(sc, ax=ax, label="year")
    if var == "temp":
        plt.xlabel("Temperature (ᵒC)")
        plt.title("Temperature profiles in artic region")
    elif var == "psal":
        plt.xlabel("Salinity (PSU)")
        plt.title("Salinity profiles in artic region")
    plt.ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    plt.savefig(f"imgs/{var}_depth_year.png")
    plt.clf()


def plot_var_raw_vs_good(datasets: list[xr.Dataset], var):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 9), sharey=True)
    axs = axs.ravel()

    all_data = []
    for ds in datasets:
        data = {
            var: ds[var].values,
            var + "_flag": ds[var + "_flag"].values,
            "depth": ds["depth"].values,
            "depth_flag": ds["depth_flag"].values,
            "year": np.repeat(np.array([convert_to_int(date[:4]) for date in ds["datestr"].values]),
                              np.unique(ds["parent_index"].values, return_counts=True)[1])
        }

        all_data.append(data)

    df = pd.DataFrame()
    for data in all_data:
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

    df = df[df["depth"] < 2000]
    sc_bad_data = axs[0].scatter(df[var], df["depth"], s=0.5, c=df["year"], cmap="viridis")
    df_filtered = df[(df[var + "_flag"] == 1) & (df["depth_flag"] == 1)]
    sc_good_data = axs[1].scatter(df_filtered[var], df_filtered["depth"], s=0.5, c=df_filtered["year"], cmap="viridis")

    plt.colorbar(sc_good_data, ax=axs[1], label="year")
    if var == "temp":
        axs[0].set_xlabel("Temperature (ᵒC)")
        axs[1].set_xlabel("Temperature (ᵒC)")
        axs[0].set_title("Temperature profiles in the Arctic region - MEDS 2021 - Bad data")
        axs[1].set_title("Temperature profiles in the Arctic region - MEDS 2021 - Good data")
    elif var == "psal":
        axs[0].set_xlabel("Salinity (PSU)")
        axs[1].set_xlabel("Salinity (PSU)")
        axs[0].set_title("Salinity profiles in the Arctic region - MEDS 2021 - Bad data")
        axs[1].set_title("Salinity profiles in the Arctic region - MEDS 2021 - Good data")
    axs[0].set_ylabel("Depth (m)")
    # axs[1].set_ylabel("Depth (m)")
    plt.gca().invert_yaxis()
    # plt.show()
    plt.savefig(f"imgs/{var}_profiles_good_vs_bad_data.png")
    plt.clf()



def main():
    # netcdf_file_path = "/mnt/storage6/caio/AW_CAA/CTD_DATA/DFO_IOS_2022/ncfiles_id/"
    netcdf_file_path = "/mnt/storage6/caio/AW_CAA/CTD_DATA/MEDS_2021/ncfiles_id/"
    datasets = [
        xr.open_dataset(netcdf_file_path + file)
        for file in os.listdir(netcdf_file_path)
    ]

    # plot_var_depth_one_color(datasets, "temp")
    # plot_var_depth_one_color(datasets, "psal")
    plot_var_raw_vs_good(datasets, "temp")
    plot_var_raw_vs_good(datasets, "psal")
    # plot_var_depth_year(datasets, "temp")
    # plot_var_depth_year(datasets, "psal")
    


main()
