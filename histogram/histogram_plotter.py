import xarray as xr
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import calendar


def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return np.nan


# needs update
def plot_each_dataset_separately(dataset_years):
    for dataset, d_years in dataset_years.items():
        ax = plt.axes()
        x, y = np.unique(d_years, return_counts=True)
        ax.bar(x, y, label=dataset)
        ax.legend()

        if not os.path.isdir("per_dataset"):
            os.mkdir("per_dataset")
        plt.savefig(f"per_dataset/histogram_{dataset}")
        plt.clf()


def plot_multiple_histograms(dataset_years_df: pd.DataFrame):
    datasets = dataset_years_df.columns[1:]
    num_datasets = len(datasets)
    num_cols = 3
    num_rows = math.ceil(num_datasets / num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows), sharey=True)
    axs = axs.ravel()

    for i, dataset in enumerate(datasets):
        years = dataset_years_df['Year']
        counts = dataset_years_df[dataset]
        ax = axs[i]
        ax.bar(years, counts, label=dataset)
        ax.legend()
        ax.set_title(f'{dataset}')

    for j in range(num_datasets, num_rows * num_cols):
        axs[j].axis('off')

    plt.xlabel("Year")
    plt.ylabel("Profiles Count")
    plt.tight_layout()
    plt.savefig("imgs/multiple_histograms.png")
    plt.clf()
    plt.show()



def plot_histogram_year(year_var_df: pd.DataFrame, profiles: int, var: str):
    fig, ax = plt.subplots(figsize=(10, 9))
    all_years = year_var_df.iloc[:, 0]
    bottom = np.zeros(len(all_years))
    for i in range(1, len(year_var_df.columns)):
        height = year_var_df.iloc[:, i]
        ax.bar(
            all_years,
            height,
            bottom=bottom,
            label=year_var_df.columns[i],
        )
        bottom += height

    if var == "instrument":
        ax.legend(loc="center left")
    else:
        ax.legend()
    ax.set_title(f"Number of profiles: {profiles}")
    plt.xlabel("Year")
    plt.ylabel("Profiles Count")
    plt.savefig(f"imgs/interannual_histogram_{var}.png")
    plt.clf()
    # plt.show()


def plot_histogram_month(month_var_df: pd.DataFrame, profiles: int, var: str):
    fig, ax = plt.subplots(figsize=(10, 9))
    month_names = [calendar.month_name[i] for i in range(1, 13)]
    all_months = np.arange(1, 13)

    bottom = np.zeros(12)
    for i in range(1, len(month_var_df.columns)):
        height = month_var_df.iloc[:, i]
        ax.bar(
            all_months,
            height,
            bottom=bottom,
            label=month_var_df.columns[i],
        )
        bottom += height

    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(
        month_names,
        rotation=45,
    )

    if var == "instrument":
        ax.legend(loc="center left")
    else:
        ax.legend()
    ax.set_title(f"Number of profiles: {profiles}")
    plt.ylabel("Profiles Count")
    plt.savefig(f"imgs/seasonal_histogram_{var}.png")
    plt.clf()
    # plt.show()


### From a dict, create and return a dataframe where the rows are the years and the columns the datasets.
### Each point of the dataframe will have the quantity of profiles each dataset has in the specific year
def dict_to_dataframe(column_list, row_list, row_type):
    unique_items = np.unique(row_list).astype(int)
    unique_items = unique_items[:-1]
    data_dict = {attr: [0] * len(unique_items) for attr in np.unique(column_list)}
    for key, value in zip(column_list, row_list):
        (i,) = np.where(unique_items == value)
        if len(i) != 0:
            data_dict[key][i[0]] += 1

    df = pd.DataFrame(data_dict)
    df.insert(0, row_type, unique_items, allow_duplicates=True)

    return df


def main():
    merged_file = xr.open_dataset("../../merged_file.nc")
    number_profiles = merged_file.dims["profile"]
    dataset_name_list = merged_file["dataset_name"].values
    instrument_type_list = merged_file["instrument_type"].values
    date_list = merged_file["datestr"].values
    year_list = np.array([convert_to_int(year[:4]) for year in date_list])
    month_list = np.array([convert_to_int(full_date[5:7]) for full_date in date_list])

    dataset_years_df = dict_to_dataframe(dataset_name_list, year_list, row_type="Year")
    dataset_months_df = dict_to_dataframe(dataset_name_list, month_list, row_type="Month")

    # plot_histogram_year(dataset_years_df, number_profiles, "dataset")
    # plot_histogram_month(dataset_months_df, number_profiles, "dataset")
    plot_multiple_histograms(dataset_years_df)


main()
