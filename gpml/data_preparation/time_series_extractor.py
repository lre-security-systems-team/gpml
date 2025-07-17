"""
This module provides functionalities for analyzing and transforming network traffic data.

Functions:
    timeseries_transformation - Converts raw data into a time series format by grouping and aggregating specified features.
    extract_spectrum - Constructs a graph from time window data and computes its spectral properties.
    get_first_value - Retrieves the first value in a pandas series.
    timewindow_transformation - Applies time windowing techniques on the time series data, computing various spectral metrics for each window.

Contributors:
    Majed JABER

Project started on:
    11/10/2022
"""

import pandas as pd


def extract_time_series(df, stime, time_unit, features_list, sortby_list, groupby_list, aggregation_dict):
    """
    Convert raw data into a time series format by grouping and aggregating specified features.

    Parameters
    ----------
    :param df: pd.DataFrame - The input dataframe containing raw network data.
    :param features_list: list - The list of features to extract and transform.
    :param sortby_list: list - The list of columns to sort by.
    :param groupby_list: list - The list of columns to group by.
    :param aggregation_dict: dict - A dictionary specifying the aggregation operations for each feature.

    Returns
    -------
    pd.DataFrame - The transformed dataframe in time series format.
    """
    df['weight'] = 1
    df['datetime'] = pd.to_datetime(df[stime], unit=time_unit)
    # Extract columns
    ts = df[features_list]
    ts = ts.sort_values(by=sortby_list)
    # Group by columns
    ts = ts.groupby(groupby_list, as_index=False).agg(aggregation_dict)
    return ts
