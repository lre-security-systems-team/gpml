"""
This module provides functionalities for computing various spectral metrics used in dynamic graph analysis.

It includes functions to compute connectedness, flooding, wiriness, and asymmetry based on the spectral properties of graphs.

Functions:
    connectedness - Computes the connectedness metric based on the spectral properties and number of components.
    flooding - Computes the flooding metric based on the algebraic connectivity of the graph.
    wireness - Computes the wireness metric based on the maximum degree of the graph.
    asymmetry - Computes the asymmetry metric based on the differences in the spectrum.

Contributors:
    Majed JABER

Project started on:
    [Project Start Date]
"""

import math
import statistics
import numpy as np
import pandas as pd
import networkx as nx


# from gpml.data_preparation.time_series_extractor import time_series_extractor


def connectedness(num_components):
    """
    Compute the connectedness metric based on the spectral properties and number of components.

    Parameters
    ----------
    :param spec: array-like - The spectral properties of the graph.
    :param num_components: int - The number of components in the graph.

    Returns
    -------
    float - The connectedness metric.
    """
    M1 = math.exp(1 / num_components) / math.exp(1)
    return M1


def flooding(spec, num_components):
    """
    Compute the flooding metric based on the algebraic connectivity of the graph.

    Parameters
    ----------
    :param spec: array-like - The spectral properties of the graph.
    :param num_components: int - The number of components in the graph.

    Returns
    -------
    float - The flooding metric.
    """
    algebraic_conn = spec[num_components:num_components + num_components]
    mean = statistics.mean(algebraic_conn)
    return np.exp(mean) - 1.0


def wireness(spec, num_components):
    """
    Compute the wireness metric based on the maximum degree of the graph.

    Parameters
    ----------
    :param spec: array-like - The spectral properties of the graph.
    :param num_components: int - The number of components in the graph.

    Returns
    -------
    float - The wireness metric.
    """
    max_deg = spec[len(spec) - num_components:len(spec)]
    mean = statistics.mean(max_deg)
    return mean


def asymmetry(spec):
    """
    Compute the asymmetry metric based on the differences in the spectrum.

    Parameters
    ----------
    :param spec: array-like - The spectral properties of the graph.

    Returns
    -------
    int - The asymmetry metric.
    """
    _x = np.arange(len(spec))
    _dx = np.diff(_x)
    count = 0
    for k in range(len(_dx) - 1):
        if _dx[k] > math.pow(10, -12):
            count += 1
    return count


def extract_spectrum(time_window_df, src, dst, edge_weight):
    """
    Construct a graph from time window data and compute its spectral properties.

    Parameters
    ----------
    :param tw: pd.DataFrame - The dataframe containing the time window data.
    :param src: str - The name of the source column in the dataframe.
    :param dst: str - The name of the destination column in the dataframe.
    :param edge_weight: str - The name of the edge weight column in the dataframe.

    Returns
    -------
    tuple - A tuple containing the graph, its Laplacian matrix, and its sorted eigenvalues.
    """
    # Arbitrary threshold for large dataframe
    if len(time_window_df) > 500000:
        return "graph is too large to process efficiently."

    g = nx.Graph()

    for _, row in time_window_df.iterrows():
        src_ip = row[src]
        dst_ip = row[dst]
        weight = row[edge_weight]

        if g.has_edge(src_ip, dst_ip):
            g[src_ip][dst_ip]['weight'] += weight
        elif g.has_edge(dst_ip, src_ip):
            g[dst_ip][src_ip]['weight'] += weight
        else:
            g.add_edge(src_ip, dst_ip, weight=weight)
    l = nx.laplacian_matrix(g, weight='weight').todense()
    ev = np.linalg.eigvalsh(l)
    return g, l, np.sort(ev)


def get_first_value(series):
    """
    Retrieve the first value in a pandas series.

    Parameters
    ----------
    :param series: pd.Series - The input pandas series.

    Returns
    -------
    The first value in the series.
    """
    return series.iloc[0]


def extract_spectral_metrics(ts, stime, saddr, daddr, pkts='', bytes_size='', rate='', lbl_category='',
                               src_pkts='', dst_pkts='', src_bytes='', dst_bytes='', duration=''):
    """
    Apply time windowing techniques on the time series data, computing various spectral metrics for each window.

    Parameters
    ----------
    :param ts: pd.DataFrame - The input time series dataframe.
    :param stime: str - The name of the start time column in the dataframe.
    :param saddr: str - The name of the source address column in the dataframe.
    :param daddr: str - The name of the destination address column in the dataframe.
    :param pkts: str - The name of the packets column in the dataframe.
    :param bytes: str - The name of the bytes column in the dataframe.
    :param rate: str - The name of the rate column in the dataframe.
    :param lbl_category: str - The name of the label category column in the dataframe.
    :param src_pkts: str - Optional. The name of the source packets column. Defaults to an empty string if not provided.
    :param dst_pkts: str - Optional. The name of the destination packets column. Defaults to an empty string if not provided.
    :param src_bytes: str - Optional. The name of the source bytes column. Defaults to an empty string if not provided.
    :param dst_bytes: str - Optional. The name of the destination bytes column. Defaults to an empty string if not provided.
    :param duration: str - Optional. The name of the destination duration column. Defaults to an empty string if not provided.

    Returns
    -------
    pd.DataFrame - The transformed dataframe with computed spectral metrics for each time window.

    """
    df = ts.copy()

    if pkts not in df.columns and src_pkts in df.columns and dst_pkts in df.columns:
        df[pkts] = df[src_pkts] + df[dst_pkts]

    if bytes_size not in df.columns and src_bytes in df.columns and dst_bytes in df.columns:
        df[bytes_size] = df[src_bytes] + df[dst_bytes]

    if rate not in df.columns and pkts in df.columns and duration in df.columns:
        df[rate] = df[pkts] / df[duration]

    df_topredict = pd.DataFrame()

    df['next_stime'] = df[stime] + 60
    df['next_datetime'] = pd.to_datetime(df['next_stime'], unit='s')

    count_one_min_windows = 0

    for _, row in df.iterrows():
        attack_label = 0
        current_timestamp = row[stime]
        next_timestamp = row['next_stime']
        mask = (df[stime] <= next_timestamp) & (df[stime] >= current_timestamp)
        time_window = df[mask]
        unique_time_window = time_window.iloc[:, 0].unique()

        if len(unique_time_window) >= 2:
            count_one_min_windows += 1

            # ts1
            ts1 = unique_time_window[len(unique_time_window) - int(len(unique_time_window) / 2)]
            mask = (time_window[stime] <= ts1) & (time_window[stime] >= current_timestamp)
            sub_time_window = time_window[mask]

            g1_pkts, _, ev1_pkts = extract_spectrum(sub_time_window, saddr, daddr, pkts)
            num_components_pkts = nx.number_connected_components(g1_pkts)

            t1_m1_pkts = connectedness(num_components_pkts)
            t1_m2_pkts = flooding(ev1_pkts, num_components_pkts)
            t1_m3_pkts = wireness(ev1_pkts, num_components_pkts)
            t1_m4_pkts = asymmetry(ev1_pkts[1:])

            g1_bytes, _, ev1_bytes = extract_spectrum(sub_time_window, saddr, daddr, bytes_size)
            num_components_bytes = nx.number_connected_components(g1_bytes)

            t1_m1_bytes = connectedness(num_components_bytes)
            t1_m2_bytes = flooding(ev1_bytes, num_components_bytes)
            t1_m3_bytes = wireness(ev1_bytes, num_components_bytes)
            t1_m4_bytes = asymmetry(ev1_bytes[1:])

            g1_rate, _, ev1_rate = extract_spectrum(sub_time_window, saddr, daddr, rate)
            num_components_rate = nx.number_connected_components(g1_rate)

            t1_m1_rate = connectedness(num_components_rate)
            t1_m2_rate = flooding(ev1_rate, num_components_rate)
            t1_m3_rate = wireness(ev1_rate, num_components_rate)
            t1_m4_rate = asymmetry(ev1_rate[1:])

            # ts2
            ts2 = unique_time_window[len(unique_time_window) - 1]
            mask = (time_window[stime] <= ts2) & (time_window[stime] >= current_timestamp)
            sub_time_window = time_window[mask]

            if (sub_time_window.tail(1)[lbl_category] != 0).any():
                attack_label = sub_time_window.tail(1)[lbl_category].item()

            g2_pkts, _, ev2_pkts = extract_spectrum(sub_time_window, saddr, daddr, pkts)
            num_components_pkts = nx.number_connected_components(g2_pkts)

            t2_m1_pkts = connectedness(num_components_pkts)
            t2_m2_pkts = flooding(ev2_pkts, num_components_pkts)
            t2_m3_pkts = wireness(ev2_pkts, num_components_pkts)
            t2_m4_pkts = asymmetry(ev2_pkts[1:])

            g2_bytes, _, ev2_bytes = extract_spectrum(sub_time_window, saddr, daddr, bytes_size)
            num_components_bytes = nx.number_connected_components(g2_bytes)

            t2_m1_bytes = connectedness(num_components_bytes)
            t2_m2_bytes = flooding(ev2_bytes, num_components_bytes)
            t2_m3_bytes = wireness(ev2_bytes, num_components_bytes)
            t2_m4_bytes = asymmetry(ev2_bytes[1:])

            g2_rate, _, ev2_rate = extract_spectrum(sub_time_window, saddr, daddr, rate)
            num_components_rate = nx.number_connected_components(g2_rate)

            t2_m1_rate = connectedness(num_components_rate)
            t2_m2_rate = flooding(ev2_rate, num_components_rate)
            t2_m3_rate = wireness(ev2_rate, num_components_rate)
            t2_m4_rate = asymmetry(ev2_rate[1:])

            agg_series = sub_time_window.agg({
                'pkts': 'sum', 'bytes': 'sum', 'attack': get_first_value,
                'rate': 'mean', 'dur': 'mean', 'mean': 'mean', 'sum': 'mean', 'min': 'mean', 'max': 'mean',
                'spkts': 'mean', 'srate': 'mean'
                , 'drate': 'mean', 'weight': 'sum'
            })
            # Convert the series to a single-row DataFrame
            agg_row = agg_series.to_frame().transpose()
            agg_row['attack'] = attack_label

            agg_row['ts1_m1_pkts'] = [t1_m1_pkts]
            agg_row['ts2_m1_pkts'] = [t2_m1_pkts]
            agg_row['ts1_m1_bytes'] = [t1_m1_bytes]
            agg_row['ts2_m1_bytes'] = [t2_m1_bytes]
            agg_row['ts1_m1_rate'] = [t1_m1_rate]
            agg_row['ts2_m1_rate'] = [t2_m1_rate]

            agg_row['ts1_m2_pkts'] = [t1_m2_pkts]
            agg_row['ts2_m2_pkts'] = [t2_m2_pkts]
            agg_row['ts1_m2_bytes'] = [t1_m2_bytes]
            agg_row['ts2_m2_bytes'] = [t2_m2_bytes]
            agg_row['ts1_m2_rate'] = [t1_m2_rate]
            agg_row['ts2_m2_rate'] = [t2_m2_rate]

            agg_row['ts1_m3_pkts'] = [t1_m3_pkts]
            agg_row['ts2_m3_pkts'] = [t2_m3_pkts]
            agg_row['ts1_m3_bytes'] = [t1_m3_bytes]
            agg_row['ts2_m3_bytes'] = [t2_m3_bytes]
            agg_row['ts1_m3_rate'] = [t1_m3_rate]
            agg_row['ts2_m3_rate'] = [t2_m3_rate]

            agg_row['ts1_m4_pkts'] = [t1_m4_pkts]
            agg_row['ts2_m4_pkts'] = [t2_m4_pkts]
            agg_row['ts1_m4_bytes'] = [t1_m4_bytes]
            agg_row['ts2_m4_bytes'] = [t2_m4_bytes]
            agg_row['ts1_m4_rate'] = [t1_m4_rate]
            agg_row['ts2_m4_rate'] = [t2_m4_rate]

            # Concatenate the list of dictionaries into a new DataFrame
            df_topredict = pd.concat([df_topredict, agg_row], ignore_index=True)
    return df_topredict
