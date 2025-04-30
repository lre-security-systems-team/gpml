# Graph Processing for Machine Learning - GPML Library

This module provides functionalities for analyzing and
transforming network traffic data through graph representations.

## Community Metrics

### Flow diagram - Communities

![Alt text](images/community_graph_flow.png 'community graph analysis process')

### Graph Definition - Communities

For graph analysis with community metrics, connectivity graphs represent
network relationships with nodes and edges. Nodes symbolize device addresses,
while undirected edges denote communication links between two nodes.
They are unweighted.

### Graph and Metrics Extraction

#### By whole dataset

`insert_metrics_to_dataframe(dataframe, time_interval, date_time, edge_source, edge_dest, label, name, date_timestamp, community_strategy='louvain')`

Take a dataframe, represent it on dynamic graph and compute community metrics.

- **Parameters:**
- `dataframe` (pd.DataFrame): Input data.
- `time_interval` (time.timedelta): Timedelta of graph window.
- `community_strategy` (str): Name of community strategy.
- `date_time` (str): Name of the date field in csv.
- `edge_source` (str): Name of field in csv for edge source id.
- `edge_dest` (list): Name of field in csv for edge destination id.
- `label` (str): Name of the label filed in csv.
- `name` (str): Naming suffix for metrics.
- `date_timestamp` (bool): True if dates are already in datetime format.

- **Returns:**
- `pd.DataFrame`: The dataframe enriched with community metrics
- **Usage Example:**

  ```python
  import pandas as pd
  from gpml.data_preparation.data_frame_handler import insert_metrics_to_dataframe
  df = pd.read_csv('data/ugr16/ugr_sample_100k.csv')
  insert_metrics_to_dataframe(df, timedelta(minutes=5), 'Date time', ['Source IP'],
                               ['Destination IP'], 'Label', 'ip5', False)

#### By chunk

`chunk_read_and_insert_gc_metrics(file_name, output_csv, time_interval, header=None, community_strategy='louvain',
chunksize=5000000, date_time="Date time", edge_source=None, edge_dest=None,label='Label', name='default')`

Use pandas to read a csv by chunk and write another csv with the same field + graph communities metrics.

- **Parameters:**
- `file_name` (str): The input csv file
- `output_csv` (str): The output csv file
- `time_interval` (time.timedelta): Timedelta of graph window
- `community_strategy` (str): Name of community strategy
- `chunksize` (int): Number of csv line to load in memory.
- `date_time` (str): Name of the date field in csv.
- `edge_source` (str): Name of field in csv for edge source id.
- `edge_dest` (list): Name of field in csv for edge destination id.
- `label` (str): Name of the label filed in csv.
- `name` (str): Naming suffix for metrics.

- **Returns:**
Results written in `output_csv`
- **Usage Example:**

  ```python
  import pandas as pd
  from gpml.data_preparation.data_frame_handler import chunk_read_and_insert_gc_metrics 
  chunk_read_and_insert_gc_metrics('data/ugr16/ugr_sample_100k.csv',
                                   'chunk_test1.csv', timedelta(minutes=5),edge_source = ['Source IP'],
                                   edge_dest = ['Destination IP'], chunksize=110000)
  ```
## Spectral Metrics

### Flow diagram - Spectral Metrics

![Alt text](images/spectral_metrics_flow.png 'spectral graph analysis process')

### Analysis process

This module provides you with spectral graph analysis over datasets of traffic
data logs monitored over the network.
This process contains multiple steps listed below:

- Reading raw data of traffic logs and importing it into dataframe.
- Transform the dataframe into timeseries by aggregating features.
- Apply time-windowing over timeseries where at each time-window:
- Extract connectivity graph G
- Extract Laplacian Matrix L for G
- Compute the spectrum of L
- Extract our introduced spectral metrics
- As a result, a spectral dataframe is obtained, where each data row represents
  one time-window with its aggregated and spectral metrics features.

### Graph Definition - Spectral Analysis

For spectral graph analysis, connectivity graphs represent network relationships
with nodes and edges. Nodes symbolize
devices, while undirected edges denote communication links between
two nodes weighted according to the traffic between the nodes.

### Timeseries Extraction

`timeseries_transformation(df, stime, time_unit,features_list,
sortby_list,groupby_list, aggregation_dict) -> df`

Converts raw data into a time series format by grouping and aggregating
specified features.

- **Parameters:**
- `df` (pd.DataFrame): The input dataframe containing raw data.
- `stime` (str): The column name for the start time.
- `time_unit` (str): The unit of time (e.g., 's' for seconds).
- `features_list` (list): The list of features to extract and transform.
- `sortby_list` (list): The list of columns to sort by.
- `groupby_list` (list): The list of columns to group by.
- `aggregation_dict` (dict): A dict specifying the aggregation operations.

- **Returns:**
- `pd.DataFrame`: The transformed dataframe in time series format.

- **Usage Example:**

  ```python
  import pandas as pd
  from gpml.data_preparation.time_series_extractor import time_series_extractor
  df = pd.read_csv('data/bot_iot/UNSW_2018_IoT_Botnet.csv')
  extracted_timeseries = timeseries_extractor(df, 'stime', 's', ['stime',
  'saddr', 'daddr', 'pkts', 'bytes'], ['stime'], ['stime', 'saddr', 'daddr'],
  ['pkts':'sum', 'bytes':'mean'])
  ```

### Spectral Metrics Extraction

`timewindow_transformation(timeseries, stime, saddr, daddr, pkts, bytes_size,
rate, lbl_category) -> df`

Applies time windowing techniques on the time series data, computing various
spectral metrics for each window.

- **Parameters:**
- `ts` (pd.DataFrame): The input time series dataframe.
- `stime` (str): The name of the start time column in the dataframe.
- `saddr` (str): The name of the source address column in the dataframe.
- `daddr` (str): The name of the destination address column in the dataframe.
- `pkts` (str): The name of the packets column in the dataframe.
- `bytes_size` (str): The name of the bytes column in the dataframe.
- `rate` (str): The name of the rate column in the dataframe.
- `lbl_category` (str): label category column.

- **Returns:**
- `pd.DataFrame`: The transformed dataframe spectral metrics for each tw.
- **Usage Example:**

  ```python
  spectral_df = spectral_metrics_extractor(extracted_timeseries, 'stime', 'saddr',
  'daddr', 'pkts', 'bytes', 'rate', 'attack')
  ```

@copyrights2024
