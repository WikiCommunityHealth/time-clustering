#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import inspect
import pathlib
import logging
import operator
import argparse
import configparser as cp

# time
import time
import datetime

# Essential Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

# algorithms
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans


def get_longest_series(list_of_timeseries):
    maxlen_series = -1
    longest_series = None
    for series in list_of_timeseries:
        if len(series) >= maxlen_series:
            maxlen_series = len(series)
            longest_series = series

    return longest_series


def check_minmax(series, minchk, maxchk):
    return (math.isclose(min(series), minchk) and 
            math.isclose(max(series), maxchk))


def has_nan_values(series):
    # check if any value is NaN
    # https://chartio.com/resources/tutorials/
    #    how-to-check-if-any-value-is-nan-in-a-pandas-dataframe/
    return series.isnull().values.any()


def barcol(n, list_of_tuples):
    # returns a list with the `n`-th elements of each tuple from a list 
    # of tuples `list_of_tuples`
    return [*map(operator.itemgetter(n), list_of_tuples)]


def ax_index(idx, dim):
    # nrow is the (integer) division between idx and dim
    nrow = idx // dim
    # ncol is the remain of the division between idx and dim
    ncol = idx % dim

    return (nrow, ncol)


def convert_time(atime):
    # Convert datetime objects to Matplotlib dates.
    # https://matplotlib.org/api/dates_api.html
    return mdates.date2num(atime)


def main(infile, outfile, langs, n_clusters=None, plot=True):
    df = pd.read_csv(infile).set_index('languagecode')
    df = df.rename({'year_month': 'date', 'count': 'value'}, axis=1)
    df = df.drop(columns=['milestone', 'peak'])

    df = df[df.date != '2021-12']
    df['date'] = pd.to_datetime(df['date'])

    timeseries = []
    lang_series = []
    for langcode in langs:
        tsdf = df.loc[langcode].reset_index()
        tsdf = tsdf.drop(columns=['languagecode'])

        tsdf = tsdf.loc[:, ["date", "value"]]
        df_value_max = tsdf.value.max()

        tsdf['value'] = tsdf['value']/df_value_max

        # while we are at it I just filtered the columns that we will be working on
        tsdf.set_index("date", inplace=True)

        # set the date columns as index
        tsdf.sort_index(inplace=True)

        # and lastly, ordered the data according to our date index
        tsdf = tsdf[:-1]

        timeseries.append(tsdf)
        lang_series.append(langcode)

    # --- PRE-PROCESSING
    logger.info('START PREPROCESSING')

    # find longest timeseries
    longest_series = get_longest_series(timeseries)
    time_months = [convert_time(tt.to_pydatetime())
                   for tt in longest_series.index.to_list()]
    len_series = len(longest_series)

    # add missing dates and fill gaps with interpolated values
    timeseries = [ts.reindex(longest_series.index) for ts in timeseries]
    # check there are no NAN values
    assert all([len(ts) == len(longest_series) for ts in timeseries]), \
           "Unexpected timeseries length"

    # fill gaps with interpolated values
    timeseries = [ts.interpolate(limit_direction="both") for ts in timeseries]
    # check there are no NAN values
    assert all([not has_nan_values(ts) for ts in timeseries]), "Unexpected NAN values"

    # rescale all timeseries to [0, 1]
    rescaled_timeseries = [MinMaxScaler().fit_transform(ts).reshape(len(ts))
                           for ts in timeseries]

    assert all([check_minmax(ts, minchk=0.0, maxchk=1.0)
                for ts in rescaled_timeseries]), "Timeseries limits outside [0, 1]"

    # --- CLUSTERING
    logger.info('START CLUSTERING')

    dim_x = dim_y = math.ceil(math.sqrt(math.sqrt(len(rescaled_timeseries))))
    som = MiniSom(dim_x, dim_y, len_series, sigma=0.3, learning_rate=0.1)
    logger.debug(f'dim_x: {dim_x}, dim_y: {dim_y}, len_series: {len_series}')

    som.random_weights_init(rescaled_timeseries)
    som.train(rescaled_timeseries, 50000)

    # --- K-MEANS
    logger.info('START K-MEANS')

    # A good rule of thumb is choosing k as the square root of the number of points
    # in the training data set in kNN
    if n_clusters is None:
        n_clusters = math.ceil(math.sqrt(len(timeseries))) 

    km = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw")
    labels = km.fit_predict(timeseries)
    unique_labels = sorted(set(labels))

    # --- PLOTS
    logger.info('PLOTTING')
    plots_per_row = math.ceil(math.sqrt(n_clusters))
    fig, axs = plt.subplots(plots_per_row, plots_per_row, figsize=(30, 30))
    fig.suptitle('Clusters')
    # xtick format string
    years_fmt = mdates.DateFormatter('%Y-%m')

    clusters = {label: list() for label in unique_labels}

    # for each label there is, plot every series with that label
    for label, ts in zip(labels, rescaled_timeseries):
        nrow, ncol = ax_index(label, plots_per_row)
        axs[nrow, ncol].plot(time_months, ts, c="gray", alpha=0.4)
        axs[nrow, ncol].xaxis.set_major_formatter(years_fmt)
        axs[nrow, ncol].xaxis.set_major_locator(plt.MaxNLocator(20))
        clusters[label].append(ts)

    for label in unique_labels:
        nrow, ncol = ax_index(label, plots_per_row)
        axs[nrow, ncol].plot(time_months, np.average(np.vstack(clusters[label]), axis=0),
                             c="red")
        axs[nrow, ncol].set_title(f"C{label}")

    fig.autofmt_xdate()
    plt.draw()
    logger.info('K-Means 1: DONE.')

    # bar_data: [('C0', 15), ('C1', 1), ('C2', 19), ('C3', 17)]
    bar_data = [(f"C{label}", len(c)) for label, c in clusters.items()]
    plt.figure(figsize=(15, 5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(barcol(0, bar_data), barcol(1, bar_data))
    plt.draw()

    fancy_names_for_labels = [f"Cluster {label}" for label in labels]
    out_df = (pd.DataFrame(zip(lang_series, fancy_names_for_labels),
                           columns=["Series", "Cluster"])
              .sort_values(by="Cluster").set_index("Series")
              )
    out_df.to_csv(outfile, sep="\t")

    fig, axs = plt.subplots(plots_per_row, plots_per_row, figsize=(25, 25))
    fig.suptitle('Clusters')

    # for each label there is, plot every series with that label
    for label, ts in zip(labels, rescaled_timeseries):
        nrow, ncol = ax_index(label, plots_per_row)
        axs[nrow, ncol].plot(time_months, ts, c="gray", alpha=0.4)
        axs[nrow, ncol].xaxis.set_major_formatter(years_fmt)
        axs[nrow, ncol].xaxis.set_major_locator(plt.MaxNLocator(20))
        clusters[label].append(ts)

    for label in unique_labels:
        nrow, ncol = ax_index(label, plots_per_row)
        axs[nrow, ncol].plot(time_months, dtw_barycenter_averaging(np.vstack(clusters[label])),
                             c="green")
        axs[nrow, ncol].set_title(f"C{label}")

    fig.autofmt_xdate()
    plt.draw()
    logger.info('K-Means 2: DONE.')

    if plot:
        logger.info('SHOW PLOTS.')
        plt.show()

    return


#######################################################################################
def logging_setup(loglevel):
    formatter = logging.Formatter(
        '[%(asctime)s][%(levelname)s]:\t%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_out = logging.StreamHandler(sys.stdout)
    console_out.setFormatter(formatter)
    console_out.setLevel(loglevel)
    logfile_out = logging.FileHandler('time_clustering.log')

    root_logger.addHandler(console_out)
    root_logger.addHandler(logfile_out)

    return root_logger


def get_params_from_config(config_file):
    config = cp.ConfigParser()
    config.read(config_file)

    params = {}
    params['infile'] = pathlib.Path(config['general']['infile'])
    params['outfile'] = pathlib.Path(config['general']['outfile'])

    params['langs'] = [lang.strip()
                       for lang in config['general']['langs'].strip().split(',')]
    params['n_clusters'] = int(config['clustering']['n_clusters'])

    return params


def get_params_from_cli(cli_args):
    params = {}

    params['loglevel'] = logging.WARNING
    if cli_args.verbose:
        params['loglevel'] = logging.INFO
    if cli_args.debug:
        params['loglevel'] = logging.DEBUG

    if cli_args.n_clusters:
        params['n_clusters'] = int(cli_args.n_clusters)

    params['plot'] = args.plot

    return params


def get_main_params(config, cli):
    # get main parameters names
    mnames = set((inspect.signature(main).parameters).keys())

    # start with config params
    params = {k: config[k] for k in mnames.intersection(config.keys())}

    # update values with cli settings
    params.update((k, cli[k]) for k in mnames.intersection(cli.keys()))

    assert (set(params.keys()) == mnames)

    return params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cluster Wikipedia active editors timelines.")
    parser.add_argument('-c', '--config',
                        type=pathlib.Path,
                        default=pathlib.Path('time_clustering.conf'),
                        help='Config file [default: time_clustering.conf].'
                        )
    parser.add_argument('-d', '--debug',
                        action='store_true',
                        help='Enable debug (DEBUG) logging. Overrides --verbose.'
                        )
    parser.add_argument('-n', '--n-clusters',
                        type=int,
                        help='Number of clustres'
                        )
    parser.add_argument('--no-plot',
                        dest='plot',
                        action='store_false',
                        help='Do not show plots.'
                        )
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable verbose (INFO) logging.'
                        )
    args = parser.parse_args()
    params_cli = get_params_from_cli(args)

    global logger
    logger = logging_setup(params_cli['loglevel'])

    params_config = get_params_from_config(args.config)

    main_params = get_main_params(params_config, params_cli)

    start_time = time.time()
    start_time_iso = datetime.datetime.fromtimestamp(start_time).isoformat()
    logger.info(f'Starting TIME CLUSTERING at: {start_time_iso}')

    # pass dictionary of parameters to function
    main(**main_params)

    finish_time = time.time()
    finish_time_iso = datetime.datetime.fromtimestamp(finish_time).isoformat()

    elapsed_time = round(finish_time-start_time, 3)
    logger.info(f'Finishing TIME CLUSTERING at: {finish_time_iso}')
    logger.info(f'TIME CLUSTERING completed successfuly in {elapsed_time} seconds.')

    exit(0)
