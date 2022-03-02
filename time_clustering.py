# -*- coding: utf-8 -*-


# time
import time
import datetime
from dateutil import relativedelta
import calendar


import sys
import os
import math

# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Preprocessing
from sklearn.preprocessing import MinMaxScaler

# Algorithms
from minisom import MiniSom
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

# https://stats.wikimedia.org/#/all-projects
# https://meta.wikimedia.org/wiki/List_of_Wikipedias/ca
# https://meta.wikimedia.org/wiki/Research:Metrics#Volume_of_contribution


# MAIN
def main():

    """

    directory = 'kaggle/'
    mySeries = []
    namesofMySeries = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(directory+filename)
            df = df.loc[:,["date","value"]]

#            print (df.head(10))
#            input('')
            # While we are at it I just filtered the columns that we will be working on
            df.set_index("date",inplace=True)
            # ,set the date columns as index
            df.sort_index(inplace=True)
            # and lastly, ordered the data according to our date index
            mySeries.append(df)
            namesofMySeries.append(filename[:-4])


    """

#    print (mySeries)
#    input('')
    




    df = pd.read_csv('active_editors_all_languages.csv').set_index('languagecode')
    df = df.rename({'year_month': 'date', 'count':'value'}, axis=1)
    df = df.drop(columns=['milestone','peak'])

    df = df[df.date != '2021-12']
    df['date'] = pd.to_datetime(df['date'])
#    df['date'] = pd.to_datetime(df["date"].dt.strftime('%Y-%m'))

    langs = df.index.unique().tolist()
#    langs = ['ca','es','eu','en','it','ro','de','fr']
    langs = ['de','en','es','fr','ja','ar','fa','he','id','it','ko','nl','pl','pt','ru','tr','uk','vi','zh','bn','cs','fi','hu','sv','th','az','be','bg','ca','da','el','eo','et','eu','gl','hi','hr','hy','ka','lt','lv','ml','ms','no','ro','simple','sk','sl','sr','ta','ur','zh_yue']

    mySeries = []
    namesofMySeries = []
    for langcode in langs:
        df1 = df.loc[langcode].reset_index()
        df1 = df1.drop(columns=['languagecode'])

#        print (df1.head(10))
#        input('')

        df1 = df1.loc[:,["date","value"]]

        df_value_max = df1.value.max()

        df1['value'] = df1['value']/df_value_max

        # While we are at it I just filtered the columns that we will be working on
        df1.set_index("date",inplace=True)
        # ,set the date columns as index
        df1.sort_index(inplace=True)
        # and lastly, ordered the data according to our date index
        df1 = df1[:-1]

        mySeries.append(df1)
        namesofMySeries.append(langcode)



    print (namesofMySeries)
    print (mySeries)
    print ('inici')





    # PRE-PROCESSING


    series_lengths = {len(series) for series in mySeries}
    print(series_lengths)


    ind = 0
    for series in mySeries:
#        print("["+str(ind)+"] "+series.index[0]+" "+series.index[len(series)-1])
        ind+=1




    max_len = max(series_lengths)
    longest_series = None
    for series in mySeries:
        if len(series) == max_len:
            longest_series = series


    problems_index = []

    for i in range(len(mySeries)):
        if len(mySeries[i])!= max_len:
            problems_index.append(i)
            mySeries[i] = mySeries[i].reindex(longest_series.index)


    def nan_counter(list_of_series):
        nan_polluted_series_counter = 0
        for series in list_of_series:
            if series.isnull().sum().sum() > 0:
                nan_polluted_series_counter+=1
        print(nan_polluted_series_counter)

    nan_counter(mySeries)

    for i in problems_index:
        mySeries[i].interpolate(limit_direction="both",inplace=True)

    
    nan_counter(mySeries)



    for i in range(len(mySeries)):
        scaler = MinMaxScaler()
        mySeries[i] = MinMaxScaler().fit_transform(mySeries[i])
        mySeries[i]= mySeries[i].reshape(len(mySeries[i]))

    print("max: "+str(max(mySeries[0]))+"\tmin: "+str(min(mySeries[0])))
    print(mySeries[0][:5])





    # CLUSTERING

    som_x = som_y = math.ceil(math.sqrt(math.sqrt(len(mySeries))))
    som = MiniSom(som_x, som_y,len(mySeries[0]), sigma=0.3, learning_rate = 0.1)

    print (mySeries[1])
    print ('ara')

    som.random_weights_init(mySeries)
    som.train(mySeries, 50000)


    """
    def plot_som_series_averaged_center(som_x, som_y, win_map):
        fig, axs = plt.subplots(som_x,som_y,figsize=(25,25))
        fig.suptitle('Clusters')
        for x in range(som_x):
            for y in range(som_y):
                cluster = (x,y)
                if cluster in win_map.keys():
                    for series in win_map[cluster]:
                        axs[cluster].plot(series,c="gray",alpha=0.5) 
                    axs[cluster].plot(np.average(np.vstack(win_map[cluster]),axis=0),c="red")
                cluster_number = x*som_y+y+1
                axs[cluster].set_title(f"Cluster {cluster_number}")

        plt.show()


    win_map = som.win_map(mySeries)
    # Returns the mapping of the winner nodes and inputs

#    plot_som_series_averaged_center(som_x, som_y, win_map)





    cluster_c = []
    cluster_n = []
    for x in range(som_x):
        for y in range(som_y):
            cluster = (x,y)
            if cluster in win_map.keys():
                cluster_c.append(len(win_map[cluster]))
            else:
                cluster_c.append(0)
            cluster_number = x*som_y+y+1
            cluster_n.append(f"C{cluster_number}")

    plt.figure(figsize=(25,5))
    plt.title("Cluster Distribution for SOM")
    plt.bar(cluster_n,cluster_c)
    plt.show()
    """



    # K-MEANS

    print ('k-means')
    cluster_count = math.ceil(math.sqrt(len(mySeries))) 
    # A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

    cluster_count = 4

    km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

    labels = km.fit_predict(mySeries)





    plot_count = math.ceil(math.sqrt(cluster_count))

    fig, axs = plt.subplots(plot_count,plot_count,figsize=(30,30))
    fig.suptitle('Clusters')
    row_i=0
    column_j=0
    # For each label there is,
    # plots every series with that label
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
                if(labels[i]==label):
                    axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                    cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
        axs[row_i, column_j].set_title("C"+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0

     
    plt.show()
    print ('K-means 1 fet.')

    cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
    cluster_n = ["C"+str(i) for i in range(cluster_count)]
    plt.figure(figsize=(15,5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(cluster_n,cluster_c)
    plt.show()
    print ('Distribution 1')

    fancy_names_for_labels = [f"Cluster {label}" for label in labels]
    a = pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")
    a.to_csv('file_langs_clusters.tsv', sep = "\t")
    print (a.head(10))
    print (fancy_names_for_labels)
    print ('Output done.')


    input('')



    
    plot_count = math.ceil(math.sqrt(cluster_count))

    fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
    fig.suptitle('Clusters')
    row_i=0
    column_j=0
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
                if(labels[i]==label):
                    axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                    cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)),c="red")
        axs[row_i, column_j].set_title("C"+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0
            
    plt.show()
    print ('K-means 2 fet.')

    cluster_c = [len(labels[labels==i]) for i in range(cluster_count)]
    cluster_n = ["C"+str(i) for i in range(cluster_count)]
    plt.figure(figsize=(15,5))
    plt.title("Cluster Distribution for KMeans")
    plt.bar(cluster_n,cluster_c)
    plt.show()
    print ('Distribution 2')


    fancy_names_for_labels = [f"Cluster {label}" for label in labels]
    a = pd.DataFrame(zip(namesofMySeries,fancy_names_for_labels),columns=["Series","Cluster"]).sort_values(by="Cluster").set_index("Series")
    a.to_csv('file.tsv', sep = "\t")
    print (a.head(10))
    print (fancy_names_for_labels)
    print ('Output done.')

    input('')

    








    input('')

    return


    # SHOW THE ORIGINAL TIMESERIES
    fig, axs = plt.subplots(5,5,figsize=(25,25))
    fig.suptitle('Series')
    for i in range(5):
        for j in range(5):
            if i*4+j+1>len(mySeries): # pass the others that we can't fill
                continue
            axs[i, j].plot(mySeries[i*4+j].values)
            axs[i, j].set_title(namesofMySeries[i*4+j])
    plt.show()



    # series_lengths = {len(series) for series in mySeries}
    # print(series_lengths)


    print ('eh')



################################################################

# FUNCTIONS

def get_time_clustering():

    cluster_count = math.ceil(math.sqrt(len(mySeries))) 
    # A good rule of thumb is choosing k as the square root of the number of points in the training data set in kNN

    km = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw")

    labels = km.fit_predict(mySeries)


    plot_count = math.ceil(math.sqrt(cluster_count))

    fig, axs = plt.subplots(plot_count,plot_count,figsize=(25,25))
    fig.suptitle('Clusters')
    row_i=0
    column_j=0
    # For each label there is,
    # plots every series with that label
    for label in set(labels):
        cluster = []
        for i in range(len(labels)):
                if(labels[i]==label):
                    axs[row_i, column_j].plot(mySeries[i],c="gray",alpha=0.4)
                    cluster.append(mySeries[i])
        if len(cluster) > 0:
            axs[row_i, column_j].plot(np.average(np.vstack(cluster),axis=0),c="red")
        axs[row_i, column_j].set_title("Cluster "+str(row_i*som_y+column_j))
        column_j+=1
        if column_j%plot_count == 0:
            row_i+=1
            column_j=0
            
    plt.show()


    print ('eh')




#######################################################################################

class Logger_out(object): # this prints both the output to a file and to the terminal screen.
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("time_clustering.out", "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
class Logger_err(object): # this prints both the output to a file and to the terminal screen.
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("time_clustering.err", "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


if __name__ == '__main__':
    sys.stdout = Logger_out()
    sys.stderr = Logger_err()

    startTime = time.time()

    print ('* Starting the TIME CLUSTERING at this exact time: ' + str(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
    main()

    finishTime = time.time()
    print ('* Done with the TIME CLUSTERING completed successfuly after: ' + str(datetime.timedelta(seconds=finishTime - startTime)))