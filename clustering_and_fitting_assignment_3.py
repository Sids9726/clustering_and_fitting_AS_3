# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
import cluster_tools as ct
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import scipy.optimize as opt
import errors as err


def read_data_all(filename, countries):
    """


    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    countries : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    df_t : TYPE
        DESCRIPTION.

    """
    # read the data
    df = pd.read_csv(filename, skiprows=4)

    # set index
    df.index = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    countries = countries

    column_name = np.arange(1990, 2021)
    column_name = [str(n) for n in column_name]
    df = df.loc[countries, column_name]

    # transpose the data
    df_t = df.T
    df_t.index = df_t.index.astype(int)
    return df, df_t


def poly(x, a, b, c, d):
    """


    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    d : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    """ Calulates polynominal"""

    x = x - 1990
    f = a + b*x + c*x**2 + d*x**3

    return f


def build_cluster_graph(country):
    """


    Parameters
    ----------
    country : TYPE
        DESCRIPTION.

    Returns
    -------
    df_cluster : TYPE
        DESCRIPTION.

    """

    # creating new dataframe for cluster
    df_cluster = pd.DataFrame()
    df_cluster["co2"] = carbon_emission_df_t[country]
    df_cluster["forest_area"] = forest_area_df_t[country]

    df_cluster = df_cluster.dropna()
    df_norm, df_min, df_max = ct.scaler(df_cluster)

    # calculate silhouette score for 2 to 10 clusters
    '''
    for ic in range(2, 11):
        score = one_silhoutte(df_cluster, ic)
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   
        
        # allow for minus signs
    '''
    ncluster = 2

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)  # fit done on x,y pairs

    labels = kmeans.labels_

    # extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = ct.backscale(cen, df_min, df_max)

    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]

    # extract x and y values of data points
    x = df_cluster["co2"]
    y = df_cluster["forest_area"]

    # Set a more visually appealing colormap
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(15, 8), dpi=300)
    # Plot data with kmeans cluster number
    scatter = plt.scatter(x, y, 25, labels, cmap=cmap, marker="o",
                          edgecolors='k', linewidth=0.8)

    # Show cluster centres
    plt.scatter(xkmeans, ykmeans, 150, "k", marker="D",
                label="Cluster Centers", edgecolors='w', linewidth=1.5)
    plt.scatter(xkmeans, ykmeans, 150, "y", marker="+", label="Centroid",
                edgecolors='k', linewidth=1.5)

    # Add legend
    plt.legend()

    # Add grid lines
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set titles and labels
    plt.title("demonstration of graph with clusters",
              fontsize=20, color="black")
    plt.xlabel("Co2 emission (kt)", fontsize=15, color="black")
    plt.ylabel("forst area (% of land area)", fontsize=15, color="black")

    # Show the colorbar for better interpretation
    plt.colorbar(scatter, label='two cluster', ticks=range(ncluster))

    plt.savefig("cluster of "+country+".png", dpi=300, va="center")
    # Show the plot
    plt.show()
    return df_cluster


def build_fitting_graph(df_cluster, indicator, title):
    """


    Parameters
    ----------
    df_cluster : TYPE
        DESCRIPTION.
    indicator : TYPE
        DESCRIPTION.
    title : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    df_cluster["Year"] = df_cluster.index
    param, covar = opt.curve_fit(poly, df_cluster["Year"],
                                 df_cluster[indicator])
    df_cluster["fit"] = poly(df_cluster["Year"], *param)

    # forecasting of data
    year = np.arange(1990, 2030)
    forecast = poly(year, *param)
    sigma = err.error_prop(year, poly, param, covar)
    low = forecast - sigma
    up = forecast + sigma

    df_cluster["fit"] = poly(df_cluster["Year"], *param)

    plt.figure(figsize=(15, 8), dpi=250)
    plt.plot(df_cluster["Year"], df_cluster[indicator], label=indicator)
    plt.plot(year, forecast, label="forecast")

    # labelling of graph
    plt.xlabel("Year", fontsize=15)
    lis = title.split(" ")
    plt.ylabel(lis[0]+" "+lis[1], fontsize=15)

    # set title
    plt.title(title, fontsize=20)

    # plot uncertainty range
    plt.fill_between(year, low, up, color="yellow",
                     alpha=0.6, label="Confidence margin")
    plt.savefig(title+".png", dpi=300, va="center")

    plt.legend()
    plt.show()


###################### main function ########################

# function calling
title1 = ["Forest area of Indonesia", "Forest area of Spain"]
title2 = ["Co2 emission of Indonesia", "Co2 emission of Spain"]
countries = ["Indonesia", "Spain"]


var_x = "carbon_emission.csv"
carbon_emission_df, carbon_emission_df_t = read_data_all(var_x, countries)
var_x = "forest_area.csv"
forest_area_df, forest_area_df_t = read_data_all(var_x, countries)
var_x = "agricultural_land.csv"
agricultural_land_df, agricultural_land_df_t = read_data_all(var_x, countries)


# clustering and fitting
for x, y, z in zip(countries, title1, title2):
    df_cluster = build_cluster_graph(x)
    build_fitting_graph(df_cluster, "forest_area", y)
    build_fitting_graph(df_cluster, "co2", z)
