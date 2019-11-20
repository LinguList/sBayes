#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import matplotlib as mpl
import matplotlib.pyplot as plt
# import seaborn as sns
plt.style.use('seaborn-paper')
plt.tight_layout()

import numpy as np
import math

from src.util import zones_autosimilarity, add_edge, compute_delaunay, colorline
from src.util import bounding_box, round_int
from scipy.stats import gamma, linregress
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import minimum_spanning_tree
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.ops import cascaded_union, polygonize
from shapely import geometry
from descartes import PolygonPatch
import geopandas as gpd
import os
os.environ["PROJ_LIB"] = "C:/Users/ranacher/Anaconda3/Library/share"


def get_colors():
    """This function creates a dict with colors
    Returns:
        (dict): a dictionary comprising the plot colors
    """

    plot_colors = {'histogram': {'fitted_line': (1.000, 0.549, 0.000),      # darkorange
                                 'background': (1, 1, 1)},                  # white
                   'zone': {'background_nodes': (0.502, 0.502, 0.502),      # grey
                            'in_zone': (1.000, 0.549, 0.000),               # darkorange
                            'triangulation': (1.000, 0.549, 0.000)},
                   'trace': {'lh': (0.502, 0.502, 0.502),                   # grey
                             'maximum': (1.000, 0.549, 0.000),              # darkorange
                             'precision': (0.216, 0.494, 0.722),            # blue
                             'recall': (1.000, 0.549, 0.000)},              # darkorange
                   'boxplot': {'median': (1.000, 0.549, 0.000),             # darkorange
                               'whiskers': (0.502, 0.502, 0.502),           # grey
                               'box': (1.000, 0.549, 0.000, 0.2)},          # darkorange (transparent)
                   'zones': {'background_nodes': (0.502, 0.502, 0.502),
                             'in_zones': [(0.894, 0.102, 0.11),             # red
                                          (0.216, 0.494, 0.722),            # blue
                                          (0.302, 0.686, 0.29),             # green
                                          (0.596, 0.306, 0.639),            # violett
                                          (1.000, 0.549, 0.000),            # darkorange)
                                          (1, 1, 0.2),                      # yellow
                                          (0.651, 0.337, 0.157),            # brown
                                          (0.969, 0.506, 0.749),            # pinkish
                                          (0, 0, 0)],                       # black ]
                             'triangulation': (0.4, 0., 0.)}}
    return plot_colors


def plot_triangulation_edges(samples, net, triangulation, ax=None):
    """ This function adds a triangulation to the points in the posterior

    Args:
        samples (np.array): the samples from the MCMC
        net (dict): The full network containing all sites.
        triangulation (str): type of triangulation, either "mst" or "delaunay"
        ax (axis): matlibplot axis
    """

    if ax is None:
        ax = plt.gca()

    all_sites = net['locations']
    # todo: change to np.any

    zone_locations = all_sites[samples[0]]
    dist_mat = net['dist_mat']

    delaunay = compute_delaunay(zone_locations)
    mst = delaunay.multiply(dist_mat)

    if triangulation == "delaunay":
        tri = delaunay

    elif triangulation == "mst":
        tri = mst

    print(tri.shape, "tri")
    # Get data
    col = get_colors()
    adj_mat = net['adj_mat']
    locations = net['locations']

    # Add edge evidence for the sampled zone
    n_samples, n_v = samples.shape
    weights = np.ones(n_samples)
    edge_counts = (weights[:, None] * samples).T.dot(samples)
    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples).clip(0, 1)
    edge_freq[edge_freq < 0.1] = 0.

    # Plot background
    size = 4
    bg=plt.scatter(*locations.T, s=size, color=col['zones']['background_nodes'])
    print('Nonzero:', np.count_nonzero(edge_freq))

    # Plot posterior
    edges = np.argwhere(edge_freq)
    line_col = col['zones']['triangulation']
    line_col = [(line_col + (c,)) for c in edge_freq[edges[:, 0], edges[:, 1]]]

    lines = LineCollection(locations[edges], colors=line_col)
    ax.add_collection(lines)

    # # Remove axes
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # Add legend
    # ax.legend([bg, lines], ['All sites', 'Edges in posterior distribution'], frameon=False, fontsize=10)

    return ax


def plot_posterior(samples, net, ax=None):
    """ This function plots the posterior distribution of contact zones

    Args:
        samples (np.array): the samples from the MCMC
        net (dict): The full network containing all sites.
    """
    if ax is None:
        ax = plt.gca()

    # Get data
    col = get_colors()
    adj_mat = net['adj_mat']
    locations = net['locations']

    # Add edge evidence for the sampled zone
    n_samples, n_v = samples.shape
    weights = np.ones(n_samples)
    edge_counts = (weights[:, None] * samples).T.dot(samples)
    edge_counts *= adj_mat.toarray()
    edge_freq = (edge_counts / n_samples).clip(0, 1)
    edge_freq[edge_freq < 0.1] = 0.

    # Plot background
    size = 4
    bg=plt.scatter(*locations.T, s=size, color=col['zones']['background_nodes'])
    print('Nonzero:', np.count_nonzero(edge_freq))

    # Plot posterior
    edges = np.argwhere(edge_freq)
    line_col = col['zones']['triangulation']
    line_col = [(line_col + (c,)) for c in edge_freq[edges[:, 0], edges[:, 1]]]

    lines = LineCollection(locations[edges], colors=line_col)
    ax.add_collection(lines)

    # # Remove axes
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    # # Add legend
    # ax.legend([bg, lines], ['All sites', 'Edges in posterior distribution'], frameon=False, fontsize=10)

    return ax


def plot_zone(zone, net):
    """ This function plots a contact zone proposed by the the MCMC

    Args:
        zone (np.array): The current zone (boolean array).
        net (dict): The full network containing all sites.
    """

    # Initialize plot
    fig, ax = plt.subplots()
    col = get_colors()
    all_sites = net['locations']
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zone']['background_nodes'])
    zo = ax.scatter(*all_sites[zone].T, s=size*3, color=col['zone']['triangulation'])

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, zo], ['All sites', 'Sites in proposed contact zone'], frameon=False, fontsize=10)
    plt.show()


def plot_zones(zones, net, ax=None):
    """ This function plots the parallel contact zones proposed by the MCMC

    Args:
        zones (np.array): The current zone (boolean array).
        net (dict): The full network containing all sites.
    """

    # Initialize plot
    if ax is None:
        return_ax = False
        _, ax = plt.subplots()
    else:
        return_ax = True

    col = get_colors()
    all_sites = net['locations']
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zones']['background_nodes'])
    zo = []

    if isinstance(zones, dict):
        zones = list(zones.values())

    for z, zone in enumerate(zones):

        zo.append(ax.scatter(*all_sites[zones[z]].T, s=size * 6, color=col['zones']['in_zones'][int(z)]))

    # Remove axes
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Add legend
    ax.legend([bg, zo[0]], ['All sites', 'Sites in proposed contact zones'], frameon=False, fontsize=10)

    if return_ax:
        return ax
    else:
        plt.show()


# def plot_posterior_density(zones, net):
#     # TODO: Labels, Color palette
#
#     """ This function creates a kernel density plot of all sites in the posterior distribution
#
#     Args:
#         zones (np.array): the posterior of all zones
#         net (dict): The full network containing all sites.
#     """
#
#     all_sites = net['locations']
#     points_in_post = []
#
#     for z in zones:
#         try:
#             points_in_post += all_sites[z[0]]
#
#         except ValueError:
#             points_in_post = all_sites[z[0]]
#
#     points_in_post = np.array(points_in_post)
#     ax = sns.kdeplot(points_in_post, cmap="Reds", shade=True, shade_lowest=False)
#
#     # Remove axes
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
#     plt.show()


def plot_posterior_frequency(mcmc_res, net, nz=-1, burn_in=0.2, show_zone_bbox=False, ts_posterior_freq=0.8,
                             size=20, cmap=plt.cm.get_cmap('jet'), fname='posterior_frequency'):
    """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
    its frequency in the posterior

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        net (dict): The full network containing all sites.
        nz (int): For multiple zones: which zone should be plotted? If -1, plot all.
        burn_in (float): Percentage of first samples which is discarded as burn-in
        show_zone_bbox (boolean): Adds box(es) with annotation to zone(s)
        ts_posterior_freq (float): If zones are annotated this threshold
        size (int): size of points
        cmap (matplotlib.cm): colormap for posterior frequency of points
        fname (str): a path followed by a the name of the file
    """


    fontsize = 24
    frame_width = 1.5

    fig, ax = plt.subplots(figsize=(15, 10))

    zones = mcmc_res['zones']
    n_zones = len(zones)

    # getting positions of points
    locations = net['locations']
    positions = net['locations'].T


    # plot all zones
    if nz == -1:
        # get samples from all zones
        n_samples = len(zones[0])
        zones_reformatted = [sum(k) for k in zip(*zones)]

        # exclude burn-in
        end_bi = math.ceil(len(zones_reformatted) * burn_in)
        density = (np.sum(zones_reformatted[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # plot only one zone (passed as argument)
    else:
        # get samples of the zone
        zone = zones[nz-1]
        n_samples = len(zone)

        # exclude burn-in
        end_bi = math.ceil(n_samples * burn_in)

        # compute frequency of each point in that zone
        density = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))


    # sorting points according to their frequency in the posterior
    # -> high frequency points are plotted over low frequency points
    positions_sorted = [[c for _, c in sorted(zip(density, coords), key=lambda pair: pair[0])] for coords in positions]
    density_sorted = sorted(density)
    density_sorted = list(map(lambda x: x * 100, density_sorted))

    # adding scatter plot to axes
    ax.scatter(*positions_sorted, c=density_sorted, s=size, cmap=cmap, alpha=0.6, linewidths=0)

    if show_zone_bbox:
        # create list with all zone indices
        indices_zones = [nz-1] if nz != -1 else range(n_zones)
        print(f'Zone indices: {indices_zones}')

        for zone_index in indices_zones:
            # get samples of the zone
            zone = zones[zone_index]
            n_samples = len(zone)

            # exclude burn-in
            end_bi = math.ceil(n_samples * burn_in)

            # compute frequency of each point in that zone
            posterior_freq = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))
            is_contact_point = posterior_freq > ts_posterior_freq
            cp_locations = locations[is_contact_point]
            print(cp_locations.shape)
            zone_bbox = bounding_box(cp_locations)
            print(zone_bbox)
            bbox_ll = (round_int(zone_bbox['x_min'], 'down'), round_int(zone_bbox['y_min'], 'down'))
            bbox_height = round_int(zone_bbox['y_max'], 'up') - round_int(zone_bbox['y_min'], 'down')
            bbox_width = round_int(zone_bbox['x_max'], 'up') - round_int(zone_bbox['x_min'], 'down')
            bbox = mpl.patches.Rectangle(bbox_ll, bbox_width, bbox_height, fill=False, edgecolor='k',
                                         lw=1)

            ax.add_patch(bbox)



    # getting axes ranges and rounding them
    x_min, x_max = np.min(positions[0,:]), np.max(positions[0,:])
    y_min, y_max = np.min(positions[1,:]), np.max(positions[1,:])
    x_min, x_max = round_int(x_min, 'down'), round_int(x_max, 'up')
    y_min, y_max = round_int(y_min, 'down'), round_int(y_max, 'up')

    # print(f'x range {x_min}--{x_max}')
    # print(f'y range {y_min}--{y_max}')

    # x axis
    ax.set_xlim([x_min, x_max])
    x_ticks = list(np.linspace(x_min, x_max, 6))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=fontsize)

    # y axis
    ax.set_ylim([y_min, y_max])
    y_ticks = list(np.linspace(y_min, y_max, 6))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)


    [ax.spines[side].set_linewidth(frame_width) for side in ['left', 'bottom', 'right', 'top']]
    plt.rcParams["axes.linewidth"] = frame_width

    # colorbar
    cbar_ticks = list(np.linspace(0, 1, 6))
    cbar_ticklabels = [f'{round(t * 100, 0):.0f}' for t in cbar_ticks]

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="3%", pad=1, pack_start=True)
    fig.add_axes(cax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1),
                                     orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xlabel('Frequency of point in posterior (%)', fontsize=fontsize, labelpad=-75)
    cbar.ax.set_xticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=fontsize)

    # ax.set_title(title, fontsize=fontsize)


    fig.savefig(fname, bbox_inches='tight', dpi=400)
    # plt.show()


def plot_posterior_frequency_map(mcmc_res, net, nz=-1, burn_in=0.2, plot_family=None, family_alpha_shape=None,
                             family_color=None, bg_map=False, proj4=None, geojson_map=None,
                             geo_json_river=None, offset_factor=0.03, plot_edges=False,
                             labels=False, labels_offset=None, size=20, cmap=plt.cm.get_cmap('jet'),
                             annotate_zones=False, fname='posterior_frequency'):
    """ This function creates a scatter plot of all sites in the posterior distribution. The color of a site reflects
    its frequency in the posterior

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        net (dict): The full network containing all sites.
        nz (int): For multiple zones: which zone should be plotted? If -1, plot all.
        burn_in (float): Percentage of first samples which is discarded as burn-in
        plot_family (str): Visualize all sites belonging to a family (either "alha_shapes", "color" or None)
        family_alpha_shape (float): Alpha value passed to the function compute_alpha_shapes
        family_color (str): Color of family in plot
        bg_map (bool): Use a background map for for the visualization?
        proj4 (str): projection information when using a background map
        geojson_map (str): file location of the background map
        geo_json_river (str): file location of river data (for making the background map a bit nicer)
        offset_factor (float): map extent is tailored to the location of the sites. This defines the offset.
        plot_edges (bool): Plot the edges of the mst triangulation for the zone?
        labels (bool): Plot the labels of the families?
        labels_offset (float, float): Offset of the labels in both x and y
        size (int): size of the points
        fname (str): a path followed by a the name of the file
    """

    fontsize = 24
    frame_width = 1.5

    fig, ax = plt.subplots(figsize=(15, 10))

    zones = mcmc_res['zones']
    n_zones = len(zones)

    # getting positions and name of points
    positions = net['locations'].T
    names = net['names']

    # Find index of first sample after burn-in
    if bg_map:
        if proj4 is None and geojson_map is None:
            raise Exception('If you want to use a map provide a geojson and a crs')

        world = gpd.read_file(geojson_map)
        world = world.to_crs(proj4)
        # world.plot(ax=ax, color=(.95,.95,.95), edgecolor='grey')
        world.plot(ax=ax, color='w', edgecolor='black')

        if geo_json_river is not None:
            rivers = gpd.read_file(geo_json_river)
            rivers = rivers.to_crs(proj4)
            rivers.plot(ax=ax, color=None, edgecolor="skyblue")

    # plot all zones
    if nz == -1:
        # get samples from all zones
        n_samples = len(zones[0])
        zones = [sum(k) for k in zip(*zones)]

        # exclude burn-in
        end_bi = math.ceil(len(zones) * burn_in)
        density = (np.sum(zones[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # plot only one zone (passed as argument)
    else:
        # get samples of the zone
        zone = zones[nz - 1]
        n_samples = len(zone)

        # exclude burn-in
        end_bi = math.ceil(n_samples * burn_in)

        # compute frequency of each point in that zone
        density = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    # sorting points according to their frequency in the posterior
    # -> high frequency points are plotted over low frequency points
    X, Y = positions
    positions_sorted = [[c for _, c in sorted(zip(density, coords), key=lambda pair: pair[0])] for coords in positions]

    density_sorted = sorted(density)
    density_sorted = list(map(lambda x: x * 100, density_sorted))

    # adding scatter plot to axes
    ax.scatter(*positions_sorted, c=density_sorted, s=size, cmap=cmap, vmin=0, alpha=0.6, linewidths=0)

    # Add labels for those sites which occur most often in the posterior
    if labels:
        if labels_offset is None:
            labels_offset = (10., 10.)
        for i, name in enumerate(names):
            if density[i] > 0.1:
                plt.annotate(name, all_sites[i] + [labels_offset[0], labels_offset[1]], zorder=11, fontsize=9)

    # Customize plotting layout
    if plot_family == "alpha_shapes":
        alpha_shape = compute_alpha_shapes(mcmc_res['true_families'], net, family_alpha_shape)
        smooth_shape = alpha_shape.buffer(100, resolution=16, cap_style=1, join_style=1, mitre_limit=5.0)
        patch = PolygonPatch(smooth_shape, fc=family_color, ec=family_color, alpha=0.5,
                             fill=True, zorder=-1)

        ax.add_patch(patch)

    # elif plot_family == "color":
    #     fam_sites = np.sum(mcmc_res['true_families'], axis=0, dtype=np.int32)
    #     fam_sites = np.ma.masked_where(fam_sites == 0, fam_sites)
    #     ax.scatter(*all_sites.T, c=fam_sites, s=size, cmap="inferno", zorder=-1)

    if plot_edges:
        plot_triangulation_edges(samples=np.array(zones[end_bi:]), net=net, triangulation="mst", ax=ax)
        # plot_posterior(np.array(zone[end_bi:]), net, ax=ax)

    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])

    if annotate_zones:
        # create list with all zone indices
        indices_zones = [nz - 1] if nz != -1 else range(n_zones)
        print(f'Zone indices: {indices_zones}')

        for zone_index in indices_zones:
            print()

    # getting axes ranges and rounding them
    x_min, x_max = np.min(positions[0, :]), np.max(positions[0, :])
    y_min, y_max = np.min(positions[1, :]), np.max(positions[1, :])
    x_min, x_max = round_int(x_min, 'down'), round_int(x_max, 'up')
    y_min, y_max = round_int(y_min, 'down'), round_int(y_max, 'up')

    # print(f'x range {x_min}--{x_max}')
    # print(f'y range {y_min}--{y_max}')

    # x axis
    ax.set_xlim([x_min, x_max])
    x_ticks = list(np.linspace(x_min, x_max, 6))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{x_tick:.0f}' for x_tick in x_ticks], fontsize=fontsize)

    # y axis
    ax.set_ylim([y_min, y_max])
    y_ticks = list(np.linspace(y_min, y_max, 6))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y_tick:.0f}' for y_tick in y_ticks], fontsize=fontsize)

    [ax.spines[side].set_linewidth(frame_width) for side in ['left', 'bottom', 'right', 'top']]
    plt.rcParams["axes.linewidth"] = frame_width

    # colorbar
    cbar_ticks = list(np.linspace(0, 1, 6))
    cbar_ticklabels = [f'{round(t * 100, 0):.0f}' for t in cbar_ticks]

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="3%", pad=1, pack_start=True)
    fig.add_axes(cax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1),
                                     orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xlabel('Frequency of point in posterior (%)', fontsize=fontsize, labelpad=-75)
    cbar.ax.set_xticklabels(cbar_ticklabels)
    cbar.ax.tick_params(labelsize=fontsize)

    # ax.set_title(title, fontsize=fontsize)

    fig.savefig(fname, bbox_inches='tight', dpi=400)
    # plt.show()


def plot_minimum_spanning_tree(mcmc_res, network, z=0, burn_in=0.2, ts_posterior_freq=0.8,
                               cmap=plt.get_cmap('jet'), fname='minimum_spanning_tree.png'):
    """ This function plots the minimum spanning tree of the sites that are above the posterior frequency threshold.

    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        network (dict): The full network containing all sites.
        z (int): which zone should be plotted?
        burn_in (float): Percentage of first samples which is discarded as burn-in
        ts_posterior_freq (float): threshold for sites to be included in the mst
        fname (str): a path followed by a the name of the file
    """

    fontsize = 24
    line_thickness = 2
    frame_width = 1.5
    size = 5
    plt.rcParams["axes.linewidth"] = frame_width


    fig, ax = plt.subplots(figsize=(15, 10))

    # getting zone data
    zones = mcmc_res['zones']
    zone = zones[z-1]
    n_samples = len(zone)

    # exclude burn-in
    end_bi = math.ceil(n_samples * burn_in)

    # compute posterior frequency of each point in the zone
    posterior_freq = (np.sum(zone[end_bi:], axis=0, dtype=np.int32) / (n_samples - end_bi))

    print(f'Shape posterior frequency: {posterior_freq.shape}')

    # contact points have a higher posterior frequency than passed threshold
    is_contact_point = posterior_freq > ts_posterior_freq
    n_contact_points = np.count_nonzero(is_contact_point == True)

    print(f'Number of contact points: {n_contact_points}')


    # getting locations and distance matrix for those locations from the network
    locations = network['locations']
    dist_mat = network['dist_mat']

    print(f'Shape locations: {locations.shape}')
    print(f'Shape distance matrix: {dist_mat.shape}')

    # subsetting locations, posterior frequencies, and distance matrix to contact points (cp)
    cp_locations = locations[is_contact_point,:]
    cp_posterior_freq = posterior_freq[is_contact_point]
    cp_dist_mat = dist_mat[is_contact_point]
    cp_dist_mat = cp_dist_mat[:,is_contact_point]

    print(f'Shape sub locations: {cp_locations.shape}')
    print(f'Shape sub distance matrix: {cp_dist_mat.shape}')

    if not len(cp_locations) > 3:
        raise Exception(f'Cannot compute minimum spanning tree of network with size {len(cp_locations)}.')

    # computing the minimum spanning tree of contact points
    cp_delaunay = compute_delaunay(cp_locations)
    cp_mst = minimum_spanning_tree(cp_delaunay.multiply(cp_dist_mat))

    # converting minimum spanning tree to boolean array denoting whether contact points are connected
    cp_mst = cp_mst.toarray()
    cp_connections = cp_mst > 0

    # plotting every edge (connections of points) of the network
    for index, connected in np.ndenumerate(cp_connections):
        if connected:

            i1, i2 = index

            # first contact point
            cp1_loc = cp_locations[i1]
            cp1_freq = cp_posterior_freq[i1]

            # second contact point
            cp2_loc = cp_locations[i2]
            cp2_freq = cp_posterior_freq[i2]

            # print(f'Point1: ({vertex1[0]}/{vertex1[1]}) Point2: ({vertex2[0]}/{vertex2[1]})')
            n_fragments = 100
            # computing color gradient
            x = np.linspace(cp1_loc[0], cp2_loc[0], n_fragments)
            y = np.linspace(cp1_loc[1], cp2_loc[1], n_fragments)
            z = np.linspace(cp1_freq, cp2_freq, n_fragments)

            colorline(x, y, z=z, cmap=cmap, norm=plt.Normalize(0, 1),
                      linewidth=line_thickness)
            # ax.plot([cp1_loc[0], cp2_loc[0]], [cp1_loc[1], cp2_loc[1]], 'ro-')



    # getting colors for all points according to their frequency
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    colors = np.array([cmap(norm(freq)) for freq in list(posterior_freq)])


    # plotting contact points
    ax.scatter(locations[is_contact_point,0], locations[is_contact_point,1], c=colors[is_contact_point],
               linewidths=0, edgecolors='black')



    # plotting non contact points
    not_contact_point = np.logical_not(is_contact_point)
    sc = ax.scatter(locations[not_contact_point, 0], locations[not_contact_point, 1], c=colors[not_contact_point])



    # getting axes ranges and rounding them
    x_min, x_max = np.min(cp_locations[:, 0]), np.max(cp_locations[:, 0])
    y_min, y_max = np.min(cp_locations[:, 1]), np.max(cp_locations[:, 1])
    x_min, x_max = round_int(x_min, 'down'), round_int(x_max, 'up')
    y_min, y_max = round_int(y_min, 'down'), round_int(y_max, 'up')


    print(f'x range {x_min}--{x_max}')
    print(f'y range {y_min}--{y_max}')

    # x axis
    ax.set_xlim([x_min, x_max])
    x_step = (x_max - x_min) // 5
    x_ticks = np.arange(x_min, x_max+x_step, x_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks, fontsize=fontsize)

    # y axis
    ax.set_ylim([y_min, y_max])
    y_step = (y_max - y_min) // 5
    y_ticks = np.arange(y_min, y_max+y_step, y_step)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks, fontsize=fontsize)


    # colorbar
    n_ticks = 6 if (ts_posterior_freq * 10) % 2 == 0 else 11
    cbar_ticks = list(np.linspace(0, 1, n_ticks))
    cbar_step = 1 / (n_ticks - 1)
    i_label = int(ts_posterior_freq / cbar_step)
    cbar_ticklabels = [f'{round(t * 100, 0):.0f}' for t in cbar_ticks]
    cbar_ticklabels[i_label] = f'{cbar_ticklabels[i_label]} (ts)'

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="3%", pad=1, pack_start=True)
    fig.add_axes(cax)
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=1),
                                     orientation='horizontal', ticks=cbar_ticks)
    cbar.ax.set_xlabel('Frequency of point in posterior (%)', fontsize=fontsize, labelpad=-75)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.ax.plot([ts_posterior_freq] * 2, [0, 1], 'k')
    cbar.ax.set_xticklabels(cbar_ticklabels)


    fig.savefig(fname, bbox_inches='tight', dpi=400)




def f_score_mle(posterior):
    # Todo: Remove 0:20000
    # Todo: Add contact zone to posterior
    """ This function computes the precison, recall and f score of the maximum likelihood estimate in the posterior

    Args:
        posterior (tuple): the full posterior of a model

    Returns:
        (float): the precision of the model
        (float): the recall of the model
        (float): the f score of the model

    """
    contact_zones_idxs = get_contact_zones(6)
    n_zones = len(contact_zones_idxs)
    posterior[1]['true_zone'] = np.zeros((n_zones, network['n']), bool)
    for k, cz_idxs in enumerate(contact_zones_idxs.values()):
        posterior[1]['true_zone'][k, cz_idxs] = True

    m = max(posterior[1]['step_likelihoods'][0:20000])
    mle_pos = [i for i, j in enumerate(posterior[1]['step_likelihoods']) if j == m][0]

    best_zone = posterior[0][mle_pos]
    true_zone = posterior[1]['true_zone']

    true_positives = np.sum(np.logical_and(best_zone, true_zone), axis=1)[0]
    false_positives = np.sum(np.logical_and(best_zone, np.logical_not(true_zone)), axis=1)[0]
    false_negatives = np.sum(np.logical_and(np.logical_not(best_zone), true_zone), axis=1)[0]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score


def compute_alpha_shapes(sites, net, alpha):

    """Compute the alpha shape (concave hull) of a set of sites
    Args:
        sites (np.array): subset of sites around which to create the alpha shapes (e.g. family, zone, ...)
        net (dict): The full network containing all sites.
        alpha (float): alpha value to influence the gooeyness of the convex hull Smaller numbers don't fall inward
        as much as larger numbers. Too large, and you lose everything!"

    Returns:
        (polygon): the alpha shape"""

    all_sites = net['locations']
    points = all_sites[sites[0]]
    tri = Delaunay(points, qhull_options="QJ Pp")

    edges = set()
    edge_nodes = []

    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = math.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = math.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)

        # Semiperimeter of triangle
        s = (a + b + c) / 2.0

        # Area of triangle by Heron's formula
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)

        if circum_r < 1.0 / alpha:

            add_edge(edges, edge_nodes, points, ia, ib)
            add_edge(edges, edge_nodes, points, ib, ic)
            add_edge(edges, edge_nodes, points, ic, ia)


    m = geometry.MultiLineString(edge_nodes)

    triangles = list(polygonize(m))
    polygon = cascaded_union(triangles)

    return polygon


def plot_trace_recall_precision(mcmc_res, burn_in=0.2, recall=True, precision=True, fname='trace_recall_precision'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        burn_in (float): First n% of samples are burn-in
        recall (boolean): plot recall?
        precision (boolean): plot precision?
        fname (str): a path followed by a the name of the file
    """

    fontsize = 24

    line_thickness = 2
    frame_width = 1.5
    plt.rcParams["axes.linewidth"] = frame_width


    fig, ax = plt.subplots(figsize=(15, 10))
    col = get_colors()

    # Recall
    if recall:
        y = mcmc_res['recall']
        x = range(len(y))
        ax.plot(x, y, lw=line_thickness, color=col['trace']['recall'], label='Recall')

    # Precision
    if precision:
        y = mcmc_res['precision']
        x = range(len(y))
        ax.plot(x, y, lw=line_thickness, color=col['trace']['precision'], label='Precision')

    ax.set_ylim(bottom=0)


    # Find index of first sample after burn-in
    end_bi = math.ceil(len(y) * burn_in)
    end_bi_label = math.ceil(len(y) * (burn_in - 0.04))

    color_burn_in = 'grey'
    ax.axvline(x=end_bi, lw=line_thickness, color=color_burn_in, linestyle='--')
    plt.text(end_bi_label, 0.5, 'Burn-in', rotation=90, size=fontsize, color=color_burn_in)

    xmin, xmax, xstep = 0, 1000, 200
    ax.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)

    ymin, ymax, ystep = 0, 1, 0.2
    ax.set_ylim([ymin, ymax+(ystep/2)])
    yticks = np.arange(ymin, ymax+ystep, ystep)
    ax.set_yticks(yticks)
    yticklabels = [f'{ytick:.1f}' for ytick in yticks]
    yticklabels[0] = '0'
    ax.set_yticklabels(yticklabels, fontsize=fontsize)


    ax.set_xlabel('Sample', fontsize=fontsize, fontweight='bold')
    # ax.set_title(title)

    ax.legend(loc=4, prop={'size': fontsize}, frameon=False)

    fig.savefig(fname, bbox_inches='tight', dpi=400)

    #plt.show()


def plot_dics(dics):
    """This function plots dics. What did you think?
    Args:
        dics(dict): A dict of DICs from different models

    """
    fig, ax = plt.subplots()
    col = get_colors()

    x = range(len(dics))
    y = dics.values()

    ax.plot(x, y, lw=0.75, color=col['trace']['recall'], label='DIC')
    names = dics.keys()
    plt.xticks(x, names)
    plt.show()


def plot_trace_lh(mcmc_res, burn_in=0.2, true_lh=True, fname='trace_likelihood.png'):
    """
    Function to plot the trace of the MCMC chains both in terms of likelihood and recall
    Args:
        mcmc_res (dict): the output from the MCMC neatly collected in a dict
        burn_in: (float): First n% of samples are burn-in
        true_lh (boolean): Visualize the true likelihood
        fname (str): a path followed by a the name of the file
    """

    fontsize = 24

    line_thickness = 2
    frame_width = 1.5
    plt.rcParams["axes.linewidth"] = frame_width


    fig, ax = plt.subplots(figsize=(15, 10))
    col = get_colors()
    n_zones = len(mcmc_res['zones'])



    y = mcmc_res['lh']
    x = range(len(y))
    color = 'red' # col['trace']['lh']

    if true_lh:
        ax.axhline(y=mcmc_res['true_lh'], xmin=x[0], xmax=x[-1], lw=2, color=color, linestyle='-.', label='True')
    ax.plot(x, y, lw=line_thickness, color=color, linestyle='-', label='Predicted')





    y_min, y_max = min(y), max(y)


    # Find index of first sample after burn-in
    end_bi = math.ceil(len(x) * burn_in)
    end_bi_label = math.ceil(len(x) * (burn_in - 0.03))

    color_burn_in = 'grey'
    ax.axvline(x=end_bi, lw=line_thickness, color=color_burn_in, linestyle='--')
    ypos_label = y_min + (y_max - y_min) / 2
    ax.text(end_bi_label, ypos_label, 'Burn-in', rotation=90, size=fontsize, color=color_burn_in)

    y_min, y_max = round_int(y_min, 'down'), round_int(y_max, 'up')
    y_step = (y_max - y_min) // 6
    ax.set_ylim([y_min, y_max])
    y_ticks = np.arange(y_min, y_max+y_step, y_step)
    ax.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    ax.set_yticklabels(yticklabels, fontsize=fontsize)


    xmin, xmax, xstep = 0, 1000, 200
    ax.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)


    ax.set_xlabel('Sample', fontsize=fontsize, fontweight='bold')

    if n_zones == 1:
        yaxis_label = "Likelihood of simulated area (log)"
    else:
        yaxis_label = "Likelihood of simulated areas (log)"
    ax.set_ylabel(yaxis_label, fontsize=fontsize, fontweight='bold')

    # ax.set_title('Trace of the likelihood')

    ax.legend(loc=4, prop={'size': fontsize}, frameon=False)


    fig.savefig(fname, bbox_inches='tight', dpi=400)


    # plt.show()


def plot_histogram_weights(mcmc_res, feature):
    """
        Plots the trace for weight samples
        Args:
            mcmc_res (dict): the output from the MCMC neatly collected in a dict
            feature (int): plot weight for which feature?
        """
    fig, ax = plt.subplots()
    col = get_colors()
    n_weights = len(mcmc_res['weights'])

    # Global weight
    weights = []
    for w in mcmc_res['weights']:
        weights.append(w[feature][1])

    y = weights
    x = range(len(y))
    ax.hist(y, bins=None, range=None)
    #ax.plot(x, y, lw=0.75, color=col['trace']['recall'], label='Weights')

    plt.show()


def plot_auto_sim(mcmc_res):
    """
    Function to plot the autosimilarity of consecutive samples in MCMC chains
    Args:
           mcmc_res (dict): the output from the MCMC neatly collected in a dict
    """

    fig, ax = plt.subplots()
    col = get_colors()

    for z in mcmc_res['zones'][0:1]:
        y = []
        for t in range(1, 500):

            y.append(zones_autosimilarity(z, t))
            x = range(len(y))
            ax.plot(x, y, lw=1, color=col['trace']['lh'])

    ax.set_xlabel('lag')
    ax.set_ylabel('Autosimilarity')
    ax.set_title('Autosimilarity plot')
    plt.show()


def plot_proximity_graph(net, zone, graph, triang_type):
    """ This function generates a plot of the entire network, the current zone and its proximity graph
    Args:
        net (dict): The full network containing all sites.
        zone (np.array): The current zone (boolean array).
        graph (dict): Either a delaunay triangulation or a minimum spanning tree of the zone.
        triang_type (str): Type of the triangulation, either "delaunay" or "mst"
    """
    # Initialize plot
    fig, ax = plt.subplots()
    col = get_colors()

    all_sites = net['locations']
    v = zone.nonzero()[0]
    zone_sites = net['locations'][v]

    # Plot background and zones
    size = 4
    bg = ax.scatter(*all_sites.T, s=size, color=col['zones']['background_nodes'])
    zo = ax.scatter(*zone_sites.T, s=size*3, color=col['zones']['triangulation'])

    # Collect all edges in the triangulation in a line collection
    lines = []
    for e in graph.es:
        lines.append([tuple(zone_sites[e.tuple[0]]), tuple(zone_sites[e.tuple[1]])])

    lc = LineCollection(lines, colors=col['zones']['triangulation'], linewidths=0.5)
    ax.add_collection(lc)

    # Customize plotting layout
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    bbox = bounding_box(all_sites)
    offset_x = (bbox['x_max'] - bbox['x_min'])*0.03
    offset_y = (bbox['y_max'] - bbox['y_min']) * 0.03
    plt.xlim(bbox['x_min']-offset_x, bbox['x_max']+offset_x)
    plt.ylim(bbox['y_min']-offset_y, bbox['y_max']+offset_y)

    if triang_type == "delaunay":
        triang_legend = "Delaunay Triangulation"
    elif triang_type == "mst":
        triang_legend = "Minimum Spanning Tree"

    # Add legend
    ax.legend([bg, zo, lc], ['All sites', 'Sites in contact zone', triang_legend], frameon=False, fontsize=15)
    plt.show()


def plot_histogram_empirical_geo_prior(e_g_prior, g_prior_type):
    """

    Args:
        e_g_prior (dict): the empirical geo-prior
        g_prior_type (): the type of geo-prior, either "complete", "delaunay" or "mst"
    """
    # Load the color palette for plotting

    fig, ax = plt.subplots()
    col = get_colors()

    d = e_g_prior[g_prior_type]['empirical']
    ax.hist(d, 80, normed=1, facecolor='grey', edgecolor='white', alpha=0.75)

    # add a 'best fit' line
    x = np.linspace(0, max(d), 500)
    shape, loc, scale = e_g_prior[g_prior_type]['fitted_gamma']
    y = gamma.pdf(x, shape, loc, scale)
    ax.plot(x, y, color=col['histogram']['fitted_line'], linewidth=2)
    ax.set_facecolor(col['histogram']['background'])

    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.xlabel('Length [km]', labelpad=20)
    plt.ylabel('Probability', labelpad=40)
    if g_prior_type == "complete":
        title_plot = "Complete Graph"
    if g_prior_type == "delaunay":
        title_plot = " Delaunay Graph"
    if g_prior_type == "mst":
        title_plot = "Minimum Spanning Tree"
    plt.title('Average length of edges in the %s' % title_plot)
    plt.grid(False)
    plt.show()


def plot_geo_prior_vs_feature_lh(mcmc_res, r=0, burn_in=0.2):
    """
        Function to plot the Likelihood and the prior for each chain in the MCMC
        Args:
            mcmc_res (dict): the output from the MCMC neatly collected in a dict
            r (int): which run?
            burn_in: (float): First n% of samples are burn-in
        """
    colors = get_colors()['zones']['in_zones']
    fig, ax = plt.subplots()

    # Where to put the label
    x_mid = []
    n_zones = len(mcmc_res['lh'][r])

    for c in range(n_zones):

        if n_zones == 1:
            label_lh = 'Likelihood'
            label_prior = 'Prior'

        else:
            label_lh = 'Likelihood (zone' + str(c) + ')'
            label_prior = 'Prior (zone' + str(c) + ')'

        y = mcmc_res['lh'][r][c]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=colors[c], label=label_lh)

        y = mcmc_res['prior'][r][c]
        x = range(len(y))
        ax.plot(x, y, lw=0.75, color=colors[c], linestyle='--', label=label_prior)
        x_mid.append(max(y) - min(y))  # Where to put the label?

    # Find index of first sample after burn-in
    end_bi = math.ceil(len(y) * burn_in)
    end_bi_label = math.ceil(len(y) * (burn_in - 0.03))

    ax.axvline(x=end_bi, lw=1, color="grey", linestyle='--')
    ax.text(end_bi_label, max(x_mid), 'Burn-in', rotation=90, size=8)

    ax.set_xlabel('Sample')
    ax.legend(loc=4)
    plt.show()


def plot_zone_size_vs_ll(mcmc_res, lh_type, mode, individual=False):

    colors = get_colors()['zones']['in_zones']
    fig, ax = plt.subplots()

    if individual:

        for c in range(len(mcmc_res['zones'])):
            lh = []
            for _ in (range(201)):
                lh.append([])

            pt_in_zone = []
            for z in mcmc_res['zones'][c]:
                pt_in_zone.append(np.sum(z))

            for idx in pt_in_zone:
                lh[idx].append(mcmc_res[lh_type][c][idx])
            lh = lh[5:]
            sumstat_lh = []

            for l in lh:
                if mode == 'mean':
                    sumstat_lh.append(np.mean(l))
                elif mode == 'std':
                    sumstat_lh.append(np.std(l))
                elif mode == 'count':
                    sumstat_lh.append(len(l))

            y = sumstat_lh
            x = range(5, len(sumstat_lh) + 5)
            ax.plot(x, y, lw=0.75, color=colors[c])

    else:
        for c in range(len(mcmc_res['zones'])):
            lh = []
            for _ in (range(201)):
                lh.append([])

        for c in range(len(mcmc_res['zones'])):
            pt_in_zone = []
            for z in mcmc_res['zones'][c]:
                pt_in_zone.append(np.sum(z))

            for idx in pt_in_zone:
                lh[idx].append(mcmc_res[lh_type][c][idx])
        lh = lh[5:]
        sumstat_lh = []
        for l in lh:
            if mode == 'mean':
                sumstat_lh.append(np.mean(l))
            elif mode == 'std':
                sumstat_lh.append(np.std(l))
            elif mode == 'count':
                sumstat_lh.append(len(l))

        y = sumstat_lh
        x = range(5, len(sumstat_lh) + 5)
        ax.plot(x, y, lw=0.75, color=colors[0])

    ax.set_xlabel('Zone size')
    ax.set_xlabel('Mean geo prior')
    plt.title('Number of samples per zone size')
    plt.show()


def plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2, true_zone=True, fname='zone_size_over_time'):
    """
         Function to plot the zone size in the posterior
         Args:
             mcmc_res (dict): the output from the MCMC neatly collected in a dict
             r (int): which run?
             burn_in: (float): First n% of samples are burn-in
             fname (str): a path followed by a the name of the file
         """

    colors = get_colors()['zones']['in_zones']

    fontsize = 24
    line_thickness = 2
    frame_width = 1.5
    plt.rcParams["axes.linewidth"] = frame_width

    fig, ax = plt.subplots(figsize=(15, 10))

    x_mid = [] # label position
    y_max = 0 # y range
    n_zones = len(mcmc_res['zones'])

    for c in range(n_zones):
        size = []

        if n_zones == 1:
            label = 'True'
        else:
            label = f'True (Zone {c})'

        for z in mcmc_res['zones'][c]:
            size.append(np.sum(z))

        x = range(len(size))
        if true_zone:
            true_size = np.sum(mcmc_res['true_zones'][c])
            ax.axhline(y=true_size, xmin=x[0], xmax=x[-1], lw=line_thickness, color=colors[c], linestyle='-.',
                       label=label)

        ax.plot(x, size, lw=line_thickness, color=colors[c], label="Predicted")

        max_size, min_size = max(size), min(size)
        y_max = max_size if max_size > y_max else y_max
        x_mid.append(max_size - min_size)



    # Find index of first sample after burn-in
    end_bi = math.ceil(len(x) * burn_in)
    end_bi_label = math.ceil(len(x) * (burn_in - 0.03))

    color_burn_in = 'grey'
    ax.axvline(x=end_bi, lw=line_thickness, color=color_burn_in, linestyle='--')
    ax.text(end_bi_label, max(x_mid), 'Burn-in', rotation=90, size=fontsize, color=color_burn_in)



    ax.set_ylim(bottom=0)


    xmin, xmax, xstep = 0, 1000, 200
    ax.set_xlim([xmin, xmax])
    xticks = np.arange(xmin, xmax+xstep, xstep)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=fontsize)


    y_min, y_max = 0, round_int(y_max, 'up')
    y_step = (y_max - y_min) // 6
    ax.set_ylim([y_min, y_max])
    y_ticks = np.arange(y_min, y_max+y_step, y_step)
    ax.set_yticks(y_ticks)
    yticklabels = [f'{y_tick:.0f}' for y_tick in y_ticks]
    yticklabels[0] = '0'
    ax.set_yticklabels(yticklabels, fontsize=fontsize)


    ax.set_xlabel('Sample', fontsize=fontsize, fontweight='bold')
    ax.set_ylabel('Zone size', fontsize=fontsize, fontweight='bold')

    # ax.set_title(title)

    ax.legend(loc=4, prop={'size': fontsize}, frameon=False)

    fig.savefig(fname, bbox_inches='tight', dpi=400)

    #plt.show()
    
    


def plot_gamma_parameters(ecdf):

    figure, ax = plt.subplots()
    loc = []
    scale = []
    for e in ecdf:

        loc.append(ecdf[e]['mst']['fitted_gamma'][0])
        scale.append(ecdf[e]['mst']['fitted_gamma'][2])

    max_loc = max(loc)
    max_scale = max(scale)

    norm_loc = []
    for l in loc:
        norm_loc.append(l/max_loc)

    norm_scale = []
    for s in scale:
        norm_scale.append(s/max_scale)

    x = range(len(norm_loc))
    ax.plot(x, norm_loc, lw=0.75, color="red")
    ax.plot(x, norm_scale, lw=0.75, color="blue")

    plt.show()


def plot_correlation_weights(mcmc_res,  burn_in=0.2, which_weight="global"):
    """This function plots the correlation between the mean of the estimated weights and the true weights
    Args:
        mcmc_res(dict):  the output from the MCMC, neatly collected in a dict
        burn_in (float): ratio of initial samples that are discarded as burn-in
        which_weight(str): compute correlation for which weight? ("global", "contact", "inheritance")
    """
    if which_weight == "global":
        weight_idx = 0

    elif which_weight == "contact":
        weight_idx = 1

    elif which_weight == "inheritance":
        weight_idx = 2

    else:
        raise ValueError('weight must be "global", "contact" or "inheritance" ')

    fig, ax = plt.subplots()

    # Find index of first sample after burn-in

    end_bi = math.ceil(len(mcmc_res['weights']) * burn_in)
    #end = math.ceil(len(mcmc_res['weights']) * 0.005)

    weights = np.asarray(mcmc_res['weights'][end_bi:])
    w_est = weights[:, :, weight_idx]
    w_mean_est = w_est.mean(axis=0)

    w_true = mcmc_res['true_weights'][:, weight_idx]
    slope, intercept, r_value, p_value, std_err = linregress(w_true, w_mean_est)
    line = slope * w_true + intercept

    # todo: compute range of
    #
    # p_range_per_feature = []
    # for f in range(len(mcmc_res['true_p_global'])):
    #     p_range_per_feature.append(mcmc_res['true_p_global'][f])

    ax.plot(w_true, w_mean_est, 'o')
    ax.plot(w_true, line)

    ax.set_aspect('equal')
    ax.scatter(w_true, w_mean_est)
    ax.set_xlabel('True weights')
    ax.set_ylabel('Mean estimated weights')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.show()


# todo: finish
def plot_correlation_p(mcmc_res,  which_p, burn_in=0.2, which_nr=0, which_cat=0):
    """This function plots the correlation between the mean of the estimated weights and the true weights
    Args:
        mcmc_res(dict):  the output from the MCMC, neatly collected in a dict
        burn_in (float): ratio of initial samples that are discarded as burn-in
        which_p(str): compute correlation for "p_zones" or "p_families"?
        which_nr(int): in case of many zones or families, which one should be visualized? (default=0)
        which_cat(int): which category?
    """
    if which_p == "p_zones" or which_p == "p_families":
        p_est = mcmc_res[which_p][which_nr]
        p_true = mcmc_res["true_" + str(which_p)][which_nr]

    else:
        raise ValueError('which_p must be "p_zones", or "p_families" ')

    p_est_out = []
    for p in p_est:
        f_out = []
        for f in p:
            f_out.append(f[which_cat])
        p_est_out.append(f_out)
    p_est = p_est_out

    p_true_out = []

    for f in p_true:
        p_true_out.append(f[which_cat])
    p_true = np.asarray(p_true_out)

    fig, ax = plt.subplots()

    # Find index of first sample after burn-in

    end_bi = math.ceil(len(p_est) * burn_in)
    p_est = np.asarray(p_est[end_bi:])

    p_mean_est = p_est.mean(axis=0)

    slope, intercept, r_value, p_value, std_err = linregress(p_true, p_mean_est)
    line = slope * p_true + intercept

    ax.plot(p_true, p_mean_est, 'o')
    ax.plot(p_true, line)

    ax.set_aspect('equal')
    ax.scatter(p_true, p_mean_est)
    ax.set_xlabel('True p')
    ax.set_ylabel('Mean estimated p')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.show()


def plot_parallel_posterior(post):
    """ This function first sorts the posterior of parallel zones in mcmc_res, such that the first list comprises
    the posterior of the largest zone, the second the posterior of the second largest zone, ... , and then
    creates a boxplot of the resulting sorted posteriors of each zone
    Args:
        post (np.ndarray): the posterior of all parallel zones
    """
    # Get color palette
    colors = get_colors()

    # Sort the array
    z_array = np.vstack(post)
    z_array[::-1].sort(axis=0)


    # make boxplot
    fig, ax = plt.subplots()
    ax.set_title('(log)-Posterior of parallel zones')
    ax.set_xlabel('Parallel zone')
    ax.set_ylabel('(log)-Posterior')

    ax.boxplot(z_array.tolist(), showcaps=False, showmeans=False, patch_artist=True,
               widths=0.2,  medianprops=dict(color=colors['boxplot']['median']),
               whiskerprops=dict(color=colors['boxplot']['whiskers']),
               boxprops=dict(facecolor=colors['boxplot']['box'], linewidth=0.1,
                             color=colors['boxplot']['box']))

    plt.show()


if __name__ == '__main__':
    from src.util import load_from
    from src.config import NETWORK_PATH, FEATURES_PATH, LOOKUP_TABLE_PATH,ECDF_GEO_PATH
    from src.preprocessing import get_contact_zones

    TEST_SAMPLING_DIRECTORY = 'data/results/test/zones/2018-10-02_10-03-13/'

    # Zone, ease and number of runs
    zone = 6
    ease = 1
    n_runs = 1

    mcmc_res = {'lh': [[] for _ in range(n_runs)],
                'prior': [[] for _ in range(n_runs)],
                'recall': [[] for _ in range(n_runs)],
                'precision': [[] for _ in range(n_runs)],
                'zones': [[] for _ in range(n_runs)],
                'posterior': [[] for _ in range(n_runs)],
                'lh_norm': [[] for _ in range(n_runs)],
                'posterior_norm': [[] for _ in range(n_runs)],
                'true_zones':[[] for _ in range(n_runs)]}

    for r in range(n_runs):

        # Load the MCMC results
        sample_path = TEST_SAMPLING_DIRECTORY + 'zone_z' + str(zone) + '_e' + \
                      str(ease) + '_' + str(r) + '.pkl'

        samples = load_from(sample_path)
        # Todo:  Handle parallel zones
        # Todo: Run with burn-in
        for t in range(len(samples['sample_zones'])):

            # Zones, likelihoods and priors
            zones = np.asarray(samples['sample_zones'][t])

            mcmc_res['zones'][r].append(zones)
            mcmc_res['lh'][r].append(samples['sample_likelihoods'][t])
            mcmc_res['prior'][r].append(samples['sample_priors'][t])

            # Normalized likelihood and posterior

            posterior = [x + y for x, y in zip(samples['sample_likelihoods'][t], samples['sample_priors'][t])]
            true_posterior = samples['true_zones_lls'][t] + samples['true_zones_priors'][t]
            mcmc_res['posterior'][r].append(posterior)
            lh_norm = np.asarray(samples['sample_likelihoods'][t]) / samples['true_zones_lls'][t]
            posterior_norm = np.asarray(posterior) / true_posterior

            # Recall and precision
            true_z = samples['true_zones'][t]
            n_true = np.sum(true_z)

            # zones = zones[:, 0, :]
            intersections = np.minimum(zones, true_z)
            total_recall = np.sum(intersections, axis=1)/n_true
            precision = np.sum(intersections, axis=1)/np.sum(zones, axis=1)

            # Store to dict
            mcmc_res['lh_norm'][r].append(lh_norm)
            mcmc_res['posterior_norm'][r].append(posterior_norm)
            mcmc_res['recall'][r].append(total_recall)
            mcmc_res['precision'][r].append(precision)
            mcmc_res['true_zones'][r].append(true_z)

    network = load_from(NETWORK_PATH)

    #print(mcmc_res['posterior'])
    #for u in mcmc_res['posterior']:
    #     plot_parallel_posterior(u)
    # print(mcmc_res['zones'][0])
    #print(mcmc_res['zones'][0][0][0])
    #print(len(mcmc_res['zones'][0][0]))
    #

    plot_posterior_frequency(mcmc_res['zones'], network, nz=0, r=0, burn_in=0.2)

    # plot_geo_prior_vs_feature_lh(mcmc_res, r=0, burn_in=0.2 )

    # plot_zone_size_over_time(mcmc_res, r=0, burn_in=0.2)
    #

    #plot_zone_size_vs_ll(mcmc_res, 'geo_prior', mode='mean', individual=True)

    #ecdf = load_from(ECDF_GEO_PATH)
    #plot_gamma_parameters(ecdf)
    #print(np.sum(mcmc_res['zones'][1][-1][0]))
    #for u in range(len(mcmc_res['zones'])):
    #    plot_zone(mcmc_res['zones'][u][-1][0], network)



    #plot_auto_sim(mcmc_res)
    # POSTERIOR_PATH = 'data/results/test/2018-05-13/sampling_e1_a1_mgenerative_2.pkl'
    # posterior = load_from(POSTERIOR_PATH)
    # plt.close()
    #
    # # Load data
    # network = load_from(NETWORK_PATH)
    # features = load_from(FEATURES_PATH)
    # ll_lookup = load_from(LOOKUP_TABLE_PATH)
    # locations = network['locations']
    # adj_mat = network['adj_mat']

    #
    # # Plot posterior frequency
    # plot_posterior_frequency(posterior[0], network)
    #
    # # Plot posterior density
    # plot_posterior_density(posterior[0][1:30], network)
    #
    # # Compute F-statistics
    # #f = f_score_mle(posterior)
    #
    # # Plot alpha shapes
    # plot_alpha_shapes(posterior[0][1], network, alpha=0.0000000008)

    # Plot the empirical distribution of the geo-prior




    #d = [4000, 20000, 30000, 100000]

    # for s in d:
    #     print(gamma.mean(*x_3))
    #     geo_prior = np.log(1-gamma.cdf(s, *x_1))
    #     print(geo_prior)


    # Plot locations as scatter
    # plt.scatter(*locations.T, s=1, lw=0)
    #
    # # Plot edges weighted by edge ll
    # edges = np.argwhere(adj_mat)
    # n = len(locations)
    # edges_ll = []
    # for v1, v2 in edges:
    #     feature_counts = features[v1] + features[v2]
    #     ll = 0
    #     for f_idx, f_count in enumerate(feature_counts):
    #         ll += ll_lookup[f_idx][2][f_count]
    #     edges_ll.append(ll)
    # edges_ll = np.array(edges_ll)
    # edges_ll /= np.max(edges_ll)
    #
    # rgba_colors = [(0.4, 0.05, 0.2, ll**4) for ll in edges_ll]
    # lines = LineCollection(locations[edges], colors=rgba_colors, linewidth=1.)
    #
    # plt.axes().add_collection(lines)
    #
    # plt.axes().set_xticks([])
    # plt.axes().set_yticks([])
    # plt.tight_layout(True)
    #plt.show()
