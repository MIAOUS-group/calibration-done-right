# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import exit
import numpy as np
from scipy import optimize
import sys

# TODO
# sys.argv[1] should be the root
# with root-result_lite.csv.bz2 the result
# and .stats.csv
# root.slices a slice mapping - done
# root.cores a core + socket mapping - done -> move to analyse csv ?
#
# Facet plot with actual dot cloud + plot the linear regression
# each row is a slice
# each row is an origin core
# each column a helper core if applicable


stats = pd.read_csv(sys.argv[1] + ".stats.csv",
                    dtype={
                        "main_core": np.int8,
                        "helper_core": np.int8,
                        # "address": int,
                        "hash": np.int8,
                        # "time": np.int16,
                        "clflush_remote_hit": np.float64,
                        "clflush_shared_hit": np.float64,
                        # "clflush_miss_f": np.int32,
                        # "clflush_local_hit_f": np.int32,
                        "clflush_miss_n": np.float64,
                        "clflush_local_hit_n": np.float64,
                        # "reload_miss": np.int32,
                        # "reload_remote_hit": np.int32,
                        # "reload_shared_hit": np.int32,
                        # "reload_local_hit": np.int32
                    }
                    )

slice_mapping = pd.read_csv(sys.argv[1] + ".slices.csv")
core_mapping = pd.read_csv(sys.argv[1] + ".cores.csv")

print(core_mapping.to_string())
print(slice_mapping.to_string())

print("core {} is mapped to '{}'".format(4, repr(core_mapping.iloc[4])))

min_time_miss = stats["clflush_miss_n"].min()
max_time_miss = stats["clflush_miss_n"].max()


def remap_core(key):
    def remap(core):
        remapped = core_mapping.iloc[core]
        return remapped[key]

    return remap


stats["main_socket"] = stats["main_core"].apply(remap_core("socket"))
stats["main_core_fixed"] = stats["main_core"].apply(remap_core("core"))
stats["main_ht"] = stats["main_core"].apply(remap_core("hthread"))
stats["helper_socket"] = stats["helper_core"].apply(remap_core("socket"))
stats["helper_core_fixed"] = stats["helper_core"].apply(remap_core("core"))
stats["helper_ht"] = stats["helper_core"].apply(remap_core("hthread"))

# slice_mapping = {3: 0, 1: 1, 2: 2, 0: 3}

stats["slice_group"] = stats["hash"].apply(lambda h: slice_mapping["slice_group"].iloc[h])

graph_lower_miss = int((min_time_miss // 10) * 10)
graph_upper_miss = int(((max_time_miss + 9) // 10) * 10)

print("Graphing from {} to {}".format(graph_lower_miss, graph_upper_miss))

g_ = sns.FacetGrid(stats, col="main_core_fixed", row="slice_group")

g_.map(sns.distplot, 'clflush_miss_n', bins=range(graph_lower_miss, graph_upper_miss), color="b")
#g.map(sns.scatterplot, 'slice_group', 'clflush_local_hit_n', color="g")
plt.show()



# also explains remote
# shared needs some thinking as there is something weird happening there.

#
# M 0 1 2 3 4 5 6 7
#


print(stats.head())

num_core = len(stats["main_core_fixed"].unique())
print("Found {}".format(num_core))


def miss_topology(main_core_fixed, slice_group, C, h):
    return C + h * abs(main_core_fixed - slice_group) + h * abs(slice_group + 1)

def miss_topology_df(x, C, h):
    return x.apply(lambda x, C, h: miss_topology(x["main_core_fixed"], x["slice_group"], C, h), args=(C, h), axis=1)



res_miss = optimize.curve_fit(miss_topology_df, stats[["main_core_fixed", "slice_group"]], stats["clflush_miss_n"])
print("Miss topology:")
print(res_miss)


memory = -1
gpu_if_any = num_core


def exclusive_hit_topology_gpu(main_core, slice_group, helper_core, C, h1, h2):
    round_trip = gpu_if_any - memory

    if slice_group <= num_core/2:
        # send message towards higher cores first
        if helper_core < slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(round_trip - (helper_core - memory))
        else:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - slice_group)
    else:
        # send message toward lower cores first
        if helper_core > slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - memory)
        else:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - slice_group)
    return r


def exclusive_hit_topology_gpu_df(x, C, h1, h2):
    return x.apply(lambda x, C, h1, h2: exclusive_hit_topology_gpu(x["main_core_fixed"], x["slice_group"], x["helper_core_fixed"], C, h1, h2), args=(C, h1, h2), axis=1)


def exclusive_hit_topology_gpu2(main_core, slice_group, helper_core, C, h1, h2):
    round_trip = gpu_if_any + 1 - memory

    if slice_group <= num_core/2:
        # send message towards higher cores first
        if helper_core < slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(round_trip - (helper_core - memory))
        else:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - slice_group)
    else:
        # send message toward lower cores first
        if helper_core > slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - memory)
        else:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - slice_group)
    return r


def exclusive_hit_topology_gpu2_df(x, C, h1, h2):
    return x.apply(lambda x, C, h1, h2: exclusive_hit_topology_gpu2(x["main_core_fixed"], x["slice_group"], x["helper_core_fixed"], C, h1, h2), args=(C, h1, h2), axis=1)


# unlikely
def exclusive_hit_topology_nogpu(main_core, slice_group, helper_core, C, h1, h2):
    round_trip = (num_core-1) - memory

    if slice_group <= num_core/2:
        # send message towards higher cores first
        if helper_core < slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(round_trip - (helper_core - memory))
        else:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - slice_group)
    else:
        # send message toward lower cores first
        if helper_core > slice_group:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - memory)
        else:
            r = C + h1 * abs(main_core - slice_group) + h2 * abs(helper_core - slice_group)
    return r


def exclusive_hit_topology_nogpu_df(x, C, h1, h2):
    return x.apply(lambda x, C, h1, h2: exclusive_hit_topology_nogpu(x["main_core_fixed"], x["slice_group"],  x["helper_core_fixed"], C, h1, h2), args=(C, h1, h2), axis=1)


#res_no_gpu = optimize.curve_fit(exclusive_hit_topology_nogpu_df, stats[["main_core_fixed", "slice_group", "helper_core_fixed"]], stats["clflush_remote_hit"])
#print("Exclusive hit topology (No GPU):")
#print(res_no_gpu)

res_gpu = optimize.curve_fit(exclusive_hit_topology_gpu_df, stats[["main_core_fixed", "slice_group", "helper_core_fixed"]], stats["clflush_remote_hit"])
print("Exclusive hit topology (GPU):")
print(res_gpu)

#res_gpu2 = optimize.curve_fit(exclusive_hit_topology_gpu2_df, stats[["main_core_fixed", "slice_group", "helper_core_fixed"]], stats["clflush_remote_hit"])
#print("Exclusive hit topology (GPU2):")
#print(res_gpu2)



def remote_hit_topology_2(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return C + h * abs(main_core - slice_group) + h * abs(slice_group - helper_core) + h * abs(helper_core - main_core)


def shared_hit_topology_1(x, C, h):
    main_core = x["main_core_fixed"]
    slice_group = x["slice_group"]
    helper_core = x["helper_core_fixed"]
    return C + h * abs(main_core - slice_group) + h * max(abs(slice_group - main_core), abs(slice_group - helper_core))


def plot_func(function, *params):
    def plot_it(x, **kwargs):
#        plot_x = []
#        plot_y = []
#        for x in set(x):
#            plot_y.append(function(x, *params))
#            plot_x = x
        print(x)
        plot_y = function(x, *params)
        sns.lineplot(x, plot_y, **kwargs)
    return plot_it

stats["predicted_miss"] = miss_topology_df(stats, *(res_miss[0]))

figure_median_I = sns.FacetGrid(stats, col="main_core_fixed")
figure_median_I.map(sns.scatterplot, 'slice_group', 'clflush_miss_n', color="b")
figure_median_I.map(sns.lineplot, 'slice_group', 'predicted_miss', color="b")
figure_median_I.set_titles(col_template="$A$ = {col_name}")
figure_median_I.tight_layout()

import tikzplotlib

tikzplotlib.save("fig-median-I.tex", axis_width=r'0.175\textwidth', axis_height=r'0.25\textwidth')
plt.show()

#stats["predicted_remote_hit_no_gpu"] = exclusive_hit_topology_nogpu_df(stats, *(res_no_gpu[0]))
stats["predicted_remote_hit_gpu"] = exclusive_hit_topology_gpu_df(stats, *(res_gpu[0]))
#stats["predicted_remote_hit_gpu2"] = exclusive_hit_topology_gpu_df(stats, *(res_gpu2[0]))


stats_A0 = stats[stats["main_core_fixed"] == 0]
figure_median_E_A0 = sns.FacetGrid(stats_A0, col="slice_group")
figure_median_E_A0.map(sns.scatterplot, 'helper_core_fixed', 'clflush_remote_hit', color="r")
figure_median_E_A0.map(sns.lineplot, 'helper_core_fixed', 'predicted_remote_hit_gpu', color="r")
figure_median_E_A0.set_titles(col_template="$S$ = {col_name}")

tikzplotlib.save("fig-median-E-A0.tex", axis_width=r'0.175\textwidth', axis_height=r'0.25\textwidth')
plt.show()

g = sns.FacetGrid(stats, row="main_core_fixed")

g.map(sns.scatterplot, 'slice_group', 'clflush_miss_n', color="b")
g.map(sns.scatterplot, 'slice_group', 'clflush_local_hit_n', color="g")

g0 = sns.FacetGrid(stats, row="slice_group")

g0.map(sns.scatterplot, 'main_core_fixed', 'clflush_miss_n', color="b")
g0.map(sns.scatterplot, 'main_core_fixed', 'clflush_local_hit_n', color="g") # this gives away the trick I think !
# possibility of sending a general please discard this everyone around one of the ring + wait for ACK - direction depends on the core.



g2 = sns.FacetGrid(stats, row="main_core_fixed", col="slice_group")
g2.map(sns.scatterplot, 'helper_core_fixed', 'clflush_remote_hit', color="r")
g2.map(sns.lineplot, 'helper_core_fixed', 'predicted_remote_hit_gpu', color="r")
#g2.map(sns.lineplot, 'helper_core_fixed', 'predicted_remote_hit_gpu2', color="g")
#g2.map(sns.lineplot, 'helper_core_fixed', 'predicted_remote_hit_no_gpu', color="g")
#g2.map(plot_func(exclusive_hit_topology_nogpu_df, *(res_no_gpu[0])), 'helper_core_fixed', color="g")

g3 = sns.FacetGrid(stats, row="main_core_fixed", col="slice_group")
g3.map(sns.scatterplot, 'helper_core_fixed', 'clflush_shared_hit', color="y")

# more ideas needed

plt.show()
