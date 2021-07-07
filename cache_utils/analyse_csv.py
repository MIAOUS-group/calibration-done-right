# SPDX-FileCopyrightText: 2021 Guillaume DIDIER
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: MIT

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
from sys import exit
import wquantiles as wq
import numpy as np

from functools import partial

import sys

# For cyber cobay sanity check :
from gmpy2 import popcount
functions_i9_9900 = [
             0b1111111111010101110101010001000000,
             0b0110111110111010110001001000000000,
             0b1111111000011111110010110000000000]


def complex_hash(addr):
    r = 0
    for f in reversed(functions_i9_9900):
        r <<= 1
        r |= (popcount(f & addr) & 1)
    return r


def convert64(x):
    return np.int64(int(x, base=16))

def convert8(x):
    return np.int8(int(x, base=16))

df = pd.read_csv(sys.argv[1] + "-results_lite.csv.bz2",
        dtype={
            "main_core": np.int8,
            "helper_core": np.int8,
            # "address": int,
            # "hash": np.int8,
            "time": np.int16,
            "clflush_remote_hit": np.int32,
            "clflush_shared_hit": np.int32,
            "clflush_miss_f": np.int32,
            "clflush_local_hit_f": np.int32,
            "clflush_miss_n": np.int32,
            "clflush_local_hit_n": np.int32,
            "reload_miss": np.int32,
            "reload_remote_hit": np.int32,
            "reload_shared_hit": np.int32,
            "reload_local_hit": np.int32},
        converters={'address': convert64, 'hash': convert8},
        )

sample_columns = [
"clflush_remote_hit",
"clflush_shared_hit",
"clflush_miss_f",
"clflush_local_hit_f",
"clflush_miss_n",
"clflush_local_hit_n",
"reload_miss",
"reload_remote_hit",
"reload_shared_hit",
"reload_local_hit",
]

sample_flush_columns = [
    "clflush_remote_hit",
    "clflush_shared_hit",
    "clflush_miss_f",
    "clflush_local_hit_f",
    "clflush_miss_n",
    "clflush_local_hit_n",
]


slice_mapping = pd.read_csv(sys.argv[1] + ".slices.csv")
core_mapping = pd.read_csv(sys.argv[1] + ".cores.csv")

def remap_core(key):
    def remap(core):
        remapped = core_mapping.iloc[core]
        return remapped[key]

    return remap


df["main_socket"] = df["main_core"].apply(remap_core("socket"))
df["main_core_fixed"] = df["main_core"].apply(remap_core("core"))
df["main_ht"] = df["main_core"].apply(remap_core("hthread"))
df["helper_socket"] = df["helper_core"].apply(remap_core("socket"))
df["helper_core_fixed"] = df["helper_core"].apply(remap_core("core"))
df["helper_ht"] = df["helper_core"].apply(remap_core("hthread"))

# slice_mapping = {3: 0, 1: 1, 2: 2, 0: 3}

slice_remap = lambda h: slice_mapping["slice_group"].iloc[h]
df["slice_group"] = df["hash"].apply(slice_remap)


print(df.columns)
#df["Hash"] = df["Addr"].apply(lambda x: (x >> 15)&0x3)

addresses = df["address"].unique()
print(addresses)
print(*[bin(a) for a in addresses], sep='\n')

print(df.head())

print(df["hash"].unique())

min_time = df["time"].min()
max_time = df["time"].max()

q10s = [wq.quantile(df["time"], df[col], 0.1) for col in sample_flush_columns]
q90s = [wq.quantile(df["time"], df[col], 0.9) for col in sample_flush_columns]

graph_upper = int(((max(q90s) + 19) // 10) * 10)
graph_lower = int(((min(q10s) - 10) // 10) * 10)
# graph_lower = (min_time // 10) * 10
# graph_upper = ((max_time + 9) // 10) * 10

print("graphing between {}, {}".format(graph_lower, graph_upper))

df_main_core_0 = df[df["main_core"] == 0]
#df_helper_core_0 = df[df["helper_core"] == 0]

colours = ["b", "r", "g", "y"]

def custom_hist(x, *y, **kwargs):
    for (i, yi) in enumerate(y):
        kwargs["color"] = colours[i]
        sns.distplot(x, range(graph_lower, graph_upper), hist_kws={"weights": yi, "histtype":"step"}, kde=False, **kwargs)


custom_hist(df["time"], df["clflush_miss_n"], df["clflush_remote_hit"])

tikzplotlib.save("fig-hist-all.tex")#, axis_width=r'0.175\textwidth', axis_height=r'0.25\textwidth')
plt.show()

attacker = 2
victim = 7
slice = 14

df_ax_vx_sx = df[(df["hash"] == slice) & (df["main_core"] == attacker) & (df["helper_core"] == victim)]

custom_hist(df_ax_vx_sx["time"], df_ax_vx_sx["clflush_miss_n"], df_ax_vx_sx["clflush_remote_hit"])
tikzplotlib.save("fig-hist-good-A{}V{}S{}.tex".format(attacker,victim,slice))#, axis_width=r'0.175\textwidth', axis_height=r'0.25\textwidth')
plt.show()

attacker = 9
victim = 4
slice = 8

df_ax_vx_sx = df[(df["hash"] == slice) & (df["main_core"] == attacker) & (df["helper_core"] == victim)]

custom_hist(df_ax_vx_sx["time"], df_ax_vx_sx["clflush_miss_n"], df_ax_vx_sx["clflush_remote_hit"])
tikzplotlib.save("fig-hist-bad-A{}V{}S{}.tex".format(attacker,victim,slice))#, axis_width=r'0.175\textwidth', axis_height=r'0.25\textwidth')
plt.show()


g = sns.FacetGrid(df_main_core_0, col="helper_core", row="hash", legend_out=True)
g2 = sns.FacetGrid(df, col="main_core", row="hash", legend_out=True)


# Color convention here :
# Blue = miss
# Red = Remote Hit
# Green = Local Hit
# Yellow = Shared Hit

g.map(custom_hist, "time", "clflush_miss_n", "clflush_remote_hit", "clflush_local_hit_n", "clflush_shared_hit")

g2.map(custom_hist, "time", "clflush_miss_n", "clflush_remote_hit", "clflush_local_hit_n", "clflush_shared_hit")

# g.map(sns.distplot, "time", hist_kws={"weights": df["clflush_hit"]}, kde=False)

#plt.show()
#plt.figure()

#df_mcf6 = df[df["main_core_fixed"] == 6]
#df_mcf6_slg7 = df_mcf6[df_mcf6["slice_group"] == 7]
#g3 = sns.FacetGrid(df_mcf6_slg7, row="helper_core_fixed", col="main_ht")
#g3.map(custom_hist, "time", "clflush_miss_n", "clflush_remote_hit", "clflush_local_hit_n", "clflush_shared_hit")

#g4 = sns.FacetGrid(df_mcf6_slg7, row="helper_core_fixed", col="helper_ht")
g#4.map(custom_hist, "time", "clflush_miss_n", "clflush_remote_hit", "clflush_local_hit_n", "clflush_shared_hit")

def stat(x, key):
    return wq.median(x["time"], x[key])


miss = df.groupby(["main_core", "helper_core", "hash"]).apply(stat, "clflush_miss_n")
hit_remote = df.groupby(["main_core", "helper_core", "hash"]).apply(stat, "clflush_remote_hit")
hit_local = df.groupby(["main_core", "helper_core", "hash"]).apply(stat, "clflush_local_hit_n")
hit_shared = df.groupby(["main_core", "helper_core", "hash"]).apply(stat, "clflush_shared_hit")

stats = miss.reset_index()
stats.columns = ["main_core", "helper_core", "hash", "clflush_miss_n"]
stats["clflush_remote_hit"] = hit_remote.values
stats["clflush_local_hit_n"] = hit_local.values
stats["clflush_shared_hit"] = hit_shared.values

stats.to_csv(sys.argv[1] + ".stats.csv", index=False)

#print(stats.to_string())

plt.show()
exit(0)
g = sns.FacetGrid(stats, row="Core")

g.map(sns.distplot, 'Miss', bins=range(100, 480), color="r")
g.map(sns.distplot, 'Hit', bins=range(100, 480))
plt.show()

#stats["clflush_miss_med"] = stats[[0]].apply(lambda x: x["miss_med"])
#stats["clflush_hit_med"] = stats[[0]].apply(lambda x: x["hit_med"])
#del df[[0]]
#print(hit.to_string(), miss.to_string())

# test = pd.DataFrame({"value" : [0, 5], "weight": [5, 1]})
# plt.figure()
# sns.distplot(test["value"], hist_kws={"weights": test["weight"]}, kde=False)

exit(0)
