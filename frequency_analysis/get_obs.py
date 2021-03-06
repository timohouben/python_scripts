#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 11:24:15 2018

@author: houben
"""

import os
import re


def get_obs(path_to_project):
    """
    Give the path to the ogs project and get a list with the names of the
    observation points, the file names and the number included in the name from
    .tec-files!
    This number can be used as a distance or coordinate of a transect.
    This function ONLY WORKS FOR POLYLINES currently!
    """

    file_names = [f for f in os.listdir(path_to_project) if f.endswith(".tec")]
    file_names.sort()

    string_begin = "ply_"
    string_end = "_t"
    obs_names = []
    obs_locs = []
    for file_name in file_names:
        obs_name_temp = file_name[
            file_name.find(string_begin) + 4 : file_name.rfind(string_end)
        ]
        obs_names.append(obs_name_temp)
        obs_locs.append(int(re.search(r"\d+", obs_name_temp).group()))
    return file_names, obs_names, obs_locs


if __name__ == "__main__":
    """
    main function to test
    """
    file_names_c = [
        "transect_01_ply_obs_0000_t1_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0010_t2_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0100_t3_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0200_t4_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0300_t5_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0400_t6_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0500_t7_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0600_t8_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0700_t9_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0800_t10_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0900_t11_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0950_t12_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0960_t13_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0970_t14_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0980_t15_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_0990_t16_GROUNDWATER_FLOW.tec",
        "transect_01_ply_obs_1000_t17_GROUNDWATER_FLOW.tec",
        "transect_01_ply_tobs_0010_t2_GROUNDWATER_FLOW.tec",
    ]

    obs_names_c = [
        "obs_0000",
        "obs_0010",
        "obs_0100",
        "obs_0200",
        "obs_0300",
        "obs_0400",
        "obs_0500",
        "obs_0600",
        "obs_0700",
        "obs_0800",
        "obs_0900",
        "obs_0950",
        "obs_0960",
        "obs_0970",
        "obs_0980",
        "obs_0990",
        "obs_1000",
        "tobs_0010",
    ]

    obs_locs_c = [
        0,
        10,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        950,
        960,
        970,
        980,
        990,
        1000,
        10,
    ]

    path_to_project = "/Users/houben/PhD/python/scripts/tests/get_obs"

    file_names, obs_names, obs_locs = get_obs(path_to_project)
    print(file_names)
    print(file_names_c)
    print(obs_names)
    print(obs_names_c)
    print(obs_locs)
    print(obs_locs_c)
    if (
        file_names == file_names_c
        and obs_names == obs_names_c
        and obs_locs == obs_locs_c
    ):
        print("Test passed.")
    else:
        print("Test NOT passed.")
