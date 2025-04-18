# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for flat terrain."""

import isaaclab.terrains as terrain_gen

from ..terrain_generator_cfg import TerrainGeneratorCfg

FLAT_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.001,
    slope_threshold=0.0,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1,
            border_width=0.1,
            noise_range=(0.0, 0.0),
            noise_step=0.02,
        ),
    },
)
"""Flat terrain configuration."""
