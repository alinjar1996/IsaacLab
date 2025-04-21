import isaaclab.terrains as terrain_gen
from ..terrain_generator_cfg import TerrainGeneratorCfg



SLOPE_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.001,
    slope_threshold=0.5,
    use_cache=False,
    sub_terrains={
        "slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0,
            border_width=0.0,
            slope_range=(0.5, 0.5),         # Controls steepness (tan(θ))
            size=(10,10)
        ),
    },
)
"""Entire terrain is a slope."""
