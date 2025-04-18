import isaaclab.terrains as terrain_gen
from ..terrain_generator_cfg import TerrainGeneratorCfg

CUSTOM_MESH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.0,
    slope_threshold=0.0,
    use_cache=False,
    sub_terrains={
        "custom": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=8.0,
            platform_width=8.0,
            mesh_files=["isaac_assets/terrains/custom/my_terrain.dae"],
            border_width=0.0,
        ),
    },
)
