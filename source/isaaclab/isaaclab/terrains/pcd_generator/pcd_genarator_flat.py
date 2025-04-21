import numpy as np
import open3d as o3d


#from your_project_path.terrain_configs.flat import FLAT_TERRAIN_CFG  # adjust path
from isaaclab.terrains.config.flat import FLAT_TERRAIN_CFG
import isaaclab.terrains as terrain_gen

# Step 1: Create terrain
terrain_generator = FLAT_TERRAIN_CFG.class_type(FLAT_TERRAIN_CFG)
terrain_generator.generate()

# Step 2: Get heightmap
heightfield = terrain_generator.height_field
scale_x = FLAT_TERRAIN_CFG.horizontal_scale
scale_y = FLAT_TERRAIN_CFG.horizontal_scale
scale_z = FLAT_TERRAIN_CFG.vertical_scale

rows, cols = heightfield.shape
x_coords = np.arange(cols) * scale_x
y_coords = np.arange(rows) * scale_y
x_grid, y_grid = np.meshgrid(x_coords, y_coords)

# Flatten to (N, 3)
points = np.stack([x_grid.ravel(), y_grid.ravel(), heightfield.ravel() * scale_z], axis=1)

# Step 3: Convert to Open3D point cloud and save
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
o3d.io.write_point_cloud("flat_terrain.pcd", pcd)

print("✅ PCD file saved as 'flat_terrain.pcd'")
