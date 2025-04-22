#!/usr/bin/env python3
# STL to USD Converter Script

import os
import argparse
from isaaclab.app import AppLauncher
import asyncio





#def parse_arguments():
parser = argparse.ArgumentParser(description="Convert STL files to USD format")

# Core STL conversion arguments
parser.add_argument("--stl_file", type=str, required=True, help="Path to the STL file to convert")
parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the USD file (default: same as STL file)")
parser.add_argument("--output-name", type=str, required=True, help="The USD file name")
parser.add_argument("--no-instanceable", action="store_false", dest="make_instanceable",
                    help="Don't make the USD asset instanceable")

# Optional transformation & collision args
parser.add_argument('--collision_approximation', type=str, default="convexDecomposition",
                    choices=["convexDecomposition", "convexHull", "boundingCube",
                                "boundingSphere", "meshSimplification", "none"],
                    help="Collision approximation method")

parser.add_argument('--translation', type=float, nargs=3, default=(0.0, 0.0, 0.0),
                    help="Translation (x y z)")
parser.add_argument('--rotation', type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0),
                    help="Rotation quaternion (w x y z)")
parser.add_argument('--scale', type=float, nargs=3, default=(1.0, 1.0, 1.0),
                    help="Scale (x y z)")

# Isaac Lab app args
args = parser.parse_args()
AppLauncher.add_app_launcher_args(parser)


# Launch Isaac Lab app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.converters.mesh_converter import MeshConverter
from isaaclab.sim.schemas import schemas_cfg

async def convert_stl_to_usd(stl_file_path, output_name, output_dir=None, make_instanceable=True,
                             collision_approximation="convexDecomposition",
                             translation=(0.0, 0.0, 0.0),
                             rotation=(1.0, 0.0, 0.0, 0.0),
                             scale=(1.0, 1.0, 1.0)):
    """
    Convert an STL file to USD format using MeshConverter.
    """
    # Validate STL path
    if not os.path.exists(stl_file_path):
        raise FileNotFoundError(f"STL file not found: {stl_file_path}")

    # Resolve output directory
    output_dir = output_dir or os.path.dirname(stl_file_path)
    os.makedirs(output_dir, exist_ok=True)

    # Default physical properties
    default_mass_props = schemas_cfg.MassPropertiesCfg(mass=1.0)
    default_rigid_props = schemas_cfg.RigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        max_depenetration_velocity=1.0,
        enable_gyroscopic_forces=True
    )

    # Mesh converter configuration
    converter_cfg = MeshConverterCfg(
        asset_path=stl_file_path,
        usd_file_name=output_name,
        usd_dir=output_dir,
        make_instanceable=make_instanceable,
        mass_props=default_mass_props,
        rigid_props=default_rigid_props,
        collision_approximation=collision_approximation,
        translation=translation,
        rotation=rotation,
        scale=scale,
    )

    # Convert mesh
    converter = MeshConverter(cfg=converter_cfg)
    await converter._convert_mesh_to_usd(stl_file_path, output_name)
    return converter.usd_path


if __name__ == "__main__":

    try:
        usd_path = asyncio.run(convert_stl_to_usd(
            stl_file_path=args.stl_file,
            output_name=args.output_name,
            output_dir=args.output_dir,
            make_instanceable=args.make_instanceable,
            collision_approximation=args.collision_approximation,
            translation=tuple(args.translation),
            rotation=tuple(args.rotation),
            scale=tuple(args.scale),
        ))
        print(f"\n✅ Successfully converted STL to USD: {usd_path}")
    except Exception as e:
        print(f"\n❌ Error converting STL to USD: {e}")

# Schedule the async main() using the Isaac app event loop
