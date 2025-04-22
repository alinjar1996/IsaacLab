#!/usr/bin/env python3
# STL to USD Converter Script

import os
# import isaaclab.sim.converters.mesh_converter_cfg
# import isaaclab.sim.converters.mesh_converter
# from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
# from isaaclab.sim.converters.mesh_converter import MeshConverter

import argparse
from isaaclab.app import AppLauncher
import asyncio



# add argparse arguments
parser = argparse.ArgumentParser(description="Convert STL files to USD format")
parser.add_argument("--stl_file", help="Path to the STL file to convert")
parser.add_argument("--output-dir", help="Directory to save the USD file (default: same as STL file)")
parser.add_argument("--output-name", help="the USD file name")
parser.add_argument("--no-instanceable", action="store_false", dest="make_instanceable",
                    help="Don't make the USD asset instanceable")
args = parser.parse_args()
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments

# launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.converters.mesh_converter import MeshConverter




async def convert_stl_to_usd(stl_file_path, output_name, output_dir=None, make_instanceable=True):
    """
    Convert an STL file to USD format using MeshConverter.
    
    Args:
        stl_file_path (str): Path to the STL file to convert
        output_dir (str, optional): Directory to save the USD file in. If None, uses the same directory as the STL file.
        make_instanceable (bool): Whether to make the resulting USD instanceable. Default is True.
        
    Returns:
        str: Path to the generated USD file
    """
    # Validate input path
    if not os.path.exists(stl_file_path):
        raise FileNotFoundError(f"STL file not found: {stl_file_path}")
    
    # If no output directory specified, use the same directory as the input file
    if output_dir is None:
        output_dir = os.path.dirname(stl_file_path)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure the converter
    converter_cfg = MeshConverterCfg(
        # Path to the STL file
        asset_path=stl_file_path,
        # Output directory for the USD file
        usd_dir=output_dir,
        usd_file_name=output_name,
        # Make the USD asset instanceable
        make_instanceable=make_instanceable,
        # Scaling - default is (1.0, 1.0, 1.0)
        scale=(1.0, 1.0, 1.0),
        # Translation if needed - default is (0.0, 0.0, 0.0)
        translation=(0.0, 0.0, 0.0),
        # Rotation as quaternion (w, x, y, z) - default is (1.0, 0.0, 0.0, 0.0) (no rotation)
        rotation=(1.0, 0.0, 0.0, 0.0),
        # Set collision approximation (can be "convexHull", "convexDecomposition", or "meshSimplification")
        collision_approximation="convexHull"
    )
    
    # Create and run the converter
    converter = MeshConverter(cfg=converter_cfg)
    await converter._convert_mesh_to_usd(stl_file_path,output_name)
    
    # Return the path to the generated USD file
    return converter.usd_path

if __name__ == "__main__":
    try:
        usd_path = asyncio.run(convert_stl_to_usd(
            args.stl_file, 
            args.output_name,
            args.output_dir, 
            args.make_instanceable
        ))
        print(f"Successfully converted STL to USD: {usd_path}")
    except Exception as e:
        print(f"Error converting STL to USD: {e}")

