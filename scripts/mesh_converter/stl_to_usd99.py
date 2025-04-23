#!/usr/bin/env python3
# STL to USD Converter Script

import os
import argparse
from isaaclab.app import AppLauncher
import asyncio

# Add argparse arguments
parser = argparse.ArgumentParser(description="Convert STL files to USD format")
parser.add_argument("--stl-file", help="Path to the STL file to convert")
parser.add_argument("--output-dir", help="Directory to save the USD file (default: same as STL file)")
parser.add_argument("--output-name", help="The USD file name")
parser.add_argument("--no-instanceable", action="store_false", dest="make_instanceable", help="Don't make the USD asset instanceable")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters.mesh_converter_cfg import MeshConverterCfg
from isaaclab.sim.converters.mesh_converter import MeshConverter
from isaaclab.sim.schemas import schemas_cfg
# Create a global event loop for the entire application
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

def convert_stl_to_usd():
    try:
        # Ensure the input file (STL) and output file (USD) are provided
        output_file_path = args.output_dir if args.output_dir else os.path.dirname(args.stl_file)
        output_file_name = args.output_name
        
        # Validate input path
        if not os.path.exists(args.stl_file):
            raise FileNotFoundError(f"STL file not found: {args.stl_file}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_file_path, exist_ok=True)
        
        # Configure the converter
        converter_cfg = MeshConverterCfg(
            asset_path=args.stl_file,
            usd_dir=output_file_path,
            usd_file_name=output_file_name,
            make_instanceable=True,
            scale=(1.0, 1.0, 1.0),
            translation=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            collision_approximation="convexDecomposition",
            mass_props=schemas_cfg.MassPropertiesCfg(mass=100.0),
            rigid_props=schemas_cfg.RigidBodyPropertiesCfg(rigid_body_enabled=True,
                                                           disable_gravity=True,
                                                           enable_gyroscopic_forces=True)
        )
        
        print("converter_cfg loaded")
        
        # Create the converter
        converter = MeshConverter(cfg=converter_cfg)
        
        # Run the async method in the loop
        loop.run_until_complete(converter._convert_mesh_to_usd(args.stl_file, output_file_name))
        
        print(f"Successfully converted STL to USD at: {output_file_path}/{output_file_name}")
        
    except Exception as e:
        print(f"Error converting STL to USD: {e}")
    finally:
        # Close the simulation app after the process is done
        simulation_app.close()

if __name__ == "__main__":
    print("_Start 12345")
    try:
        print("Main_Start")
        convert_stl_to_usd()
        print("Main_Stop")
    finally:
        loop.close()
    print("_End 12345")