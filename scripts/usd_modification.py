from pxr import Usd, UsdPhysics

usd_path = "path/to/husky.usd"

# Open the USD stage
stage = Usd.Stage.Open(usd_path)

# Get the root prim
root_prim = stage.GetDefaultPrim()
if not root_prim:
    print("Error: No default prim found.")
else:
    # Apply ArticulationRootAPI if missing
    if not UsdPhysics.ArticulationRootAPI.CanApply(root_prim):
        print("Adding ArticulationRootAPI to", root_prim.GetPath())
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)

    # Save the modified USD file
    stage.GetRootLayer().Save()
    print("Saved updated USD file.")
