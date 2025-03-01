import os
import shutil

# Define source and destination directories
source_base = "/home/bingjie/repos/shared/isaacgymenvs/assets/automate/mesh"
destination_base = "/home/bingjie/Downloads/assembly_asset"

# List of assembly IDs
assembly_ids = [
    '01053',
            '00470', 
            '00499', 
            '00652', 
            '00346', 
            '00004',
            '00681', 
            '00783', 
            '00768', 
            '00192', 
            '01026', 
            '00007', 
            '00308', 
            '00437', 
            '00446', 
            '00016', 
            '00133', 
            '00855', 
            '00471', 
            '00030', 
            '00186', 
            '01132', 
            '00863', 
            '00426', 
            '00077', 
            '00703', 
            '00028', 
            '00110', 
            '00015', 
            '00329', 
            '01125', 
            '00103', 
            '00444', 
            '00014', 
            '00615', 
            '00078', 
            '00187', 
            '01029', 
            '00021', 
            '01041', 
            '00755', 
            '01102', 
            '00614', 
            '00597', 
            '00686', 
            '01129', 
            '00074', 
            '00293', 
            '00649', 
            '00638', 
            '00143', 
            '00345', 
            '00537', 
            '00648', 
            '00388', 
            '00163', 
            '00117', 
            '01079', 
            '00340', 
            '00506', 
            '00741', 
            '00271', 
            '00301', 
            '00062', 
            '00210', 
            '00731', 
            '00042', 
            '00559', 
            '00514', 
            '00726', 
            '00553', 
            '01136', 
            '00083', 
            '00319', 
            '00417', 
            '00175', 
            '00700', 
            '00360', 
            '00318', 
            '00320', 
            '00581', 
            '01092', 
            '00410', 
            '00213', 
            '00659', 
            '00831', 
            '00422', 
            '00296', 
            '00138', 
            '01036', 
            '00486', 
            '00860', 
            '00211', 
            '00480', 
            '00081', 
            '00256', 
            '00190', 
            '00032', 
            '00141', 
            '00255'
]  # Add more IDs as needed

def copy_and_rename_files(assembly_id):
    # Define source and destination paths
    source_mesh_dir = f"/home/bingjie/repos/shared/isaacgymenvs/assets/automate/mesh/{assembly_id}/"
    dest_dir = f"/home/bingjie/Downloads/assembly_asset/{assembly_id}/"
    
    source_traj_file = f"/home/bingjie/repos/shared/isaacgymenvs/isaacgymenvs/tasks/automate/data/asset_{assembly_id}_disassemble_traj.json"
    dest_traj_file = f"{dest_dir}/disassemble_traj.json"
    
    # Ensure destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Copy and rename mesh files
    for src_file, dest_file in [("asset_plug.obj", "plug.obj"), ("asset_socket.obj", "socket.obj")]:
        src_path = os.path.join(source_mesh_dir, src_file)
        dest_path = os.path.join(dest_dir, dest_file)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")
        else:
            print(f"Warning: {src_path} does not exist")
    
    # Copy disassembly trajectory file
    if os.path.exists(source_traj_file):
        shutil.copy2(source_traj_file, dest_traj_file)
        print(f"Copied {source_traj_file} to {dest_traj_file}")
    else:
        print(f"Warning: {source_traj_file} does not exist")

for assembly_id in assembly_ids:
    copy_and_rename_files(assembly_id)

print("Done!")
