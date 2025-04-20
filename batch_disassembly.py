import os

asset_root_path = '/home/yijieg/Projects/NGC_materials/isaacgymenvs_automate/assets/automate/mesh/'
asset_ids = sorted(os.listdir(asset_root_path))[:100]

for task in ['00014', '00117', '00213', '00726']:
    #os.system('mkdir /home/yijieg/Downloads/assembly_asset/%s'%task)
    command = './isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/assembly/run_disassembly_w_id.py --assembly_id=%s --headless'%task
    os.system(command)
    command = 'cp /home/yijieg/Projects/IsaacLab/disassemble_data/%s_disassemble_traj.json ~/Projects/SRSA_data_isaaclab/disassembly_paths_new/asset_%s_disassembly_traj.json'%(task, task)
    os.system(command)
