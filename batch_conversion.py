import os

asset_root_path = '/home/yijieg/Projects/NGC_materials/isaacgymenvs_automate/assets/automate/mesh/'
asset_ids = sorted(os.listdir(asset_root_path))[:100]

#for task in ['00681', '00768', '00320', '01129', '00015', '01036', '00340', '00296', '01041']: #asset_ids[70:80]:
for task in ['00117']: #asset_ids[]:
    if task in ['00681', '00768', '00320', '01129', '00015', '01036', '00340', '00296', '01041']:
        continue
    #os.system('mkdir /home/yijieg/Downloads/assembly_asset/%s'%task)
    command = './isaaclab.sh -p scripts/tools/convert_urdf.py /home/yijieg/Projects/SERL_osmo/isaacgymenvs/assets/automate/urdf/%s_socket.urdf /home/yijieg/Downloads/assembly_asset/%s/socket.usd --fix-base'%(task, task)
    os.system(command)
    command = './isaaclab.sh -p scripts/tools/convert_urdf.py /home/yijieg/Projects/SERL_osmo/isaacgymenvs/assets/automate/urdf/%s_plug.urdf /home/yijieg/Downloads/assembly_asset/%s/plug.usd'%(task, task)
    os.system(command)
    #command = 'cp /home/yijieg/Projects/SERL_osmo/isaacgymenvs/isaacgymenvs/tasks/automate/data/asset_%s_disassemble_traj.json ~/Downloads/assembly_asset/%s/disassemble_traj.json'%(task, task)
    #os.system(command)
    #command = 'cp /home/yijieg/Projects/SERL_osmo/isaacgymenvs/assets/automate/mesh/%s/asset_socket.obj /home/yijieg/Downloads/assembly_asset/%s/socket.obj'%(task, task)
    #os.system(command)
    #command = 'cp /home/yijieg/Projects/SERL_osmo/isaacgymenvs/assets/automate/mesh/%s/asset_plug.obj /home/yijieg/Downloads/assembly_asset/%s/plug.obj'%(task, task)
    #os.system(command)
