import argparse
import re
import subprocess

def update_task_param(task_cfg, asset_dir, assembly_id):
    # Read the file lines.
    with open(task_cfg, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    
    # Regex patterns to capture the assignment lines
    asset_dir_pattern = re.compile(r'^(.*ASSET_DIR\s*=\s*).*$')
    assembly_pattern = re.compile(r'^(.*assembly_id\s*=\s*).*$')
    
    for line in lines:
        if 'ASSET_DIR = ' in line:
            line = asset_dir_pattern.sub(r"\1'{}'".format(asset_dir), line)
        elif 'assembly_id =' in line:
            line = assembly_pattern.sub(r"\1'{}'".format(assembly_id), line)
        
        updated_lines.append(line)
    
    # Write the modified lines back to the file.
    with open(task_cfg, 'w') as f:
        f.writelines(updated_lines)

def main():
    parser = argparse.ArgumentParser(description="Update assembly_id and run training script.")
    #parser.add_argument("--asset_dir", type=str, help="Path to the directory containing asset data.", default="/home/bingjie/Downloads/assembly_asset")
    parser.add_argument("--asset_dir", type=str, help="Path to the directory containing asset data.", default="/home/yijieg/Downloads/assembly_asset")
    parser.add_argument("--cfg_path", type=str, help="Path to the file containing assembly_id.", default="source/isaaclab_tasks/isaaclab_tasks/direct/assembly/disassembly_tasks_cfg.py")
    parser.add_argument("--assembly_id", type=str, default='00731', help="New assembly ID to set.")
    parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environment.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    args = parser.parse_args()
        
    update_task_param(
        args.cfg_path, 
        args.asset_dir,
        args.assembly_id, 
        )

    bash_command = "./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Disassembly-Act-v0"
    
    bash_command += f" --num_envs={str(args.num_envs)}"
    bash_command += f" --seed={str(args.seed)}"

    if args.headless:
        bash_command += " --headless"

    # Run the bash command
    subprocess.run(bash_command, shell=True, check=True)

if __name__ == "__main__":
    main()
