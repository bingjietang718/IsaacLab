import argparse
import re
import subprocess

def update_assembly_id(file_path, new_id):
    # Read the file lines.
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    # This regex captures the part before the value and any current value.
    pattern = re.compile(r'^(.*assembly_id\s*=\s*).*$')
    
    for line in lines:
        if 'assembly_id =' in line:
            # Replace the entire line with the new id value.
            line = pattern.sub(r"\1'{}'".format(new_id), line)
        updated_lines.append(line)
    
    # Write the modified lines back to the file.
    with open(file_path, 'w') as f:
        f.writelines(updated_lines)
    print(f"Updated assembly_id to {new_id} in {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Update assembly_id and run training script.")
    parser.add_argument("--cfg_path", type=str, help="Path to the file containing assembly_id.", default="source/isaaclab_tasks/isaaclab_tasks/direct/assembly/assembly_tasks_cfg.py")
    parser.add_argument("--assembly_id", type=str, help="New assembly ID to set.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path.")
    parser.add_argument("--num_envs", type=int, default=32, help="Number of parallel environment.")
    parser.add_argument("--train", action='store_true', help="Run training mode.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    args = parser.parse_args()
    
    update_assembly_id(args.cfg_path, args.assembly_id)

    bash_command = None
    if args.train:
        bash_command = "./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Assembly-Direct-v0"
    else:
        if not args.checkpoint: 
            raise ValueError('No checkpoint provided for evaluation.')
        bash_command = "./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task=Assembly-Direct-v0"
    
    bash_command += f" --num_envs={str(args.num_envs)}"

    if args.checkpoint:
        bash_command += f" --checkpoint={args.checkpoint}"

    if args.headless:
        bash_command += " --headless"

    # Run the bash command
    subprocess.run(bash_command, shell=True, check=True)

if __name__ == "__main__":
    main()
