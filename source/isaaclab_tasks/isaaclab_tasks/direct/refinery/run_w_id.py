import argparse
import re
import subprocess
import sys
import os

def update_task_param(task_cfg, asset_dir, assembly_id, step_id, if_sbc, if_log_eval, sample, ac_func):
    # Read the file lines.
    with open(task_cfg, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    
    # Regex patterns to capture the assignment lines
    asset_dir_pattern = re.compile(r'^(.*ASSET_DIR\s*=\s*).*$')
    assembly_pattern = re.compile(r'^(.*assembly_id\s*=\s*).*$')
    step_pattern = re.compile(r'^(.*step_id\s*=\s*).*$')
    if_sbc_pattern = re.compile(r'^(.*if_sbc\s*:\s*bool\s*=\s*).*$')
    if_log_eval_pattern = re.compile(r'^(.*if_logging_eval\s*:\s*bool\s*=\s*).*$')
    eval_file_pattern = re.compile(r'^(.*eval_filename\s*:\s*str\s*=\s*).*$')
    sample_pattern = re.compile(r'^(.*sample_from\s*:\s*str\s*=\s*).*$')
    ac_func_pattern = re.compile(r'^(.*acquisition_function\s*:\s*str\s*=\s*).*$')
    
    for line in lines:
        if 'ASSET_DIR = ' in line:
            line = asset_dir_pattern.sub(r"\1'{}'".format(asset_dir), line)
        elif 'assembly_id =' in line:
            line = assembly_pattern.sub(r"\1'{}'".format(assembly_id), line)
        elif 'step_id =' in line:
            line = step_pattern.sub(r"\1'{}'".format(step_id), line)
        elif 'if_sbc: bool =' in line:
            line = if_sbc_pattern.sub(r"\1{}".format(str(if_sbc)), line)
        elif 'if_logging_eval: bool =' in line:
            line = if_log_eval_pattern.sub(r"\1{}".format(str(if_log_eval)), line)
        elif 'eval_filename: str = ' in line:
            line = eval_file_pattern.sub(r"\1'{}'".format(f"evaluation_{assembly_id}_{step_id}.h5"), line)
        elif 'sample_from: str = ' in line:
            line = sample_pattern.sub(r"\1'{}'".format(f"{sample}"), line)
        elif 'acquisition_function: str = ' in line:
            line = ac_func_pattern.sub(r"\1'{}'".format(f"{ac_func}"), line)
        
        updated_lines.append(line)
    
    # Write the modified lines back to the file.
    with open(task_cfg, 'w') as f:
        f.writelines(updated_lines)

def generate_task_cfg(template_file, output_file, assembly_id, part_ids, step_id):

    # output_file = "refinery_task_cfg_generated.py"

    with open(template_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if (not add_asset_configclass(line, outfile, assembly_id, part_ids)) \
                  and (not add_asset_in_task_class(line, outfile, part_ids)) \
                    and (not add_asset_articulationcfg(line, outfile, part_ids)):
                outfile.write(line)

    replace_held_asset_chunk(output_file, assembly_id, step_id)
    replace_fixed_asset_chunk(output_file, assembly_id)

def add_asset_configclass(line, outfile, assembly_id, part_ids):

    start_marker = "## start: add assembled asset config classes"
    end_marker   = "## end: add assembled asset config classes"

    if (start_marker not in line) and (end_marker not in line):
        return False
    
    lines_to_insert = generate_assets_config(assembly_id, part_ids) 

    if start_marker in line:
        outfile.write(line)
        for insert_line in lines_to_insert:
            outfile.write(insert_line)

        outfile.write(end_marker+"\n")

    return True

def generate_assets_config(assembly_id, part_ids):
    """
    Given a list of parts for assembly_id,
    return a list of lines containing the @configclass definitions.
    """
    lines = []
    for part_id in part_ids:
        lines.append("@configclass\n")
        lines.append(f"class AssembledAsset{part_id}(FixedAssetCfg):\n")
        lines.append(f"    usd_path = '{assembly_id}_{part_id}.usd'\n")
        lines.append(f"    obj_path = '{assembly_id}_{part_id}.obj'\n")
        lines.append(f"    diameter = 0.007986\n")
        lines.append(f"    height = 0.050\n")
        lines.append(f"    mass = 0.019\n")
        lines.append("\n")  # Blank line after each class
    return lines

def replace_held_asset_chunk(outfile, assembly_id, step_id):

    # outfile = "refinery_task_cfg_generated.py"

    # Exact text to find (must match your file's indentation and comments):
    old_chunk = (
        "@configclass\n"
        "class HeldAsset(HeldAssetCfg):\n"
        "    usd_path = # held asset usd\n"
        "    obj_path = # held asset obj\n"
        "    diameter = 0.007986\n"
        "    height = 0.050\n"
        "    mass = 0.019"
    )

    # Exact text to replace it with:
    new_chunk = (
        "@configclass\n"
        "class HeldAsset(HeldAssetCfg):\n"
        f"    usd_path = '{assembly_id}_{step_id}.usd'\n"
        f"    obj_path = '{assembly_id}_{step_id}.obj'\n"
        "    diameter = 0.007986\n"
        "    height = 0.050\n"
        "    mass = 0.019"
    )

    # Read in the original file
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the old text with the new text
    new_content = content.replace(old_chunk, new_chunk)

    # Write the updated content back to the file
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(new_content)

def replace_fixed_asset_chunk(outfile, assembly_id):

    # outfile = "refinery_task_cfg_generated.py"

    # Exact text to find (must match your file's indentation and comments):
    old_chunk = (
        "@configclass\n"
        "class FixedAsset(FixedAssetCfg):\n"
        "    usd_path = # fixed asset usd\n"
        "    obj_path = # fixed asset obj\n"
        "    diameter = 0.0081\n"
        "    height = 0.050896\n"
        "    base_height = 0.0"
    )

    # Exact text to replace it with:
    new_chunk = (
        "@configclass\n"
        "class FixedAsset(FixedAssetCfg):\n"
        f"    usd_path = '{assembly_id}_0.usd'\n"
        f"    obj_path = '{assembly_id}_0.obj'\n"
        "    diameter = 0.0081\n"
        "    height = 0.050896\n"
        "    base_height = 0.0"
    )

    # Read in the original file
    with open(outfile, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace the old text with the new text
    new_content = content.replace(old_chunk, new_chunk)

    # Write the updated content back to the file
    with open(outfile, "w", encoding="utf-8") as f:
        f.write(new_content)

def add_asset_in_task_class(line, outfile, part_ids):

    start_marker = "    ## start: add assembled asset in task classes"
    end_marker   = "    ## end: add assembled asset in task classes"

    if (start_marker not in line) and (end_marker not in line):
        return False

    if start_marker in line:
        outfile.write(line)
        for part_id in part_ids:
            outfile.write(f"    assembled_asset_{part_id}_cfg = AssembledAsset{part_id}()\n")

        outfile.write(end_marker+"\n")

    return True

def add_asset_articulationcfg(line, outfile, part_ids):

    start_marker = "    ## start: add assembled asset articulationcfg"
    end_marker   = "    ## end: add assembled asset articulationcfg"

    if (start_marker not in line) and (end_marker not in line):
        return False
    
    lines_to_insert = generate_asset_articulationcfg(part_ids) 

    if start_marker in line:
        outfile.write(line)
        for insert_line in lines_to_insert:
            outfile.write(insert_line)

        outfile.write(end_marker+"\n")

    return True

def generate_asset_articulationcfg(part_ids):

    lines = []
    for part_id in part_ids:

        lines.append(f"    assembled_asset{part_id}: ArticulationCfg = ArticulationCfg(\n")
        lines.append(f"        prim_path=\"/World/envs/env_.*/AssembledAsset{part_id}\",\n")
        lines.append("        spawn=sim_utils.UsdFileCfg(\n")
        lines.append(f"            usd_path=f'{{assembly_dir}}{{assembled_asset_{part_id}_cfg.usd_path}}',\n")
        lines.append("            activate_contact_sensors=True,\n")
        lines.append("            rigid_props=sim_utils.RigidBodyPropertiesCfg(\n")
        lines.append("                disable_gravity=False,\n")
        lines.append("                max_depenetration_velocity=5.0,\n")
        lines.append("                linear_damping=0.0,\n")
        lines.append("                angular_damping=0.0,\n")
        lines.append("                max_linear_velocity=1000.0,\n")
        lines.append("                max_angular_velocity=3666.0,\n")
        lines.append("                enable_gyroscopic_forces=True,\n")
        lines.append("                solver_position_iteration_count=192,\n")
        lines.append("                solver_velocity_iteration_count=1,\n")
        lines.append("                max_contact_impulse=1e32,\n")
        lines.append("            ),\n")
        lines.append("            articulation_props=sim_utils.ArticulationRootPropertiesCfg(\n")
        lines.append("                enabled_self_collisions=True,\n")
        lines.append("                fix_root_link=True, # add this so the fixed asset is set to have a fixed base\n")
        lines.append("            ),\n")
        lines.append("            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),\n")
        lines.append("            collision_props=sim_utils.CollisionPropertiesCfg(\n")
        lines.append("                contact_offset=0.005,\n")
        lines.append("                rest_offset=0.0\n")
        lines.append("            ),\n")
        lines.append("        ),\n")
        lines.append("        init_state=ArticulationCfg.InitialStateCfg(\n")
        lines.append("            pos=(0.6, 0.0, 0.05),\n")
        lines.append("            rot=(1.0, 0.0, 0.0, 0.0),\n")
        lines.append("            joint_pos={},\n")
        lines.append("            joint_vel={},\n")
        lines.append("        ),\n")
        lines.append("        actuators={}\n")
        lines.append("    )\n\n")  # extra newline for readability

    return lines

def generate_env(template_file, output_file, part_ids):

    with open(template_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            if (not add_asset_instance_in_setup_scene(line, outfile, part_ids)) \
                  and (not add_asset_to_scene_in_setup_scene(line, outfile, part_ids)) \
                    and (not add_asset_initialization(line, outfile, part_ids)):
                outfile.write(line)

    
def add_asset_instance_in_setup_scene(line, outfile, part_ids):

    start_marker = "        ## start: add assembled asset instance"
    end_marker   = "        ## end: add assembled asset instance"

    if (start_marker not in line) and (end_marker not in line):
        return False

    if start_marker in line:
        outfile.write(line)
        for part_id in part_ids:
            outfile.write(f"        self._assembled_asset{part_id} = Articulation(self.cfg_task.assembled_asset{part_id})\n")

        outfile.write(end_marker+"\n")

    return True

def add_asset_to_scene_in_setup_scene(line, outfile, part_ids):

    start_marker = "        ## start: add assembled asset to the scene"
    end_marker   = "        ## end: add assembled asset to the scene"

    if (start_marker not in line) and (end_marker not in line):
        return False

    if start_marker in line:
        outfile.write(line)
        for part_id in part_ids:
            outfile.write(f"        self.scene.articulations[\"assembled_asset{part_id}\"] = self._assembled_asset{part_id}\n")

        outfile.write(end_marker+"\n")

    return True

def add_asset_initialization(line, outfile, part_ids):

    start_marker = "        ## start: set assembled parts to be in the same state as fixed_asset"
    end_marker   = "        ## end: set assembled parts to be in the same state as fixed_asset"

    if (start_marker not in line) and (end_marker not in line):
        return False

    if start_marker in line:
        outfile.write(line)
        for part_id in part_ids:
            outfile.write(f"        self._assembled_asset{part_id}.write_root_state_to_sim(fixed_state, env_ids=env_ids)\n")
            outfile.write(f"        self._assembled_asset{part_id}.reset()\n")

        outfile.write(end_marker+"\n")

    return True

def main():
    parser = argparse.ArgumentParser(description="Update assembly_id and run training script.")
    parser.add_argument("--asset_dir", type=str, help="Path to the directory containing asset data.", default="/home/bingjie/Downloads/all_assembly_asset")
    parser.add_argument("--cfg_path", type=str, help="Path to the file containing assembly_id.", default="source/isaaclab_tasks/isaaclab_tasks/direct/refinery/refinery_tasks_cfg.py")
    parser.add_argument("--cfg_template", type=str, help="Path to the template file.", default="source/isaaclab_tasks/isaaclab_tasks/direct/refinery/refinery_tasks_cfg_template.py")
    parser.add_argument("--env_path", type=str, help="Path to the file containing assembly_id.", default="source/isaaclab_tasks/isaaclab_tasks/direct/refinery/refinery_env.py")
    parser.add_argument("--env_template", type=str, help="Path to the template file.", default="source/isaaclab_tasks/isaaclab_tasks/direct/refinery/refinery_env_template.py")
    parser.add_argument("--assembly_id", type=str, help="New assembly ID to set.")
    parser.add_argument("--step_id", type=int, help="The step of multipart assembly to initialize.")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path.")
    parser.add_argument("--num_envs", type=int, default=128, help="Number of parallel environment.")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed.")
    parser.add_argument("--sample", type=str, default="rand", help="Random seed.")
    parser.add_argument("--ac_func", type=str, default="ucb", help="Random seed.")
    parser.add_argument("--train", action='store_true', help="Run training mode.")
    parser.add_argument("--log_eval", action='store_true', help="Log evaluation results.")
    parser.add_argument("--headless", action='store_true', help="Run in headless mode.")
    args = parser.parse_args()

    part_ids = []
    for f in os.listdir(os.path.join(args.asset_dir, args.assembly_id)):
        if ".obj" in f:
            part_ids.append(f[-5])

    part_ids.sort()
    args.step_id = min(int(part_ids[-1]), args.step_id)

    assembled_part_ids = part_ids[1:args.step_id]
    
    generate_task_cfg(
        args.cfg_template,
        args.cfg_path,
        args.assembly_id,
        assembled_part_ids,
        args.step_id
    )
        
    generate_env(
        args.env_template,
        args.env_path,
        assembled_part_ids
    )

    update_task_param(
        args.cfg_path, 
        args.asset_dir,
        args.assembly_id, 
        args.step_id,
        args.train, 
        args.log_eval,
        args.sample,
        args.ac_func
        )

    bash_command = None
    if args.train:
        bash_command = "./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Refinery-Direct-v0"
        bash_command += f" --seed={str(args.seed)}"
    else:
        if not args.checkpoint: 
            raise ValueError('No checkpoint provided for evaluation.')
        bash_command = "./isaaclab.sh -p scripts/reinforcement_learning/rl_games/play.py --task=Refinery-Direct-v0"
    
    bash_command += f" --num_envs={str(args.num_envs)}"

    if args.checkpoint:
        bash_command += f" --checkpoint={args.checkpoint}"

    if args.headless:
        bash_command += " --headless"

    # Run the bash command
    subprocess.run(bash_command, shell=True, check=True)

if __name__ == "__main__":
    main()
