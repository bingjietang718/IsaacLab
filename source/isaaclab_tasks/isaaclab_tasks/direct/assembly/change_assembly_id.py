#!/usr/bin/env python3
import argparse
import re

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
    parser = argparse.ArgumentParser(
        description="Find and update the line containing 'assembly_id =' in a Python file."
    )
    parser.add_argument("id", help="New assembly id value to set (e.g., '00015')")
    
    args = parser.parse_args()
    update_assembly_id("./source/isaaclab_tasks/isaaclab_tasks/direct/assembly/assembly_tasks_cfg.py", args.id)

if __name__ == '__main__':
    main()
