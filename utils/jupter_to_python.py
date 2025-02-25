"""
A CLI-based tool to convert a jupyter notebook to a python script and fix formatting
"""

import subprocess
import argparse
from tqdm import tqdm

ignore_lines = ['#!/usr/bin/env python', '# coding: utf-8']

def fix_blank_lines(lines):
    """Remove consecutive blank lines

    Args:
        lines (list): List of lines

    Returns:
        list: List of lines with consecutive blank lines removed
    """
    fixed_lines = []

    prev_was_blank = False

    for line in lines:
        if line == '\n':
            if prev_was_blank:
                continue
            prev_was_blank = True
        else:
            prev_was_blank = False

        fixed_lines.append(line)
    
    return fixed_lines

def main(args):
    print(f"Converting {args.path} to python script")
    subprocess.run(['jupyter', 'nbconvert', '--to', 'script', args.path])

    script_path = args.path.replace('.ipynb', '.py')

    print(f"Reading from {script_path}")
    with open(script_path, 'r') as f:
        lines = f.readlines()
    print(f"Read {len(lines)} lines")
    
    body_lines = []
    import_lines = []

    last_was_empty = False

    for line in tqdm(lines, desc='Removing jupyter formatting'):    
        if any([line.startswith(ignore_line) for ignore_line in ignore_lines]):
            continue
        
        if 'import' in line or line.startswith('sys.path.append'):
            import_lines.append(line)
            continue
        
        if line.startswith('# In['):
            continue
        
        if line.startswith('# #'):
            line = line[2:]

        if line.endswith('\n\n'):
            line = line[:-1]

        else:
            body_lines.append(line)
    
    fixed_lines = import_lines + body_lines

    fixed_lines = fix_blank_lines(fixed_lines)

    print(f"Writing to {script_path}")
    with open(script_path, 'w') as f:
        f.writelines(fixed_lines)

    print("Done.")
    
    print(f"Converted {args.path} to {script_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert jupyter notebook to python script and remove jupyter's formatting")
    parser.add_argument('--path', type=str, help='Path to jupyter notebook')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)

