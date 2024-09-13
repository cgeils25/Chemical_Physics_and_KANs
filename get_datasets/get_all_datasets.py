"""
Download all associated datasets and store them in a directory called 'datasets'.

To run, use the following command:

python get_datasets/get_all_datasets.py

Alternatively, you can specify a directory to store the datasets:

python get_datasets/get_all_datasets.py --dir 'path/to/directory'
"""

import os
import sys
import requests
import argparse


def get_all_datasets():
    pass

def get_args():
    # argument parser that accepts alternate directory to store datasets
    parser = argparse.ArgumentParser(description='Download all datasets')
    parser.add_argument('--dir', type=str, default='datasets', help='Directory to store datasets')
    return parser.parse_args()

def main(dir: str):
    """Main function to download all datasets

    Args:
        dir (str): _description_
    """
    # make datasets directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)

    get_all_datasets()

if __name__ == '__main__':
    args = get_args()
    main(args.dir)