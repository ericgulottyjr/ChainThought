#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Clean up offload folder")
    parser.add_argument(
        "--offload_folder",
        type=str,
        default="./offload_folder",
        help="Path to the offload folder",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if offload folder exists
    if os.path.exists(args.offload_folder):
        print(f"Cleaning up offload folder: {args.offload_folder}")
        try:
            # Remove the folder and its contents
            shutil.rmtree(args.offload_folder)
            print(f"Successfully removed {args.offload_folder}")
        except Exception as e:
            print(f"Error removing folder: {e}")
    else:
        print(f"Offload folder {args.offload_folder} does not exist. No cleanup needed.")
    
    # Create a fresh empty folder
    try:
        os.makedirs(args.offload_folder, exist_ok=True)
        print(f"Created fresh offload folder: {args.offload_folder}")
    except Exception as e:
        print(f"Error creating folder: {e}")

if __name__ == "__main__":
    main() 