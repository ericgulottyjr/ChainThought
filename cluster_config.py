#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import platform
import torch
from transformers import BitsAndBytesConfig

def get_device_config(offload_folder="./offload_folder"):
    """
    Determine the appropriate device configuration based on available hardware.
    Returns a dict with configuration parameters for model loading.
    """
    # Check hardware environment
    is_mac_arm = platform.system() == "Darwin" and platform.machine() == "arm64"
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    
    # Print environment information
    print(f"System: {platform.system()}, Architecture: {platform.machine()}")
    print(f"CUDA available: {has_cuda}")
    print(f"MPS (Metal Performance Shaders) available: {has_mps}")
    
    # Define configuration based on hardware
    if has_cuda:
        # NVIDIA GPU with CUDA
        print("CUDA GPU detected. Using 4-bit quantization for optimal performance.")
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ),
            "device_map": "auto",
            "torch_dtype": None,  # Use default with quantization
            "offload_folder": offload_folder,
            "offload_state_dict": False,
        }
    elif is_mac_arm and has_mps:
        # Apple Silicon with Metal support
        print("Apple Silicon Mac with MPS detected. Using float16 precision.")
        return {
            "quantization_config": None,
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "offload_folder": offload_folder,
            "offload_state_dict": True,
        }
    else:
        # CPU or other hardware
        print("Using CPU configuration with offloading for memory efficiency.")
        return {
            "quantization_config": None,
            "device_map": {"": "cpu"},  # Explicit CPU mapping
            "torch_dtype": torch.float16,
            "offload_folder": offload_folder,
            "offload_state_dict": True,
        }

def setup_environment():
    """Set up environment variables for optimal performance."""
    # For CPU threading control
    if "OMP_NUM_THREADS" not in os.environ:
        # Use a reasonable default if not already set
        os.environ["OMP_NUM_THREADS"] = "8"
    
    # For MKL (Math Kernel Library) threading
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")
    
    # Print current settings
    print(f"Thread settings: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
          f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    
    return True 