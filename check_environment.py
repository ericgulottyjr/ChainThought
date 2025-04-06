#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import platform
import subprocess
import sys

def check_gpu_memory():
    """Check available GPU memory if CUDA is available."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} CUDA device(s):")
        for i in range(device_count):
            device = torch.cuda.get_device_properties(i)
            total_memory = device.total_memory / (1024**3)  # Convert to GB
            print(f"  Device {i}: {device.name}, Memory: {total_memory:.2f} GB")
    else:
        print("No CUDA devices available")

def check_cpu_info():
    """Get CPU information."""
    try:
        if platform.system() == "Linux":
            # Get CPU info from /proc/cpuinfo
            cmd = "cat /proc/cpuinfo | grep 'model name' | uniq"
            result = subprocess.check_output(cmd, shell=True).decode().strip()
            print(f"CPU: {result.split(':')[1].strip()}")
            
            # Get number of cores
            cmd = "nproc"
            result = subprocess.check_output(cmd, shell=True).decode().strip()
            print(f"CPU Cores: {result}")
        else:
            # For macOS and other systems
            print(f"CPU: {platform.processor()}")
            import multiprocessing
            print(f"CPU Cores: {multiprocessing.cpu_count()}")
    except Exception as e:
        print(f"Could not determine CPU info: {e}")

def check_cluster_modules():
    """Check if we're in a cluster environment with module system."""
    try:
        # Check if the module command exists
        result = subprocess.run("which module", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("\n=== Cluster Module System Detected ===")
            # List loaded modules
            result = subprocess.run("module list", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("Currently loaded modules:")
                print(result.stdout)
            else:
                print("Module command exists but 'module list' failed")
                
            # Check for specific modules we need
            for module in ["python", "cuda", "pytorch"]:
                result = subprocess.run(f"module avail {module}", shell=True, capture_output=True, text=True)
                if module in result.stdout:
                    print(f"Module '{module}' is available")
        else:
            print("No cluster module system detected")
    except Exception as e:
        print(f"Error checking cluster modules: {e}")

def check_slurm():
    """Check if SLURM job scheduler is available."""
    try:
        # Check if squeue command exists
        result = subprocess.run("which squeue", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("\n=== SLURM Job Scheduler Detected ===")
            # Check if we're running in a SLURM job
            if "SLURM_JOB_ID" in os.environ:
                print(f"Running in SLURM job: {os.environ['SLURM_JOB_ID']}")
                
                # Get job info
                job_id = os.environ["SLURM_JOB_ID"]
                result = subprocess.run(f"scontrol show job {job_id}", shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    # Extract and print key job information
                    for line in result.stdout.split("\n"):
                        if any(key in line for key in ["JobId", "NumNodes", "NumCPUs", "GPUS", "TimeLimit"]):
                            print(f"  {line.strip()}")
            else:
                print("SLURM available but not running in a SLURM job")
    except Exception as e:
        print(f"Error checking SLURM: {e}")

def main():
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {platform.python_version()}")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n=== CPU Information ===")
    check_cpu_info()
    
    print("\n=== CUDA Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'Not available'}")
    
    print("\n=== GPU Information ===")
    check_gpu_memory()
    
    print("\n=== MPS Information ===")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    print("\n=== Environment Variables ===")
    for var in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTHONPATH", "LD_LIBRARY_PATH", "PATH"]:
        print(f"{var}: {os.environ.get(var, 'Not set')}")
    
    # Check for cluster-specific environments
    check_cluster_modules()
    check_slurm()
    
    print("\n=== Package Versions ===")
    packages = [
        "transformers", "bitsandbytes", "peft", "datasets", "accelerate", 
        "torch", "safetensors", "trl", "wandb", "tensorboard"
    ]
    for package in packages:
        try:
            module = __import__(package)
            print(f"{package}: {module.__version__}")
        except (ImportError, AttributeError):
            print(f"{package}: Not installed or version not available")
    
    print("\n=== Disk Space ===")
    try:
        if platform.system() == "Linux" or platform.system() == "Darwin":
            subprocess.run("df -h .", shell=True)
    except Exception as e:
        print(f"Error checking disk space: {e}")
    
    print("\n=== Memory Information ===")
    try:
        if platform.system() == "Linux":
            subprocess.run("free -h", shell=True)
    except Exception as e:
        print(f"Error checking memory: {e}")

if __name__ == "__main__":
    main() 