import sys
import os
import platform
import shutil
import psutil
import torch
import subprocess

print("================================================================")
print("TRUTHLENS AI - SYSTEM & RUNTIME ENVIRONMENT INSPECTION")
print("================================================================")

# 1. Operating System
print("\n[1. OPERATING SYSTEM]")
print(f"OS Name        : {platform.system()} {platform.release()}")
print(f"OS Version     : {platform.version()}")
print(f"Architecture   : {platform.machine()}")

# 2. RAM
print("\n[2. MEMORY / RAM]")
mem = psutil.virtual_memory()
print(f"Total RAM      : {mem.total / (1024**3):.2f} GB ({mem.total:,} bytes)")
print(f"Available RAM  : {mem.available / (1024**3):.2f} GB ({mem.available:,} bytes)")
print(f"Used RAM       : {mem.used / (1024**3):.2f} GB ({mem.percent}%)")

# 3. CPU
print("\n[3. CPU PROCESSOR]")
print(f"Physical Cores : {psutil.cpu_count(logical=False)}")
print(f"Logical Threads: {psutil.cpu_count(logical=True)}")

# 4. GPU & VRAM via WMIC
print("\n[4. GPU HARDWARE & VRAM]")
try:
    res = subprocess.run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM,DriverVersion"], capture_output=True, text=True, check=True)
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    if len(lines) >= 2:
        print(f"Headers        : {lines[0]}")
        for l in lines[1:]:
            print(f"Video Device   : {l}")
    else:
        print(res.stdout.strip())
except Exception as e:
    print(f"GPU Query Error: {e}")

# 5. PyTorch & CUDA
print("\n[5. PYTORCH & CUDA ACCELERATION]")
print(f"Python Exec    : {sys.executable}")
print(f"Python Version : {sys.version.split(chr(10))[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("CUDA Device    : None detected (PyTorch is configured for pure CPU execution)")

# 6. Disk Storage
print("\n[6. DISK STORAGE]")
for drive in ['C:\\', 'E:\\']:
    if os.path.exists(drive):
        total, used, free = shutil.disk_usage(drive)
        pct_free = (free / total) * 100
        print(f"Drive {drive:<9}: Total = {total/(1024**3):6.2f} GB | Used = {used/(1024**3):6.2f} GB | Free = {free/(1024**3):6.2f} GB ({pct_free:.1f}% free)")
print("================================================================")
