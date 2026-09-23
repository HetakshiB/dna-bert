#!/usr/bin/env python3
"""CUDA & Environment Verification Script for Multimodal CRISPR Predictor.

Checks PyTorch installation, CUDA availability, GPU device details,
and core multimodal package dependencies.
"""

import sys
import platform
import subprocess

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def check_system_hardware():
    """Query Windows display devices for diagnostic context."""
    if platform.system() == "Windows":
        try:
            cmd = "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
            adapters = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            return adapters
        except Exception:
            return []
    return []

def main():
    print("=" * 65)
    print(" Multimodal CRISPR Predictor - System & CUDA Verification")
    print("=" * 65)
    print(f"Python Version  : {platform.python_version()} ({platform.python_implementation()})")
    print(f"Executable      : {sys.executable}")
    print(f"Operating System: {platform.system()} {platform.release()} ({platform.machine()})")
    print("-" * 65)

    # 1. PyTorch & CUDA Check
    print("[1] PyTorch & Device Detection:")
    try:
        import torch
        print(f"  [OK] PyTorch is installed (version {torch.__version__})")
        
        cuda_available = torch.cuda.is_available()
        print(f"  [..] torch.cuda.is_available(): {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"  [OK] CUDA Version : {torch.version.cuda}")
            print(f"  [OK] Device Count : {device_count}")
            for idx in range(device_count):
                dev_name = torch.cuda.get_device_name(idx)
                capability = torch.cuda.get_device_capability(idx)
                total_mem = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
                print(f"       - Device [{idx}]: {dev_name} (Compute: {capability[0]}.{capability[1]}, VRAM: {total_mem:.2f} GB)")
            active_device = torch.device("cuda")
        else:
            print("  [WARN] CUDA is NOT available for PyTorch on this host.")
            detected_gpus = check_system_hardware()
            if detected_gpus:
                print(f"  [INFO] Detected Display Adapter(s): {', '.join(detected_gpus)}")
                if not any("nvidia" in g.lower() for g in detected_gpus):
                    print("  [INFO] Note: No NVIDIA GPU detected. Integrated graphics (e.g. Intel Iris Xe) do not support CUDA.")
            print("  [INFO] PyTorch will execute using CPU tensors.")
            active_device = torch.device("cpu")

        # Test simple tensor operation
        x = torch.randn(3, 3, device=active_device)
        y = torch.matmul(x, x)
        print(f"  [OK] Tensor matrix multiplication on device '{active_device}' successful.")

    except ImportError:
        print("  [FAIL] PyTorch is NOT installed in this Python environment.")
        print("         Run: pip install torch")

    print("-" * 65)

    # 2. Key Libraries Check
    print("[2] Multimodal Stack Dependencies:")
    packages = [
        ("transformers", "transformers"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("scanpy", "scanpy"),
        ("tqdm", "tqdm"),
        ("tensorboard", "tensorboard"),
    ]

    for display_name, module_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK] {display_name:<15} : {ver}")
        except ImportError:
            print(f"  [FAIL] {display_name:<15} : NOT installed")

    # BigWig accessor check
    bigwig_found = False
    try:
        import pyBigWig
        print(f"  [OK] {'pyBigWig':<15} : {getattr(pyBigWig, '__version__', 'installed')} (C-extension)")
        bigwig_found = True
    except ImportError:
        pass

    try:
        import pybigtools
        print(f"  [OK] {'pybigtools':<15} : {getattr(pybigtools, '__version__', 'installed')} (Rust BigWig engine)")
        bigwig_found = True
    except ImportError:
        pass

    if not bigwig_found:
        print(f"  [WARN] {'BigWig Reader':<15} : Neither pyBigWig nor pybigtools is installed")

    print("=" * 65)

if __name__ == "__main__":
    main()
