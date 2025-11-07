#!/usr/bin/env python3
"""
Script để dọn dẹp các file không cần thiết
"""

import os
import shutil
from pathlib import Path


def clean_pycache(root_dir="."):
    """Xóa tất cả __pycache__ folders"""
    root = Path(root_dir)
    count = 0
    
    for pycache in root.rglob("__pycache__"):
        if pycache.is_dir():
            print(f"Removing {pycache}")
            shutil.rmtree(pycache)
            count += 1
    
    print(f"✓ Removed {count} __pycache__ folders")


def clean_pyc_files(root_dir="."):
    """Xóa tất cả .pyc, .pyo files"""
    root = Path(root_dir)
    count = 0
    
    for pattern in ["*.pyc", "*.pyo", "*.pyd"]:
        for pyc_file in root.rglob(pattern):
            if pyc_file.is_file():
                print(f"Removing {pyc_file}")
                pyc_file.unlink()
                count += 1
    
    print(f"✓ Removed {count} compiled Python files")


def clean_temp_data():
    """Xóa các file tạm trong data folder"""
    temp_files = [
        "../data/multiwoz24/MULTIWOZ2.4.zip",
        "../data/multiwoz24/MULTIWOZ2.4"
    ]
    
    count = 0
    for temp_file in temp_files:
        temp_path = Path(temp_file)
        if temp_path.exists():
            if temp_path.is_dir():
                print(f"Removing directory {temp_path}")
                shutil.rmtree(temp_path)
            else:
                print(f"Removing file {temp_path}")
                temp_path.unlink()
            count += 1
    
    print(f"✓ Removed {count} temporary data files/folders")


def show_disk_usage():
    """Hiển thị dung lượng các folder"""
    import subprocess
    
    print("\n" + "=" * 60)
    print("DISK USAGE")
    print("=" * 60)
    
    folders = [
        "../data/multiwoz24",
        "../data/processed",
        "../venv"
    ]
    
    for folder in folders:
        if Path(folder).exists():
            result = subprocess.run(
                ["du", "-sh", folder],
                capture_output=True,
                text=True
            )
            print(result.stdout.strip())


def main():
    print("=" * 60)
    print("CLEANING UP PROJECT")
    print("=" * 60)
    
    # Clean Python cache
    print("\n1. Cleaning Python cache files...")
    clean_pycache("..")
    clean_pyc_files("..")
    
    # Clean temporary data
    print("\n2. Cleaning temporary data files...")
    clean_temp_data()
    
    # Show disk usage
    show_disk_usage()
    
    print("\n" + "=" * 60)
    print("✓ CLEANUP COMPLETE!")
    print("=" * 60)
    
    print("\nFiles kept:")
    print("  data/multiwoz24/:")
    print("    - data.json (263 MB)")
    print("    - ontology.json")
    print("    - valListFile.json")
    print("    - testListFile.json")
    print("\n  data/processed/:")
    print("    - train.json, val.json, test.json")
    print("    - train_stats.json, val_stats.json, test_stats.json")
    print("    - dataset_stats.json")
    print("    - ontology.json")


if __name__ == "__main__":
    main()
