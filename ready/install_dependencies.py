#!/usr/bin/env python3
"""
Install missing dependencies for GLASS Fabric Detection
"""
import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install():
    """Check for missing packages and install them"""
    required_packages = [
        "psutil>=5.8.0",
        "pynvml>=11.0.0",
        "timm>=0.6.0"
    ]
    
    missing_packages = []
    
    # Check each package
    for package in required_packages:
        package_name = package.split(">=")[0]
        try:
            __import__(package_name)
            print(f"✅ {package_name} - already installed")
        except ImportError:
            print(f"❌ {package_name} - missing")
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}")
        
        print("\nAll dependencies checked!")
    else:
        print("\n🎉 All required packages are already installed!")

if __name__ == "__main__":
    print("🔧 GLASS Dependencies Installer")
    print("=" * 40)
    check_and_install()