#!/usr/bin/env python3
"""
Setup script for Sherlock Holmes Text Generator
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def check_data_file():
    """Check if training data file exists"""
    data_file = "sherlock-holm.es_stories_plain-text_advs.txt"
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"✅ Training data found: {data_file} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Training data file not found: {data_file}")
        print("Please ensure the Sherlock Holmes text file is in the project directory")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ["templates", "logs", "models"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("ℹ️  CUDA not available - will use CPU")
            return False
    except ImportError:
        print("ℹ️  PyTorch not installed yet")
        return False

def main():
    """Main setup function"""
    print("🔧 Setting up Sherlock Holmes Text Generator")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    check_cuda()
    
    # Create directories
    create_directories()
    
    # Check data file
    if not check_data_file():
        print("\n⚠️  Warning: Training data not found")
        print("You can still run the web app if you have a trained model")
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Train the model: python train.py")
    print("2. Run the web app: python app.py")
    print("3. Open your browser to: http://127.0.0.1:5000")
    
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 