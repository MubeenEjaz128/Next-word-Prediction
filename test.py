#!/usr/bin/env python3
"""
Test script for Sherlock Holmes Text Generator
"""

import os
import sys
import torch
import pickle
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from models import TextGenerationModel
        print("✅ models.py imported successfully")
    except Exception as e:
        print(f"❌ Error importing models.py: {e}")
        return False
    
    try:
        from utils import Tokenizer, get_device
        print("✅ utils.py imported successfully")
    except Exception as e:
        print(f"❌ Error importing utils.py: {e}")
        return False
    
    try:
        from config import config
        print("✅ config.py imported successfully")
    except Exception as e:
        print(f"❌ Error importing config.py: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation"""
    print("\n🧠 Testing model creation...")
    
    try:
        from models import TextGenerationModel
        from config import config
        
        model = TextGenerationModel(
            vocab_size=1000,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout
        )
        
        # Test forward pass
        x = torch.randint(0, 1000, (1, 10))
        output = model(x)
        
        print(f"✅ Model created successfully")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Vocabulary size: 1000")
        
        return True
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def test_tokenizer():
    """Test tokenizer functionality"""
    print("\n🔤 Testing tokenizer...")
    
    try:
        from utils import Tokenizer
        
        tokenizer = Tokenizer(max_words=1000, max_sequence_len=20)
        
        # Test text fitting
        sample_texts = [
            "Sherlock Holmes was a detective.",
            "Watson was his friend and assistant."
        ]
        
        tokenizer.fit_on_texts(sample_texts)
        
        # Test sequence conversion
        sequences = tokenizer.texts_to_sequences(sample_texts)
        
        print(f"✅ Tokenizer created successfully")
        print(f"   Vocabulary size: {len(tokenizer.word_index)}")
        print(f"   Sample sequences: {sequences}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing tokenizer: {e}")
        return False

def test_config():
    """Test configuration system"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from config import config
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Model embedding dim: {config.model.embedding_dim}")
        print(f"   Model hidden dim: {config.model.hidden_dim}")
        print(f"   Training epochs: {config.training.epochs}")
        print(f"   App host: {config.app.host}")
        print(f"   App port: {config.app.port}")
        
        return True
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n📁 Testing model files...")
    
    model_files = [
        "sherlock_holmes_model.pth",
        "tokenizer.pkl", 
        "max_sequence_len.pkl"
    ]
    
    all_exist = True
    for file in model_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"✅ {file} exists ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file} not found")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Some model files are missing. You'll need to train the model first.")
        print("   Run: python train.py")
    
    return all_exist

def test_data_file():
    """Test if training data exists"""
    print("\n📚 Testing training data...")
    
    data_file = "sherlock-holm.es_stories_plain-text_advs.txt"
    
    if os.path.exists(data_file):
        size_mb = os.path.getsize(data_file) / (1024 * 1024)
        print(f"✅ Training data exists ({size_mb:.1f} MB)")
        
        # Check if file has content
        with open(data_file, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if first_line:
                print(f"   First line: {first_line[:50]}...")
            else:
                print("   ⚠️  File appears to be empty")
                return False
        return True
    else:
        print(f"❌ Training data file not found: {data_file}")
        return False

def test_dependencies():
    """Test if all dependencies are installed"""
    print("\n📦 Testing dependencies...")
    
    dependencies = [
        "torch",
        "numpy", 
        "flask",
        "sklearn",
        "matplotlib",
        "seaborn",
        "tqdm"
    ]
    
    all_installed = True
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"❌ {dep} not installed")
            all_installed = False
    
    return all_installed

def main():
    """Run all tests"""
    print("🧪 Testing Sherlock Holmes Text Generator")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Tokenizer", test_tokenizer),
        ("Model Creation", test_model_creation),
        ("Training Data", test_data_file),
        ("Model Files", test_model_files)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n📋 Next steps:")
        if not test_model_files():
            print("1. Train the model: python train.py")
        print("2. Run the web app: python app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Ensure training data file exists")
        print("- Train the model first: python train.py")

if __name__ == "__main__":
    main() 