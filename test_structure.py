"""
Simple test script to verify project structure and basic imports
This script tests the project without requiring all dependencies
"""

import os
import sys

def test_file_structure():
    """Test if all required files exist"""
    required_files = [
        'main.py',
        'model.py', 
        'train.py',
        'test_agent.py',
        'utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("🔍 Checking project structure...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def test_python_syntax():
    """Test Python syntax of all Python files"""
    python_files = ['main.py', 'model.py', 'train.py', 'test_agent.py', 'utils.py']
    
    print("\n🔍 Checking Python syntax...")
    syntax_errors = []
    
    for file in python_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
            compile(content, file, 'exec')
            print(f"✅ {file} - Syntax OK")
        except SyntaxError as e:
            print(f"❌ {file} - Syntax Error: {e}")
            syntax_errors.append(file)
        except Exception as e:
            print(f"⚠️  {file} - Warning: {e}")
    
    if syntax_errors:
        print(f"\n❌ Syntax errors in: {syntax_errors}")
        return False
    else:
        print("\n✅ All Python files have valid syntax!")
        return True

def test_imports():
    """Test basic imports (without heavy dependencies)"""
    print("\n🔍 Testing basic imports...")
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError:
        print("❌ numpy - Not installed")
    
    try:
        import matplotlib
        print("✅ matplotlib")
    except ImportError:
        print("❌ matplotlib - Not installed")
    
    try:
        import cv2
        print("✅ opencv-python")
    except ImportError:
        print("❌ opencv-python - Not installed")
    
    try:
        import gym
        print("✅ gym")
    except ImportError:
        print("❌ gym - Not installed")
    
    try:
        import tensorflow as tf
        print("✅ tensorflow")
    except ImportError:
        print("❌ tensorflow - Not installed")
    
    try:
        import rl
        print("✅ keras-rl2")
    except ImportError:
        print("❌ keras-rl2 - Not installed")

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 SPACE INVADERS DQN PROJECT - STRUCTURE TEST")
    print("=" * 60)
    
    # Test file structure
    structure_ok = test_file_structure()
    
    # Test Python syntax
    syntax_ok = test_python_syntax()
    
    # Test imports
    test_imports()
    
    print("\n" + "=" * 60)
    if structure_ok and syntax_ok:
        print("✅ PROJECT STRUCTURE TEST PASSED!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test: python main.py --mode check")
        print("3. Start training: python main.py --mode train")
    else:
        print("❌ PROJECT STRUCTURE TEST FAILED!")
        print("Please fix the issues above before proceeding.")
    print("=" * 60)

if __name__ == "__main__":
    main()


