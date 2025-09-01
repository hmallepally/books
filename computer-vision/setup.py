#!/usr/bin/env python3
"""
Setup script for Computer Vision development environment
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_sample_data():
    """Create sample data directory and files"""
    print("📁 Creating sample data directory...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/models", exist_ok=True)
    
    # Create sample image (simple pattern)
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save("data/images/sample_image.jpg")
        print("✅ Sample image created")
    except ImportError:
        print("⚠️  PIL not available, skipping sample image creation")
    
    print("✅ Sample data directory created")

def test_imports():
    """Test if key libraries can be imported"""
    print("🧪 Testing imports...")
    
    test_imports = [
        "cv2",
        "numpy",
        "torch",
        "torchvision",
        "matplotlib.pyplot",
        "PIL",
        "sklearn"
    ]
    
    failed_imports = []
    
    for lib in test_imports:
        try:
            __import__(lib)
            print(f"✅ {lib}")
        except ImportError:
            print(f"❌ {lib}")
            failed_imports.append(lib)
    
    if failed_imports:
        print(f"⚠️  Some imports failed: {failed_imports}")
        return False
    
    print("✅ All imports successful")
    return True

def create_sample_code():
    """Create sample code files"""
    print("💻 Creating sample code files...")
    
    # Create basic image processing example
    basic_code = '''import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_display_image(image_path):
    """Load and display an image"""
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title("Sample Image")
    plt.axis('off')
    plt.show()
    
    return image

if __name__ == "__main__":
    # Try to load sample image
    sample_path = "data/images/sample_image.jpg"
    image = load_and_display_image(sample_path)
    
    if image is not None:
        print(f"Image loaded successfully! Shape: {image.shape}")
        print("Setup complete! You can now start learning Computer Vision.")
    else:
        print("Please create a sample image in data/images/ directory")
'''
    
    with open("sample_basic.py", "w") as f:
        f.write(basic_code)
    
    print("✅ Sample code files created")

def create_jupyter_kernel():
    """Create Jupyter kernel for the project"""
    print("📓 Setting up Jupyter kernel...")
    
    try:
        # Install ipykernel if not already installed
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ipykernel"])
        
        # Create kernel
        kernel_name = "computer-vision"
        subprocess.check_call([
            sys.executable, "-m", "ipykernel", "install", 
            "--user", "--name", kernel_name, "--display-name", "Computer Vision"
        ])
        
        print(f"✅ Jupyter kernel '{kernel_name}' created")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Failed to create Jupyter kernel")
        return False

def print_setup_instructions():
    """Print setup completion instructions"""
    print("\n" + "="*60)
    print("🎉 Computer Vision Environment Setup Complete!")
    print("="*60)
    print("\n📚 Next Steps:")
    print("1. Open the book/computer-vision-comprehensive-guide.md file")
    print("2. Run sample code: python sample_basic.py")
    print("3. Start Jupyter: jupyter lab")
    print("4. Choose 'Computer Vision' kernel in notebooks")
    print("\n🔧 Troubleshooting:")
    print("- If you encounter import errors, try: pip install -r requirements.txt")
    print("- For GPU support, install torch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("- Check OpenCV installation: python -c 'import cv2; print(cv2.__version__)'")
    print("\n📖 Happy Learning!")

def main():
    """Main setup function"""
    print("🚀 Setting up Computer Vision development environment...")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.executable}")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Some packages failed to install. Continuing with setup...")
    
    # Create directories and sample data
    create_sample_data()
    
    # Test imports
    test_imports()
    
    # Create sample code
    create_sample_code()
    
    # Setup Jupyter
    create_jupyter_kernel()
    
    # Print instructions
    print_setup_instructions()

if __name__ == "__main__":
    main()

