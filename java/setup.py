#!/usr/bin/env python3
"""
Core Java Mastery Guide - Setup Script
This script helps set up the Java development environment for the book.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print a welcome header for the setup."""
    print("=" * 60)
    print("🚀 Core Java Mastery Guide - Setup Script")
    print("=" * 60)
    print()

def check_java_installation():
    """Check if Java is installed and return version info."""
    print("🔍 Checking Java installation...")
    
    try:
        # Check Java version
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, check=True)
        java_version = result.stderr.strip().split('\n')[0]
        print(f"✅ Java found: {java_version}")
        
        # Check Java compiler version
        result = subprocess.run(['javac', '-version'], 
                              capture_output=True, text=True, check=True)
        javac_version = result.stdout.strip()
        print(f"✅ Java Compiler found: {javac_version}")
        
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Java is not installed or not in PATH")
        return False
    except FileNotFoundError:
        print("❌ Java is not installed or not in PATH")
        return False

def check_java_home():
    """Check and display JAVA_HOME environment variable."""
    print("\n🔍 Checking JAVA_HOME environment variable...")
    
    java_home = os.environ.get('JAVA_HOME')
    if java_home:
        print(f"✅ JAVA_HOME is set to: {java_home}")
        
        # Check if the path exists
        if os.path.exists(java_home):
            print("✅ JAVA_HOME path exists")
        else:
            print("⚠️  JAVA_HOME path does not exist")
    else:
        print("⚠️  JAVA_HOME is not set")
        
        # Try to find Java installation
        print("🔍 Attempting to find Java installation...")
        if platform.system() == "Windows":
            possible_paths = [
                "C:\\Program Files\\Java\\jdk-11",
                "C:\\Program Files\\Java\\jdk-17",
                "C:\\Program Files\\Eclipse Adoptium\\jdk-11",
                "C:\\Program Files\\Eclipse Adoptium\\jdk-17"
            ]
        else:
            possible_paths = [
                "/usr/lib/jvm/java-11-openjdk",
                "/usr/lib/jvm/java-17-openjdk",
                "/usr/java/jdk-11",
                "/usr/java/jdk-17"
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"💡 Found Java installation at: {path}")
                print(f"   Consider setting JAVA_HOME to: {path}")
                break

def create_directories():
    """Create necessary directories for the Java book."""
    print("\n📁 Creating necessary directories...")
    
    directories = [
        'code',
        'exercises', 
        'notebooks',
        'data',
        'models',
        'results'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"ℹ️  Directory already exists: {directory}")

def create_sample_files():
    """Create sample Java files for testing."""
    print("\n📝 Creating sample Java files...")
    
    # Create a simple Hello World program
    hello_world_code = '''public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, Java World!");
        System.out.println("Java version: " + System.getProperty("java.version"));
        System.out.println("JVM vendor: " + System.getProperty("java.vendor"));
    }
}'''
    
    with open('code/HelloWorld.java', 'w') as f:
        f.write(hello_world_code)
    print("✅ Created: code/HelloWorld.java")
    
    # Create a simple test runner
    test_runner_code = '''import java.io.File;
import java.io.IOException;

public class TestRunner {
    public static void main(String[] args) {
        System.out.println("🧪 Running Java compilation test...");
        
        try {
            // Compile HelloWorld.java
            ProcessBuilder pb = new ProcessBuilder("javac", "code/HelloWorld.java");
            Process process = pb.start();
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                System.out.println("✅ HelloWorld.java compiled successfully!");
                
                // Run the compiled class
                ProcessBuilder runPb = new ProcessBuilder("java", "-cp", "code", "HelloWorld");
                Process runProcess = runPb.start();
                runProcess.waitFor();
                
                System.out.println("✅ HelloWorld executed successfully!");
            } else {
                System.out.println("❌ Compilation failed!");
            }
            
        } catch (IOException | InterruptedException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}'''
    
    with open('code/TestRunner.java', 'w') as f:
        f.write(test_runner_code)
    print("✅ Created: code/TestRunner.java")

def run_basic_tests():
    """Run basic tests to verify Java setup."""
    print("\n🧪 Running basic Java tests...")
    
    try:
        # Compile HelloWorld
        result = subprocess.run(['javac', 'code/HelloWorld.java'], 
                              capture_output=True, text=True, check=True)
        print("✅ HelloWorld.java compiled successfully")
        
        # Run HelloWorld
        result = subprocess.run(['java', '-cp', 'code', 'HelloWorld'], 
                              capture_output=True, text=True, check=True)
        print("✅ HelloWorld executed successfully")
        print("Output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        print(f"Error output: {e.stderr}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def print_setup_instructions():
    """Print setup instructions for the user."""
    print("\n" + "=" * 60)
    print("📚 Setup Instructions")
    print("=" * 60)
    
    print("""
1. **Java Installation** ✅
   - Java 11 or higher is required
   - Make sure both 'java' and 'javac' commands work

2. **IDE Setup** (Choose one):
   - IntelliJ IDEA: Download from https://www.jetbrains.com/idea/download/
   - Eclipse: Download from https://www.eclipse.org/downloads/
   - VS Code: Install Java extensions

3. **Project Structure** ✅
   - All necessary directories have been created
   - Sample Java files are ready for testing

4. **Next Steps**:
   - Open the project in your preferred IDE
   - Start with Chapter 1: Java Fundamentals
   - Run the sample programs to verify setup
   - Begin your Java mastery journey!

5. **Troubleshooting**:
   - If Java is not found, install JDK 11+ from:
     https://adoptium.net/ or https://www.oracle.com/java/technologies/downloads/
   - Set JAVA_HOME environment variable if needed
   - Ensure Java is in your system PATH

6. **Book Structure**:
   - 📖 book/ - Main book content
   - 💻 code/ - Code examples and samples
   - 🎯 exercises/ - Practice problems
   - 📓 notebooks/ - Jupyter notebooks (if applicable)
   - 📊 data/ - Sample data files
   - 🏗️ models/ - Generated models
   - 📈 results/ - Output and results
""")

def main():
    """Main setup function."""
    print_header()
    
    # Check Java installation
    if not check_java_installation():
        print("\n❌ Java is not properly installed.")
        print("Please install Java 11 or higher and try again.")
        return
    
    # Check JAVA_HOME
    check_java_home()
    
    # Create directories
    create_directories()
    
    # Create sample files
    create_sample_files()
    
    # Run basic tests
    run_basic_tests()
    
    # Print setup instructions
    print_setup_instructions()
    
    print("\n🎉 Setup completed successfully!")
    print("Ready to master Core Java! 🚀")

if __name__ == "__main__":
    main()
