#!/usr/bin/env python3
"""
Convert the Living Strategy PDF version from Markdown to PDF
This creates a PDF with proper formatting for KDP publishing
"""

import os
import subprocess
import sys

def create_pdf_from_markdown():
    """Convert Markdown to PDF using pandoc"""
    
    # Check if pandoc is available
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        pandoc_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pandoc_available = False
    
    if not pandoc_available:
        print("Pandoc not found. Please install pandoc to convert Markdown to PDF.")
        print("You can download it from: https://pandoc.org/installing.html")
        return False
    
    # Input and output files
    input_file = "living_strategy_pdf_version.md"
    output_file = "Living_Strategy_PDF_Version.pdf"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found!")
        return False
    
    # Pandoc command for PDF conversion
    cmd = [
        'pandoc',
        input_file,
        '-o', output_file,
        '--pdf-engine=wkhtmltopdf',  # or '--pdf-engine=xelatex'
        '--margin-top=1in',
        '--margin-bottom=1in',
        '--margin-left=1in',
        '--margin-right=1in',
        '--font-size=11pt',
        '--toc',  # Generate table of contents
        '--toc-depth=3',
        '--number-sections',  # Number sections
        '--highlight-style=tango',  # Syntax highlighting
        '--standalone'  # Create standalone document
    ]
    
    try:
        print(f"Converting {input_file} to {output_file}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully created {output_file}")
            print(f"File size: {os.path.getsize(output_file)} bytes")
            return True
        else:
            print(f"❌ Error converting to PDF:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running pandoc: {e}")
        return False

def create_html_version():
    """Create HTML version as backup"""
    try:
        subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        pandoc_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pandoc_available = False
    
    if not pandoc_available:
        return False
    
    input_file = "living_strategy_pdf_version.md"
    output_file = "Living_Strategy_PDF_Version.html"
    
    cmd = [
        'pandoc',
        input_file,
        '-o', output_file,
        '--toc',
        '--toc-depth=3',
        '--number-sections',
        '--standalone',
        '--css=style.css'  # Optional CSS file
    ]
    
    try:
        print(f"Creating HTML version: {output_file}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully created {output_file}")
            return True
        else:
            print(f"❌ Error creating HTML: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error creating HTML: {e}")
        return False

if __name__ == "__main__":
    print("Creating PDF version of Living Strategy...")
    print("=" * 50)
    
    # Change to pdf_version directory
    os.chdir('pdf_version')
    
    # Create PDF
    pdf_success = create_pdf_from_markdown()
    
    # Create HTML as backup
    html_success = create_html_version()
    
    print("=" * 50)
    if pdf_success:
        print("✅ PDF version created successfully!")
        print("📄 You can now edit page numbers manually in the PDF")
        print("📤 Ready for KDP upload")
    else:
        print("❌ PDF creation failed")
        if html_success:
            print("✅ HTML version created as backup")
    
    print("\nFiles created:")
    if os.path.exists("Living_Strategy_PDF_Version.pdf"):
        print("- Living_Strategy_PDF_Version.pdf")
    if os.path.exists("Living_Strategy_PDF_Version.html"):
        print("- Living_Strategy_PDF_Version.html")

