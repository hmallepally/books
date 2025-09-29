#!/usr/bin/env python3
"""
Convert the Living Strategy PNG title image to PDF format
"""

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("PIL not available. Please install Pillow: pip install Pillow")

import os

def create_title_pdf():
    """Convert the Living Strategy PNG to PDF"""
    if not PIL_AVAILABLE:
        print("Cannot create PDF without PIL. Please install Pillow.")
        return
    
    # Check if the source image exists
    source_image = "living strategy.png"
    if not os.path.exists(source_image):
        print(f"Source image '{source_image}' not found!")
        return
    
    try:
        # Open the PNG image
        img = Image.open(source_image)
        
        # Convert to RGB if necessary (PDF requires RGB)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as PDF
        pdf_filename = "Living_Strategy_Title.pdf"
        img.save(pdf_filename, "PDF", resolution=300.0, quality=95)
        
        print(f"Successfully created {pdf_filename}")
        print(f"Image size: {img.size[0]}x{img.size[1]} pixels")
        print(f"Image mode: {img.mode}")
        
        # Also create a copy in the enhanced_version folder
        enhanced_pdf = "enhanced_version/Living_Strategy_Title.pdf"
        img.save(enhanced_pdf, "PDF", resolution=300.0, quality=95)
        print(f"Also saved as {enhanced_pdf}")
        
    except Exception as e:
        print(f"Error creating PDF: {e}")

if __name__ == "__main__":
    create_title_pdf()

