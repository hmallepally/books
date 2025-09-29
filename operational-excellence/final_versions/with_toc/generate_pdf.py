#!/usr/bin/env python3
"""
PDF Generation Script for Operational Excellence with AI Book
"""

import pdfkit
import os
from pathlib import Path

def generate_pdf():
    """Generate PDF from HTML file"""
    
    # Get current directory
    current_dir = Path(__file__).parent
    
    # Input and output file paths
    html_file = current_dir / "Operational_Excellence_with_AI_WITH_TOC.html"
    pdf_file = current_dir / "Operational_Excellence_with_AI_WITH_TOC.pdf"
    
    print(f"Converting {html_file} to {pdf_file}")
    
    # PDF generation options
    options = {
        'page-size': 'Letter',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'no-outline': None,
        'enable-local-file-access': None,
        'print-media-type': None,
        'disable-smart-shrinking': None,
        'dpi': 300,
        'image-quality': 100,
        'javascript-delay': 1000,
        'load-error-handling': 'ignore',
        'load-media-error-handling': 'ignore'
    }
    
    try:
        # Generate PDF
        pdfkit.from_file(
            str(html_file),
            str(pdf_file),
            options=options
        )
        
        print(f"✅ PDF generated successfully: {pdf_file}")
        
        # Get file size
        file_size = pdf_file.stat().st_size
        print(f"📄 File size: {file_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating PDF: {e}")
        return False

if __name__ == "__main__":
    success = generate_pdf()
    if success:
        print("\n🎉 PDF generation completed successfully!")
    else:
        print("\n💥 PDF generation failed. Please check the error messages above.")