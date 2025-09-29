#!/usr/bin/env python3
"""
Generate PDF from the optimized HTML content with table structures
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_pdf():
    """Generate PDF from the optimized HTML file"""
    
    # Define paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    output_file = "Operational_Excellence_with_AI_OPTIMIZED_TABLES.pdf"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print(f"Generating PDF from: {html_file}")
    print(f"Output file: {output_file}")
    
    try:
        # Use wkhtmltopdf for better table rendering
        cmd = [
            "wkhtmltopdf",
            "--page-size", "A4",
            "--margin-top", "0.75in",
            "--margin-right", "0.75in", 
            "--margin-bottom", "0.75in",
            "--margin-left", "0.75in",
            "--encoding", "UTF-8",
            "--enable-local-file-access",
            "--print-media-type",
            "--disable-smart-shrinking",
            "--zoom", "1.0",
            "--dpi", "300",
            html_file,
            output_file
        ]
        
        print("Running command:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ PDF generated successfully: {output_file}")
            print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
            return True
        else:
            print(f"❌ Error generating PDF:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ wkhtmltopdf not found. Trying alternative method...")
        return try_alternative_method(html_file, output_file)
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def try_alternative_method(html_file, output_file):
    """Try alternative PDF generation methods"""
    
    # Try using weasyprint
    try:
        import weasyprint
        
        print("Using WeasyPrint for PDF generation...")
        
        # Read HTML content
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Generate PDF
        html_doc = weasyprint.HTML(string=html_content, base_url=os.path.dirname(os.path.abspath(html_file)))
        html_doc.write_pdf(output_file)
        
        print(f"✅ PDF generated successfully with WeasyPrint: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ WeasyPrint not available. Trying pdfkit...")
        return try_pdfkit_method(html_file, output_file)
    except Exception as e:
        print(f"❌ WeasyPrint error: {e}")
        return try_pdfkit_method(html_file, output_file)

def try_pdfkit_method(html_file, output_file):
    """Try using pdfkit as fallback"""
    
    try:
        import pdfkit
        
        print("Using pdfkit for PDF generation...")
        
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in', 
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'enable-local-file-access': None,
            'print-media-type': None,
            'disable-smart-shrinking': None,
            'zoom': '1.0',
            'dpi': '300'
        }
        
        pdfkit.from_file(html_file, output_file, options=options)
        
        print(f"✅ PDF generated successfully with pdfkit: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ pdfkit not available. Please install one of:")
        print("   - wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        print("   - pip install weasyprint")
        print("   - pip install pdfkit")
        return False
    except Exception as e:
        print(f"❌ pdfkit error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Generating PDF from optimized HTML content...")
    print("=" * 60)
    
    success = generate_pdf()
    
    if success:
        print("\n✅ PDF generation completed successfully!")
        print("\n📋 Review the PDF to check:")
        print("   • Table formatting and readability")
        print("   • Space utilization improvements")
        print("   • Professional appearance")
        print("   • Page count reduction")
        print("\n🔄 If satisfied, we can apply similar optimizations to other sections.")
    else:
        print("\n❌ PDF generation failed. Please check the error messages above.")
        print("\n💡 Suggested solutions:")
        print("   1. Install wkhtmltopdf: https://wkhtmltopdf.org/downloads.html")
        print("   2. Or install Python packages: pip install weasyprint pdfkit")
        print("   3. Check that the HTML file exists and is accessible")

if __name__ == "__main__":
    main()
