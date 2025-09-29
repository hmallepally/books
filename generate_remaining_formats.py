#!/usr/bin/env python3
"""
Generate remaining EPUB and PDF files for the complete file set
"""

import os
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {str(e)}")
        return False

def copy_css_file():
    """Copy CSS file to final_versions directory"""
    print("📋 Copying CSS file...")
    try:
        shutil.copy2("operational-excellence/epub-styles.css", "final_versions/epub-styles.css")
        print("✅ CSS file copied successfully")
        return True
    except Exception as e:
        print(f"❌ CSS file copy failed: {str(e)}")
        return False

def generate_epub_files():
    """Generate EPUB files for both versions"""
    print("📚 Generating EPUB files...")
    
    # EPUB for WITH TOC version
    cmd1 = 'pandoc "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html" -o "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.epub" --metadata title="Operational Excellence with AI" --metadata author="Hari Mallepally" --css "final_versions/epub-styles.css"'
    run_command(cmd1, "EPUB generation (WITH TOC)")
    
    # EPUB for WITHOUT TOC version
    cmd2 = 'pandoc "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html" -o "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.epub" --metadata title="Operational Excellence with AI" --metadata author="Hari Mallepally" --css "final_versions/epub-styles.css"'
    run_command(cmd2, "EPUB generation (WITHOUT TOC)")

def generate_pdf_files():
    """Generate PDF files for both versions using Playwright"""
    print("📄 Generating PDF files...")
    
    # PDF for WITH TOC version
    pdf_script1 = '''
import os
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    html_path = os.path.abspath("final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html")
    page.goto(f"file://{html_path}")
    page.pdf(
        path="final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.pdf",
        format="A4",
        margin={"top": "0.75in", "right": "0.75in", "bottom": "0.75in", "left": "0.75in"}
    )
    browser.close()
'''
    
    with open("temp_pdf1.py", "w") as f:
        f.write(pdf_script1)
    
    run_command("python temp_pdf1.py", "PDF generation (WITH TOC)")
    
    # PDF for WITHOUT TOC version
    pdf_script2 = '''
import os
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    html_path = os.path.abspath("final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html")
    page.goto(f"file://{html_path}")
    page.pdf(
        path="final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.pdf",
        format="A4",
        margin={"top": "0.75in", "right": "0.75in", "bottom": "0.75in", "left": "0.75in"}
    )
    browser.close()
'''
    
    with open("temp_pdf2.py", "w") as f:
        f.write(pdf_script2)
    
    run_command("python temp_pdf2.py", "PDF generation (WITHOUT TOC)")

def generate_individual_distribution_pdf():
    """Generate PDF with cover image for individual distribution"""
    print("🎨 Generating individual distribution PDF with cover...")
    
    # Create HTML with cover image
    cover_html = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Operational Excellence with AI</title>
    <style>
        body { margin: 0; padding: 0; }
        .cover-page { 
            width: 100vw; 
            height: 100vh; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            background: white;
        }
        .cover-image { 
            max-width: 100%; 
            max-height: 100%; 
            object-fit: contain; 
        }
    </style>
</head>
<body>
    <div class="cover-page">
        <img src="operational excellence with AI cover.png" alt="Operational Excellence with AI Cover" class="cover-image">
    </div>
</body>
</html>
'''
    
    # Write cover HTML
    with open("final_versions/individual_distribution/cover.html", "w", encoding="utf-8") as f:
        f.write(cover_html)
    
    # Generate PDF with cover
    pdf_script3 = '''
import os
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    html_path = os.path.abspath("final_versions/individual_distribution/cover.html")
    page.goto(f"file://{html_path}")
    page.pdf(
        path="final_versions/individual_distribution/Operational_Excellence_with_AI_INDIVIDUAL_DISTRIBUTION.pdf",
        format="A4",
        margin={"top": "0.5in", "right": "0.5in", "bottom": "0.5in", "left": "0.5in"}
    )
    browser.close()
'''
    
    with open("temp_pdf3.py", "w") as f:
        f.write(pdf_script3)
    
    run_command("python temp_pdf3.py", "Individual distribution PDF with cover")

def cleanup_temp_files():
    """Clean up temporary files"""
    print("🧹 Cleaning up temporary files...")
    temp_files = ["temp_pdf1.py", "temp_pdf2.py", "temp_pdf3.py"]
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"⚠️ Could not remove {temp_file}: {str(e)}")
    print("✅ Cleanup completed")

def main():
    """Main execution function"""
    print("🚀 Generating remaining EPUB and PDF files")
    print("=" * 50)
    
    # Step 1: Copy CSS file
    copy_css_file()
    
    # Step 2: Generate EPUB files
    generate_epub_files()
    
    # Step 3: Generate PDF files
    generate_pdf_files()
    
    # Step 4: Generate individual distribution PDF
    generate_individual_distribution_pdf()
    
    # Step 5: Cleanup
    cleanup_temp_files()
    
    print("=" * 50)
    print("🎉 REMAINING FORMATS GENERATED!")
    print("📁 Check the 'final_versions' directory for all files")

if __name__ == "__main__":
    main()
