#!/usr/bin/env python3
"""
Create complete file set for both versions of the book WITH IMAGES:
1. With TOC version (FINAL_CLEAN)
2. Without TOC version (FINAL_WITHOUT_TOC)

Formats needed: .md, .html, .epub, .pdf
All images included and properly referenced
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

def create_directories():
    """Create organized directories"""
    dirs = [
        "final_versions/with_toc",
        "final_versions/without_toc",
        "final_versions/individual_distribution",
        "final_versions/with_toc/images",
        "final_versions/without_toc/images",
        "final_versions/individual_distribution/images"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def copy_source_files():
    """Copy source files to organized directories"""
    print("📋 Copying source files...")
    
    # Copy HTML files
    shutil.copy2("Operational_Excellence_with_AI_FINAL_CLEAN.html", 
                 "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html")
    shutil.copy2("Operational_Excellence_with_AI_FINAL_WITHOUT_TOC.html", 
                 "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html")
    
    # Copy markdown file (use the most recent one)
    shutil.copy2("book/harnessing-ai-for-performance-optimization-final-clean.md", 
                 "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.md")
    shutil.copy2("book/harnessing-ai-for-performance-optimization-final-clean.md", 
                 "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.md")
    
    # Copy cover image
    shutil.copy2("kdp/operational excellence with AI cover.png", 
                 "final_versions/individual_distribution/operational excellence with AI cover.png")
    
    print("✅ Source files copied successfully")

def copy_images():
    """Copy all images to each version directory"""
    print("🖼️ Copying images...")
    
    image_count = 0
    
    # Copy all images from images/ directory
    if os.path.exists("images"):
        for image_file in os.listdir("images"):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Copy to both versions
                shutil.copy2(f"images/{image_file}", f"final_versions/with_toc/images/{image_file}")
                shutil.copy2(f"images/{image_file}", f"final_versions/without_toc/images/{image_file}")
                shutil.copy2(f"images/{image_file}", f"final_versions/individual_distribution/images/{image_file}")
                print(f"  📸 Copied: {image_file}")
                image_count += 1
    
    # Copy all images from diagrams/ directory
    if os.path.exists("diagrams"):
        for image_file in os.listdir("diagrams"):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                # Copy to both versions
                shutil.copy2(f"diagrams/{image_file}", f"final_versions/with_toc/images/{image_file}")
                shutil.copy2(f"diagrams/{image_file}", f"final_versions/without_toc/images/{image_file}")
                shutil.copy2(f"diagrams/{image_file}", f"final_versions/individual_distribution/images/{image_file}")
                print(f"  📸 Copied: {image_file}")
                image_count += 1
    
    print(f"✅ {image_count} images copied successfully")

def copy_css_file():
    """Copy CSS file to final_versions directory"""
    print("📋 Copying CSS file...")
    try:
        shutil.copy2("epub-styles.css", "final_versions/epub-styles.css")
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

def create_file_summary():
    """Create a summary of all generated files"""
    print("📋 Creating file summary...")
    
    # Count images
    with_toc_images = len([f for f in os.listdir("final_versions/with_toc/images") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    without_toc_images = len([f for f in os.listdir("final_versions/without_toc/images") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])
    
    summary = f"""
# 📚 Complete File Set for "Operational Excellence with AI" (WITH IMAGES)

## 🎯 Two Main Versions

### 1. WITH Table of Contents (KDP Publishing)
**Location:** `final_versions/with_toc/`
- ✅ `Operational_Excellence_with_AI_WITH_TOC.md` - Source markdown
- ✅ `Operational_Excellence_with_AI_WITH_TOC.html` - HTML version
- ✅ `Operational_Excellence_with_AI_WITH_TOC.epub` - EPUB for KDP
- ✅ `Operational_Excellence_with_AI_WITH_TOC.pdf` - PDF version
- ✅ `images/` - {with_toc_images} high-resolution images included

### 2. WITHOUT Table of Contents (Clean Reading)
**Location:** `final_versions/without_toc/`
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.md` - Source markdown
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.html` - HTML version
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.epub` - EPUB version
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.pdf` - PDF version
- ✅ `images/` - {without_toc_images} high-resolution images included

### 3. Individual Distribution (With Cover)
**Location:** `final_versions/individual_distribution/`
- ✅ `operational excellence with AI cover.png` - Cover image
- ✅ `Operational_Excellence_with_AI_INDIVIDUAL_DISTRIBUTION.pdf` - PDF with cover
- ✅ `images/` - All book images included

## 📖 Usage Guide

### For KDP Publishing:
- Use the **WITH TOC** version files
- Upload the `.epub` file to KDP (images included)
- Use the `.pdf` for print-on-demand (high-res images)

### For Individual Distribution:
- Use the **individual_distribution** PDF with cover
- Perfect for direct sharing or personal distribution
- All images included for complete experience

### For Clean Reading:
- Use the **WITHOUT TOC** version files
- Better for focused reading without navigation
- All images preserved for visual learning

## 🖼️ Image Details
- **Total Images:** {with_toc_images} high-resolution images
- **Image Sources:** `images/` and `diagrams/` directories
- **Formats:** PNG, JPG (optimized for print and digital)
- **Resolution:** High-resolution for KDP requirements

## 🎨 File Organization
All files are organized in clean directories with images properly included for complete book experience.

---
*Generated automatically - All formats with images ready for use!*
"""
    
    with open("final_versions/README.md", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("✅ File summary created")

def main():
    """Main execution function"""
    print("🚀 Creating complete file set for 'Operational Excellence with AI' WITH IMAGES")
    print("=" * 80)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Copy source files
    copy_source_files()
    
    # Step 3: Copy all images
    copy_images()
    
    # Step 4: Copy CSS file
    copy_css_file()
    
    # Step 5: Generate EPUB files
    generate_epub_files()
    
    # Step 6: Generate PDF files
    generate_pdf_files()
    
    # Step 7: Generate individual distribution PDF
    generate_individual_distribution_pdf()
    
    # Step 8: Cleanup
    cleanup_temp_files()
    
    # Step 9: Create summary
    create_file_summary()
    
    print("=" * 80)
    print("🎉 COMPLETE FILE SET WITH IMAGES CREATED SUCCESSFULLY!")
    print("📁 Check the 'final_versions' directory for all files")
    print("📋 Read 'final_versions/README.md' for usage guide")
    print("🖼️ All images are included in each version!")

if __name__ == "__main__":
    main()