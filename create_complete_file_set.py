#!/usr/bin/env python3
"""
Create complete file set for both versions of the book:
1. With TOC version (FINAL_CLEAN)
2. Without TOC version (FINAL_WITHOUT_TOC)

Formats needed: .md, .html, .epub, .pdf
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
        "final_versions/individual_distribution"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def copy_source_files():
    """Copy source files to organized directories"""
    print("📋 Copying source files...")
    
    # Copy HTML files
    shutil.copy2("operational-excellence/Operational_Excellence_with_AI_FINAL_CLEAN.html", 
                 "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html")
    shutil.copy2("operational-excellence/Operational_Excellence_with_AI_FINAL_WITHOUT_TOC.html", 
                 "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html")
    
    # Copy markdown file (use the most recent one)
    shutil.copy2("operational-excellence/book/harnessing-ai-for-performance-optimization-final-clean.md", 
                 "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.md")
    shutil.copy2("operational-excellence/book/harnessing-ai-for-performance-optimization-final-clean.md", 
                 "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.md")
    
    # Copy cover image
    shutil.copy2("operational-excellence/kdp/operational excellence with AI cover.png", 
                 "final_versions/individual_distribution/operational excellence with AI cover.png")
    
    print("✅ Source files copied successfully")

def generate_epub_files():
    """Generate EPUB files for both versions"""
    print("📚 Generating EPUB files...")
    
    # EPUB for WITH TOC version
    cmd1 = 'pandoc "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html" -o "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.epub" --metadata title="Operational Excellence with AI" --metadata author="Hari Mallepally" --css epub-styles.css'
    run_command(cmd1, "EPUB generation (WITH TOC)")
    
    # EPUB for WITHOUT TOC version
    cmd2 = 'pandoc "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html" -o "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.epub" --metadata title="Operational Excellence with AI" --metadata author="Hari Mallepally" --css epub-styles.css'
    run_command(cmd2, "EPUB generation (WITHOUT TOC)")

def generate_pdf_files():
    """Generate PDF files for both versions"""
    print("📄 Generating PDF files...")
    
    # PDF for WITH TOC version
    cmd1 = 'python -c "from playwright.sync_api import sync_playwright; import os; playwright = sync_playwright().start(); browser = playwright.chromium.launch(); page = browser.new_page(); page.goto(f\'file://{os.path.abspath(\"final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html\")}\'); page.pdf(path=\'final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.pdf\', format=\'A4\', margin={\'top\': \'0.75in\', \'right\': \'0.75in\', \'bottom\': \'0.75in\', \'left\': \'0.75in\'}); browser.close(); playwright.stop()"'
    run_command(cmd1, "PDF generation (WITH TOC)")
    
    # PDF for WITHOUT TOC version
    cmd2 = 'python -c "from playwright.sync_api import sync_playwright; import os; playwright = sync_playwright().start(); browser = playwright.chromium.launch(); page = browser.new_page(); page.goto(f\'file://{os.path.abspath(\"final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html\')}\'); page.pdf(path=\'final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.pdf\', format=\'A4\', margin={\'top\': \0.75in\', \'right\': \'0.75in\', \'bottom\': \'0.75in\', \'left\': \'0.75in\'}); browser.close(); playwright.stop()"'
    run_command(cmd2, "PDF generation (WITHOUT TOC)")

def generate_individual_distribution_pdf():
    """Generate PDF with cover image for individual distribution"""
    print("🎨 Generating individual distribution PDF with cover...")
    
    # Create HTML with cover image
    cover_html = """
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
    """
    
    # Write cover HTML
    with open("final_versions/individual_distribution/cover.html", "w", encoding="utf-8") as f:
        f.write(cover_html)
    
    # Generate PDF with cover
    cmd = 'python -c "from playwright.sync_api import sync_playwright; import os; playwright = sync_playwright().start(); browser = playwright.chromium.launch(); page = browser.new_page(); page.goto(f\'file://{os.path.abspath(\"final_versions/individual_distribution/cover.html\")}\'); page.pdf(path=\'final_versions/individual_distribution/Operational_Excellence_with_AI_INDIVIDUAL_DISTRIBUTION.pdf\', format=\'A4\', margin={\'top\': \'0.5in\', \'right\': \'0.5in\', \'bottom\': \'0.5in\', \'left\': \'0.5in\'}); browser.close(); playwright.stop()"'
    run_command(cmd, "Individual distribution PDF with cover")

def create_file_summary():
    """Create a summary of all generated files"""
    print("📋 Creating file summary...")
    
    summary = """
# 📚 Complete File Set for "Operational Excellence with AI"

## 🎯 Two Main Versions

### 1. WITH Table of Contents (KDP Publishing)
**Location:** `final_versions/with_toc/`
- ✅ `Operational_Excellence_with_AI_WITH_TOC.md` - Source markdown
- ✅ `Operational_Excellence_with_AI_WITH_TOC.html` - HTML version
- ✅ `Operational_Excellence_with_AI_WITH_TOC.epub` - EPUB for KDP
- ✅ `Operational_Excellence_with_AI_WITH_TOC.pdf` - PDF version

### 2. WITHOUT Table of Contents (Clean Reading)
**Location:** `final_versions/without_toc/`
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.md` - Source markdown
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.html` - HTML version
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.epub` - EPUB version
- ✅ `Operational_Excellence_with_AI_WITHOUT_TOC.pdf` - PDF version

### 3. Individual Distribution (With Cover)
**Location:** `final_versions/individual_distribution/`
- ✅ `operational excellence with AI cover.png` - Cover image
- ✅ `Operational_Excellence_with_AI_INDIVIDUAL_DISTRIBUTION.pdf` - PDF with cover

## 📖 Usage Guide

### For KDP Publishing:
- Use the **WITH TOC** version files
- Upload the `.epub` file to KDP
- Use the `.pdf` for print-on-demand

### For Individual Distribution:
- Use the **individual_distribution** PDF with cover
- Perfect for direct sharing or personal distribution

### For Clean Reading:
- Use the **WITHOUT TOC** version files
- Better for focused reading without navigation

## 🎨 File Organization
All files are organized in clean directories for easy management and distribution.

---
*Generated automatically - All formats ready for use!*
"""
    
    with open("final_versions/README.md", "w", encoding="utf-8") as f:
        f.write(summary)
    
    print("✅ File summary created")

def main():
    """Main execution function"""
    print("🚀 Creating complete file set for 'Operational Excellence with AI'")
    print("=" * 70)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Copy source files
    copy_source_files()
    
    # Step 3: Generate EPUB files
    generate_epub_files()
    
    # Step 4: Generate PDF files
    generate_pdf_files()
    
    # Step 5: Generate individual distribution PDF
    generate_individual_distribution_pdf()
    
    # Step 6: Create summary
    create_file_summary()
    
    print("=" * 70)
    print("🎉 COMPLETE FILE SET CREATED SUCCESSFULLY!")
    print("📁 Check the 'final_versions' directory for all files")
    print("📋 Read 'final_versions/README.md' for usage guide")

if __name__ == "__main__":
    main()
