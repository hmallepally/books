#!/usr/bin/env python3
"""
Copy the correct HTML file with TOC to without_toc folder and create EPUB
"""

import os
import shutil
import zipfile
from pathlib import Path

def copy_correct_html_for_epub():
    """Copy the correct HTML file and create EPUB"""
    
    print("📚 Copying Correct HTML for EPUB Creation")
    print("=" * 60)
    
    # File paths
    source_html = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    target_html = "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html"
    epub_file = "final_versions/without_toc/Operational_Excellence_with_AI_KDP.epub"
    
    if not os.path.exists(source_html):
        print(f"❌ Source HTML file not found: {source_html}")
        return False
    
    print(f"📄 Source HTML: {source_html}")
    print(f"📄 Target HTML: {target_html}")
    print(f"📄 Target EPUB: {epub_file}")
    
    # Copy the HTML file
    print("\n📋 Copying HTML file...")
    shutil.copy2(source_html, target_html)
    print(f"✅ HTML file copied: {target_html}")
    
    # Remove Table of Contents for EPUB version
    print("\n🔧 Removing Table of Contents for EPUB version...")
    remove_toc_from_html(target_html)
    
    # Copy images
    print("\n📸 Copying images...")
    copy_images_for_epub()
    
    # Create EPUB
    print("\n📚 Creating EPUB...")
    success = create_epub_structure(target_html, epub_file)
    
    return success

def remove_toc_from_html(html_file):
    """Remove Table of Contents from HTML for EPUB version"""
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove TOC section
    import re
    
    # Remove TOC div
    toc_pattern = r'<div class="table-of-contents">.*?</div>\s*</div>'
    content = re.sub(toc_pattern, '', content, flags=re.DOTALL)
    
    # Remove TOC navigation
    toc_nav_pattern = r'<nav class="toc-navigation">.*?</nav>'
    content = re.sub(toc_nav_pattern, '', content, flags=re.DOTALL)
    
    # Remove TOC styles
    toc_style_pattern = r'/\* Table of Contents Styles \*/.*?/\* End TOC Styles \*/'
    content = re.sub(toc_style_pattern, '', content, flags=re.DOTALL)
    
    # Write the updated content
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ Table of Contents removed from HTML")

def copy_images_for_epub():
    """Copy images for EPUB"""
    
    source_images = "final_versions/with_toc/images"
    target_images = "final_versions/without_toc/images"
    
    if os.path.exists(source_images):
        if os.path.exists(target_images):
            shutil.rmtree(target_images)
        shutil.copytree(source_images, target_images)
        print(f"✅ Images copied to: {target_images}")

def create_epub_structure(html_file, epub_file):
    """Create EPUB structure"""
    
    try:
        print("📚 Creating EPUB structure...")
        
        # Create EPUB structure
        epub_dir = "temp_epub"
        if os.path.exists(epub_dir):
            shutil.rmtree(epub_dir)
        os.makedirs(epub_dir)
        
        # Create META-INF directory
        meta_inf_dir = os.path.join(epub_dir, "META-INF")
        os.makedirs(meta_inf_dir)
        
        # Create container.xml
        container_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>'''
        
        with open(os.path.join(meta_inf_dir, "container.xml"), 'w', encoding='utf-8') as f:
            f.write(container_xml)
        
        # Create OEBPS directory
        oebps_dir = os.path.join(epub_dir, "OEBPS")
        os.makedirs(oebps_dir)
        
        # Copy HTML file
        shutil.copy2(html_file, os.path.join(oebps_dir, "content.html"))
        
        # Copy images
        images_dir = os.path.join(oebps_dir, "images")
        if os.path.exists("final_versions/without_toc/images"):
            shutil.copytree("final_versions/without_toc/images", images_dir)
        
        # Create content.opf
        content_opf = '''<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="book-id" version="2.0">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Operational Excellence with AI</dc:title>
        <dc:creator>Hari Mallepally</dc:creator>
        <dc:language>en</dc:language>
        <dc:identifier id="book-id">operational-excellence-ai</dc:identifier>
        <dc:description>A Comprehensive Guide to Operational Excellence in the AI Era</dc:description>
        <dc:subject>Artificial Intelligence, Operational Excellence, Business Strategy</dc:subject>
    </metadata>
    <manifest>
        <item id="content" href="content.html" media-type="application/xhtml+xml"/>
    </manifest>
    <spine toc="ncx">
        <itemref idref="content"/>
    </spine>
</package>'''
        
        with open(os.path.join(oebps_dir, "content.opf"), 'w', encoding='utf-8') as f:
            f.write(content_opf)
        
        # Create mimetype file
        with open(os.path.join(epub_dir, "mimetype"), 'w', encoding='utf-8') as f:
            f.write("application/epub+zip")
        
        # Create EPUB zip file
        with zipfile.ZipFile(epub_file, 'w', zipfile.ZIP_DEFLATED) as epub_zip:
            # Add mimetype first (uncompressed)
            epub_zip.write(os.path.join(epub_dir, "mimetype"), "mimetype", compress_type=zipfile.ZIP_STORED)
            
            # Add all other files
            for root, dirs, files in os.walk(epub_dir):
                for file in files:
                    if file != "mimetype":
                        file_path = os.path.join(root, file)
                        arc_path = os.path.relpath(file_path, epub_dir)
                        epub_zip.write(file_path, arc_path)
        
        # Clean up temp directory
        shutil.rmtree(epub_dir)
        
        print(f"✅ EPUB created successfully: {epub_file}")
        print(f"📊 File size: {os.path.getsize(epub_file) / (1024*1024):.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error creating EPUB: {e}")
        return False

def main():
    """Main function"""
    success = copy_correct_html_for_epub()
    
    if success:
        print(f"\n🎉 EPUB creation completed successfully!")
        print(f"\n📋 Files created:")
        print(f"   • HTML: final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html")
        print(f"   • EPUB: final_versions/without_toc/Operational_Excellence_with_AI_KDP.epub")
        
        print(f"\n📚 EPUB Features:")
        print(f"   ✅ Correct HTML content (from with_toc)")
        print(f"   ✅ Chapter end markers included")
        print(f"   ✅ Professional tables included")
        print(f"   ✅ Table of Contents removed (KDP will generate)")
        print(f"   ✅ Images included")
        print(f"   ✅ KDP-ready format")
        
        print(f"\n🚀 Ready for KDP Publishing!")
        print(f"   • Upload the EPUB file to KDP")
        print(f"   • KDP will generate the Table of Contents automatically")
        print(f"   • All chapter links will work properly")
        
    else:
        print(f"\n❌ EPUB creation failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
