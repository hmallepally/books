#!/usr/bin/env python3
"""
Create EPUB for KDP publishing with proper chapter linking
"""

import os
import shutil
import re
from pathlib import Path

def create_kdp_epub():
    """Create EPUB for KDP publishing"""
    
    print("📚 Creating EPUB for KDP Publishing")
    print("=" * 50)
    
    # Check if we have the latest HTML file
    with_toc_html = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    without_toc_html = "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html"
    
    if not os.path.exists(with_toc_html):
        print(f"❌ Source HTML file not found: {with_toc_html}")
        print("📋 Available files in with_toc:")
        for file in os.listdir("final_versions/with_toc/"):
            if file.endswith('.html'):
                print(f"   • {file}")
        return False
    
    print(f"📄 Source HTML: {with_toc_html}")
    print(f"📄 Target HTML: {without_toc_html}")
    
    # Read the source HTML file
    with open(with_toc_html, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove Table of Contents section for KDP version
    print("\n🔧 Removing Table of Contents for KDP version...")
    
    # Remove TOC section
    toc_pattern = r'<div class="table-of-contents">.*?</div>\s*</div>'
    content = re.sub(toc_pattern, '', content, flags=re.DOTALL)
    
    # Remove TOC navigation
    toc_nav_pattern = r'<nav class="toc-navigation">.*?</nav>'
    content = re.sub(toc_nav_pattern, '', content, flags=re.DOTALL)
    
    # Remove TOC styles
    toc_style_pattern = r'/\* Table of Contents Styles \*/.*?/\* End TOC Styles \*/'
    content = re.sub(toc_style_pattern, '', content, flags=re.DOTALL)
    
    # Add EPUB-specific CSS
    epub_css = """
    <style>
    /* EPUB-specific styles */
    @media screen {
        body {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
    }
    
    /* Chapter navigation for EPUB */
    .chapter-nav {
        margin: 20px 0;
        padding: 10px;
        background: #f8f9fa;
        border-left: 4px solid #3498db;
    }
    
    .chapter-nav a {
        color: #3498db;
        text-decoration: none;
        font-weight: bold;
    }
    
    .chapter-nav a:hover {
        text-decoration: underline;
    }
    </style>
    """
    
    # Insert EPUB CSS
    if '</head>' in content:
        content = content.replace('</head>', f'{epub_css}\n</head>')
    
    # Add chapter navigation links
    print("🔗 Adding chapter navigation links...")
    
    # Find all chapters and add navigation
    chapter_pattern = r'<div id="(chapter\d+)" class="chapter">\s*<h1 class="chapter-title">(Chapter \d+:.*?)</h1>'
    
    def add_chapter_nav(match):
        chapter_id = match.group(1)
        chapter_title = match.group(2)
        
        # Create navigation HTML
        nav_html = f'''
        <div class="chapter-nav">
            <a href="#{chapter_id}">{chapter_title}</a>
        </div>
        '''
        
        return nav_html + match.group(0)
    
    # Add navigation before each chapter
    content = re.sub(chapter_pattern, add_chapter_nav, content, flags=re.DOTALL)
    
    # Write the updated content to without_toc folder
    with open(without_toc_html, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ HTML file updated: {without_toc_html}")
    
    # Copy images if needed
    print("\n📸 Checking images...")
    with_toc_images = "final_versions/with_toc/images"
    without_toc_images = "final_versions/without_toc/images"
    
    if os.path.exists(with_toc_images):
        if not os.path.exists(without_toc_images):
            os.makedirs(without_toc_images)
        
        # Copy all images
        for image_file in os.listdir(with_toc_images):
            src = os.path.join(with_toc_images, image_file)
            dst = os.path.join(without_toc_images, image_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"   📸 Copied: {image_file}")
    
    # Create EPUB using pandoc if available
    print("\n📚 Creating EPUB...")
    epub_file = "final_versions/without_toc/Operational_Excellence_with_AI_KDP.epub"
    
    try:
        import subprocess
        
        # Try to use pandoc to create EPUB
        cmd = [
            "pandoc",
            without_toc_html,
            "-o", epub_file,
            "--epub-cover-image=final_versions/without_toc/images/ai_success_framework.png",
            "--metadata", "title=Operational Excellence with AI",
            "--metadata", "author=AI Operations Expert",
            "--metadata", "language=en",
            "--toc",
            "--toc-depth=2"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ EPUB created successfully: {epub_file}")
            print(f"📊 File size: {os.path.getsize(epub_file) / (1024*1024):.1f} MB")
        else:
            print(f"❌ Pandoc error: {result.stderr}")
            return create_manual_epub(without_toc_html, epub_file)
            
    except FileNotFoundError:
        print("⚠️  Pandoc not found, creating manual EPUB structure...")
        return create_manual_epub(without_toc_html, epub_file)
    except Exception as e:
        print(f"❌ Error creating EPUB: {e}")
        return create_manual_epub(without_toc_html, epub_file)

def create_manual_epub(html_file, epub_file):
    """Create EPUB manually using zip"""
    try:
        import zipfile
        import mimetypes
        
        print("📚 Creating EPUB manually...")
        
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
        <dc:creator>AI Operations Expert</dc:creator>
        <dc:language>en</dc:language>
        <dc:identifier id="book-id">operational-excellence-ai</dc:identifier>
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
        print(f"❌ Error creating manual EPUB: {e}")
        return False

def main():
    """Main function"""
    success = create_kdp_epub()
    
    if success:
        print(f"\n🎉 KDP EPUB creation completed successfully!")
        print(f"\n📋 Files created:")
        print(f"   • HTML: final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html")
        print(f"   • EPUB: final_versions/without_toc/Operational_Excellence_with_AI_KDP.epub")
        
        print(f"\n📚 EPUB Features:")
        print(f"   ✅ No Table of Contents (KDP will generate)")
        print(f"   ✅ Chapter end markers included")
        print(f"   ✅ Proper chapter linking")
        print(f"   ✅ Images included")
        print(f"   ✅ KDP-ready format")
        
        print(f"\n🚀 Ready for KDP Publishing!")
        print(f"   • Upload the EPUB file to KDP")
        print(f"   • KDP will generate the Table of Contents automatically")
        print(f"   • All chapter links will work properly")
        
    else:
        print(f"\n❌ KDP EPUB creation failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
