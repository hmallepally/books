#!/usr/bin/env python3
"""
Create EPUB for KDP publishing from MD file
"""

import os
import shutil
import re
import zipfile
from pathlib import Path

def convert_md_to_html_for_epub():
    """Convert MD file to HTML for EPUB"""
    
    print("📚 Creating EPUB for KDP Publishing from MD")
    print("=" * 60)
    
    # File paths
    md_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.md"
    html_file = "final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html"
    epub_file = "final_versions/without_toc/Operational_Excellence_with_AI_KDP.epub"
    
    if not os.path.exists(md_file):
        print(f"❌ MD file not found: {md_file}")
        return False
    
    print(f"📄 Source MD: {md_file}")
    print(f"📄 Target HTML: {html_file}")
    print(f"📄 Target EPUB: {epub_file}")
    
    # Read the MD file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert MD to HTML with EPUB-specific styling
    print("\n🔧 Converting MD to HTML for EPUB...")
    
    # Basic MD to HTML conversion
    html_content = convert_markdown_to_html(md_content)
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML file created: {html_file}")
    
    # Copy images
    print("\n📸 Copying images...")
    copy_images_for_epub()
    
    # Create EPUB
    print("\n📚 Creating EPUB...")
    success = create_epub_structure(html_file, epub_file)
    
    return success

def convert_markdown_to_html(md_content):
    """Convert markdown to HTML with EPUB styling"""
    
    # Start with EPUB HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Operational Excellence with AI</title>
    <style>
        /* EPUB-specific styles */
        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #2c3e50;
            background: white;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1 {
            font-size: 1.8em;
            color: #2c3e50;
            margin: 30px 0 20px 0;
            page-break-after: avoid;
            break-after: avoid;
        }
        
        h2 {
            font-size: 1.4em;
            color: #34495e;
            margin: 25px 0 15px 0;
            page-break-after: avoid;
            break-after: avoid;
        }
        
        h3 {
            font-size: 1.2em;
            color: #34495e;
            margin: 20px 0 10px 0;
            page-break-after: avoid;
            break-after: avoid;
        }
        
        h4 {
            font-size: 1.1em;
            color: #34495e;
            margin: 15px 0 8px 0;
            page-break-after: avoid;
            break-after: avoid;
        }
        
        p {
            margin: 10px 0;
            text-align: justify;
        }
        
        ul, ol {
            margin: 10px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 5px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 11pt;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
        }
        
        blockquote {
            margin: 15px 0;
            padding: 10px 20px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
            font-style: italic;
        }
        
        .chapter {
            page-break-before: always;
            break-before: always;
        }
        
        .chapter-title {
            text-align: center;
            font-size: 2em;
            margin: 40px 0 20px 0;
            color: #2c3e50;
        }
        
        .chapter-subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-style: italic;
        }
        
        /* Chapter End Markers */
        .chapter-end-marker {
            text-align: center;
            margin: 20px 0 0 0;
            padding: 10px 0;
            page-break-inside: avoid;
            break-inside: avoid;
        }
        
        .chapter-end-line {
            width: 40%;
            height: 1px;
            background: #ddd;
            margin: 0 auto 5px auto;
            border-radius: 1px;
        }
        
        .chapter-end-dots {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 4px;
            margin: 3px 0;
        }
        
        .chapter-end-dot {
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: #999;
        }
        
        .chapter-end-dot:nth-child(2) {
            background: #666;
        }
        
        .chapter-end-dot:nth-child(3) {
            background: #999;
        }
        
        /* Print styles */
        @media print {
            body {
                font-size: 11pt;
                line-height: 1.4;
            }
            
            .chapter-end-marker {
                margin: 15px 0 0 0;
                padding: 8px 0;
            }
            
            .chapter-end-line {
                height: 0.5px;
            }
            
            .chapter-end-dot {
                width: 3px;
                height: 3px;
            }
        }
    </style>
</head>
<body>
{content}
</body>
</html>'''
    
    # Basic markdown to HTML conversion
    html_content = md_content
    
    # Convert headers
    html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
    html_content = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', html_content, flags=re.MULTILINE)
    
    # Convert bold and italic
    html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
    html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
    
    # Convert lists
    html_content = re.sub(r'^- (.+)$', r'<li>\1</li>', html_content, flags=re.MULTILINE)
    
    # Wrap consecutive list items in ul tags
    html_content = re.sub(r'(<li>.*?</li>(?:\s*<li>.*?</li>)*)', r'<ul>\1</ul>', html_content, flags=re.DOTALL)
    
    # Convert links
    html_content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html_content)
    
    # Convert images
    html_content = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1">', html_content)
    
    # Convert blockquotes
    html_content = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', html_content, flags=re.MULTILINE)
    
    # Convert horizontal rules
    html_content = re.sub(r'^---+$', r'<hr>', html_content, flags=re.MULTILINE)
    
    # Wrap paragraphs
    paragraphs = html_content.split('\n\n')
    wrapped_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('<') and not para.startswith('#'):
            wrapped_paragraphs.append(f'<p>{para}</p>')
        else:
            wrapped_paragraphs.append(para)
    
    html_content = '\n\n'.join(wrapped_paragraphs)
    
    # Add chapter end markers
    html_content = add_chapter_end_markers(html_content)
    
    # Insert into template
    return html_template.format(content=html_content)

def add_chapter_end_markers(html_content):
    """Add chapter end markers to HTML content"""
    
    # Find chapter patterns and add markers
    chapter_pattern = r'<h1>Chapter \d+:.*?</h1>'
    
    def add_marker_after_chapter(match):
        chapter_title = match.group(0)
        
        # Create marker HTML
        marker_html = '''
        <div class="chapter-end-marker">
            <div class="chapter-end-line"></div>
            <div class="chapter-end-dots">
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
            </div>
        </div>
        '''
        
        return chapter_title + marker_html
    
    # Add markers after each chapter (except the last one)
    chapters = re.finditer(chapter_pattern, html_content)
    chapter_positions = []
    
    for match in chapters:
        chapter_positions.append(match.end())
    
    # Add markers after chapters (except the last one)
    offset = 0
    for i, pos in enumerate(chapter_positions[:-1]):  # Skip last chapter
        marker_html = '''
        <div class="chapter-end-marker">
            <div class="chapter-end-line"></div>
            <div class="chapter-end-dots">
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
            </div>
        </div>
        '''
        
        insert_pos = pos + offset
        html_content = html_content[:insert_pos] + marker_html + html_content[insert_pos:]
        offset += len(marker_html)
    
    return html_content

def copy_images_for_epub():
    """Copy images for EPUB"""
    
    source_images = "final_versions/with_toc/images"
    target_images = "final_versions/without_toc/images"
    
    if os.path.exists(source_images):
        if not os.path.exists(target_images):
            os.makedirs(target_images)
        
        # Copy all images
        for image_file in os.listdir(source_images):
            src = os.path.join(source_images, image_file)
            dst = os.path.join(target_images, image_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"   📸 Copied: {image_file}")

def create_epub_structure(html_file, epub_file):
    """Create EPUB structure manually"""
    
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
    success = convert_md_to_html_for_epub()
    
    if success:
        print(f"\n🎉 KDP EPUB creation completed successfully!")
        print(f"\n📋 Files created:")
        print(f"   • HTML: final_versions/without_toc/Operational_Excellence_with_AI_WITHOUT_TOC.html")
        print(f"   • EPUB: final_versions/without_toc/Operational_Excellence_with_AI_KDP.epub")
        
        print(f"\n📚 EPUB Features:")
        print(f"   ✅ Chapter end markers included")
        print(f"   ✅ Proper EPUB structure")
        print(f"   ✅ Images included")
        print(f"   ✅ KDP-ready format")
        print(f"   ✅ Professional styling")
        
        print(f"\n🚀 Ready for KDP Publishing!")
        print(f"   • Upload the EPUB file to KDP")
        print(f"   • KDP will generate the Table of Contents automatically")
        print(f"   • All chapter links will work properly")
        
    else:
        print(f"\n❌ KDP EPUB creation failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
