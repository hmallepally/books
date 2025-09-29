#!/usr/bin/env python3
"""
Generate PDF with correct KDP trim size (8.5" x 11")
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_kdp_pdf():
    """Generate PDF with correct KDP trim size"""
    
    print("📚 Generating PDF with Correct KDP Trim Size")
    print("=" * 60)
    
    # File paths
    md_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.md"
    html_file = "final_versions/without_toc/Operational_Excellence_with_AI_KDP.html"
    pdf_file = "final_versions/without_toc/Operational_Excellence_with_AI_KDP.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ MD file not found: {md_file}")
        return False
    
    print(f"📄 Source MD: {md_file}")
    print(f"📄 Target HTML: {html_file}")
    print(f"📄 Target PDF: {pdf_file}")
    
    # Read the MD file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Create HTML with KDP-specific styling
    print("\n🔧 Creating HTML with KDP trim size...")
    html_content = create_kdp_html(md_content)
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML file created: {html_file}")
    
    # Generate PDF with correct trim size
    print("\n📊 Generating PDF with KDP trim size (8.5\" x 11\")...")
    success = generate_pdf_with_playwright_kdp(html_file, pdf_file)
    
    return success

def create_kdp_html(md_content):
    """Create HTML with KDP-specific styling"""
    
    # KDP HTML template with correct page size
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Operational Excellence with AI</title>
    <style>
        /* KDP-specific page setup */
        @page {
            size: 8.5in 11in; /* US Letter size for KDP */
            margin: 0.75in;
        }
        
        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #2c3e50;
            background: white;
            margin: 0;
            padding: 0;
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
        
        /* Print styles for KDP */
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
    
    # Convert markdown to HTML
    html_content = convert_markdown_to_html(md_content)
    
    # Insert into template
    return html_template.replace('{content}', html_content)

def convert_markdown_to_html(md_content):
    """Convert markdown to HTML"""
    
    import re
    
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
    
    return html_content

def add_chapter_end_markers(html_content):
    """Add chapter end markers to HTML content"""
    
    import re
    
    # Find chapter patterns and add markers
    chapter_pattern = r'<h1>Chapter \d+:.*?</h1>'
    
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

def generate_pdf_with_playwright_kdp(html_file, pdf_file):
    """Generate PDF using Playwright with KDP trim size"""
    try:
        import asyncio
        from playwright.async_api import async_playwright
        
        async def generate():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Load HTML file
                await page.goto(f"file://{os.path.abspath(html_file)}")
                
                # Wait for content to load
                await page.wait_for_load_state('networkidle')
                
                # Generate PDF with KDP trim size (8.5" x 11")
                await page.pdf(
                    path=pdf_file,
                    format='Letter',  # US Letter size (8.5" x 11")
                    margin={
                        'top': '0.75in',
                        'right': '0.75in',
                        'bottom': '0.75in',
                        'left': '0.75in'
                    },
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=False
                )
                
                await browser.close()
        
        print("Using Playwright for PDF generation with KDP trim size...")
        asyncio.run(generate())
        
        print(f"✅ PDF generated successfully: {pdf_file}")
        print(f"File size: {os.path.getsize(pdf_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ Playwright not available")
        return False
    except Exception as e:
        print(f"❌ Playwright error: {e}")
        return False

def main():
    """Main function"""
    success = generate_kdp_pdf()
    
    if success:
        print(f"\n🎉 KDP PDF generation completed successfully!")
        print(f"\n📋 Files created:")
        print(f"   • HTML: final_versions/without_toc/Operational_Excellence_with_AI_KDP.html")
        print(f"   • PDF: final_versions/without_toc/Operational_Excellence_with_AI_KDP.pdf")
        
        print(f"\n📚 PDF Features:")
        print(f"   ✅ Correct KDP trim size: 8.5\" x 11\"")
        print(f"   ✅ Chapter end markers included")
        print(f"   ✅ Professional styling")
        print(f"   ✅ KDP-compatible format")
        
        print(f"\n🚀 Ready for KDP Publishing!")
        print(f"   • Upload the PDF file to KDP")
        print(f"   • Trim size matches KDP requirements")
        print(f"   • Professional layout and formatting")
        
    else:
        print(f"\n❌ KDP PDF generation failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
