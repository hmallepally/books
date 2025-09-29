#!/usr/bin/env python3
"""
Generate final PDF with fixed page numbering and creative chapter end markers
"""

import os
import subprocess
import sys
from pathlib import Path
import re

def add_enhanced_css_and_chapter_markers():
    """Add enhanced CSS and chapter end markers"""
    css_addition = """
    <style>
    /* Enhanced page break handling */
    h1, h2, h3, h4, h5, h6 {
        page-break-after: avoid;
        break-after: avoid;
        orphans: 3;
        widows: 3;
    }
    
    /* Keep headings with their content */
    h1 + *, h2 + *, h3 + *, h4 + *, h5 + *, h6 + * {
        page-break-before: avoid;
        break-before: avoid;
    }
    
    /* Table page break handling */
    table {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* Large tables can break but keep header */
    table.comparison-table,
    table.implementation-table,
    table.roadmap-table {
        page-break-inside: auto;
        break-inside: auto;
    }
    
    table.comparison-table thead,
    table.implementation-table thead,
    table.roadmap-table thead {
        page-break-after: avoid;
        break-after: avoid;
        display: table-header-group;
    }
    
    /* Keep table rows together when possible */
    tr {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* List handling */
    ul, ol {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* Code blocks */
    pre, code {
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    /* Images */
    img {
        page-break-inside: avoid;
        break-inside: avoid;
        max-width: 100%;
        height: auto;
    }
    
    /* Paragraph spacing */
    p {
        orphans: 2;
        widows: 2;
    }
    
    /* Chapter breaks */
    .chapter {
        page-break-before: always;
        break-before: always;
    }
    
    /* Creative Chapter End Markers */
    .chapter-end-marker {
        text-align: center;
        margin: 40px 0 30px 0;
        padding: 20px 0;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    .chapter-end-line {
        width: 60%;
        height: 3px;
        background: linear-gradient(90deg, #2c3e50, #3498db, #2c3e50);
        margin: 0 auto 15px auto;
        border-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .chapter-end-dots {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
    }
    
    .chapter-end-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }
    
    .chapter-end-dot:nth-child(2) {
        background: #2c3e50;
        transform: scale(1.2);
    }
    
    .chapter-end-dot:nth-child(3) {
        background: #e74c3c;
    }
    
    .chapter-end-text {
        font-size: 10px;
        color: #7f8c8d;
        font-style: italic;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Alternative decorative patterns */
    .chapter-end-pattern {
        text-align: center;
        margin: 40px 0 30px 0;
        padding: 20px 0;
        page-break-inside: avoid;
        break-inside: avoid;
    }
    
    .chapter-end-pattern::before {
        content: "◆ ◇ ◆ ◇ ◆";
        font-size: 16px;
        color: #3498db;
        letter-spacing: 8px;
        display: block;
        margin-bottom: 10px;
    }
    
    .chapter-end-pattern::after {
        content: "◇ ◆ ◇ ◆ ◇";
        font-size: 16px;
        color: #2c3e50;
        letter-spacing: 8px;
        display: block;
        margin-top: 10px;
    }
    
    /* Better spacing for print */
    @media print {
        body {
            font-size: 11pt;
            line-height: 1.4;
        }
        
        h1 { font-size: 18pt; margin-top: 0; }
        h2 { font-size: 16pt; margin-top: 12pt; }
        h3 { font-size: 14pt; margin-top: 10pt; }
        h4 { font-size: 12pt; margin-top: 8pt; }
        
        table {
            font-size: 10pt;
        }
        
        .comparison-table th,
        .implementation-table th,
        .roadmap-table th {
            font-size: 10pt;
            padding: 4pt;
        }
        
        .comparison-table td,
        .implementation-table td,
        .roadmap-table td {
            font-size: 9pt;
            padding: 3pt;
        }
        
        /* Hide any duplicate page numbers */
        .duplicate-page-number {
            display: none !important;
        }
    }
    </style>
    """
    return css_addition

def add_chapter_end_markers(content):
    """Add creative chapter end markers before each new chapter"""
    
    # Find all chapter headings (h1 with chapter numbers)
    chapter_pattern = r'<h1 id="chapter-\d+.*?">(Chapter \d+:.*?)</h1>'
    
    def add_marker(match):
        chapter_title = match.group(1)
        
        # Create a creative chapter end marker
        marker_html = f'''
        <div class="chapter-end-marker">
            <div class="chapter-end-line"></div>
            <div class="chapter-end-dots">
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
            </div>
            <div class="chapter-end-text">End of {chapter_title}</div>
        </div>
        '''
        
        return marker_html + match.group(0)
    
    # Add markers before each chapter (except the first one)
    chapters = re.finditer(chapter_pattern, content, re.IGNORECASE)
    chapter_positions = []
    
    for match in chapters:
        chapter_positions.append(match.start())
    
    # Add markers before chapters (starting from the second chapter)
    offset = 0
    for i, pos in enumerate(chapter_positions[1:], 1):  # Skip first chapter
        marker_html = f'''
        <div class="chapter-end-marker">
            <div class="chapter-end-line"></div>
            <div class="chapter-end-dots">
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
                <div class="chapter-end-dot"></div>
            </div>
            <div class="chapter-end-text">Chapter Transition</div>
        </div>
        '''
        
        insert_pos = pos + offset
        content = content[:insert_pos] + marker_html + content[insert_pos:]
        offset += len(marker_html)
    
    return content

def enhance_html_for_pdf_v2(html_file):
    """Enhance HTML file with better page break handling and chapter markers"""
    
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add enhanced CSS for page breaks and chapter markers
    css_addition = add_enhanced_css_and_chapter_markers()
    
    # Insert CSS before closing head tag
    if '</head>' in content:
        content = content.replace('</head>', f'{css_addition}\n</head>')
    else:
        # If no head tag, add it
        content = f'<head>{css_addition}</head>\n{content}'
    
    # Add chapter end markers
    content = add_chapter_end_markers(content)
    
    # Create enhanced version
    enhanced_file = html_file.replace('.html', '_ENHANCED_V2.html')
    with open(enhanced_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return enhanced_file

def generate_pdf_with_playwright_v2(html_file, output_file):
    """Generate PDF using Playwright with fixed page numbering"""
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
                
                # Generate PDF with enhanced settings and SINGLE page numbering
                await page.pdf(
                    path=output_file,
                    format='A4',
                    margin={
                        'top': '0.75in',
                        'right': '0.75in',
                        'bottom': '0.75in',
                        'left': '0.75in'
                    },
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=True,
                    header_template='<div></div>',  # Empty header
                    footer_template='<div style="font-size: 8px; text-align: center; width: 100%; color: #666;"><span class="pageNumber"></span></div>'  # Single page number
                )
                
                await browser.close()
        
        print("Using Playwright for PDF generation with fixed page numbering...")
        asyncio.run(generate())
        
        print(f"✅ PDF generated successfully with Playwright: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ Playwright not available")
        return False
    except Exception as e:
        print(f"❌ Playwright error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Generating Enhanced PDF V2 with Fixed Page Numbers & Chapter Markers")
    print("=" * 80)
    
    # Define paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    output_file = "final_versions/with_toc/Operational_Excellence_with_AI_FINAL.pdf"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print(f"📄 Input HTML: {html_file}")
    print(f"📄 Output PDF: {output_file}")
    
    # Enhance HTML for better PDF generation
    print("\n🔧 Enhancing HTML with chapter markers and fixed page numbering...")
    enhanced_html = enhance_html_for_pdf_v2(html_file)
    print(f"✅ Enhanced HTML created: {enhanced_html}")
    
    # Generate PDF with Playwright
    print("\n📊 Generating PDF with Playwright...")
    success = generate_pdf_with_playwright_v2(enhanced_html, output_file)
    
    # Clean up enhanced HTML file
    if os.path.exists(enhanced_html):
        os.remove(enhanced_html)
        print(f"🧹 Cleaned up temporary file: {enhanced_html}")
    
    if success:
        print(f"\n🎉 PDF generation completed successfully!")
        print(f"📄 Final PDF: {output_file}")
        print(f"📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        print("\n📋 Key improvements in this PDF:")
        print("   ✅ Fixed duplicate page numbering")
        print("   ✅ Added creative chapter end markers")
        print("   ✅ Better page break handling for headings and tables")
        print("   ✅ Enhanced spacing and typography")
        print("   ✅ Professional decorative elements")
        
        print("\n🎨 Chapter End Markers include:")
        print("   • Gradient horizontal lines")
        print("   • Colored decorative dots")
        print("   • Elegant chapter transition text")
        print("   • Professional spacing")
        
        print("\n🔍 Please review the PDF and check:")
        print("   • Single page numbers (no duplicates)")
        print("   • Beautiful chapter end markers")
        print("   • Section headings stay with their content")
        print("   • Tables render properly without page breaks")
        
    else:
        print("\n❌ PDF generation failed!")
        print("\n💡 To install Playwright:")
        print("   • pip install playwright")
        print("   • playwright install")

if __name__ == "__main__":
    main()
