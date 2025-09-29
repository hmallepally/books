#!/usr/bin/env python3
"""
Fix Title Repetition and Blank Pages
"""

import re
import subprocess
import os

def fix_title_repetition_and_blank_pages():
    """Fix title page repetition and blank pages before chapters"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_PROPER_STRUCTURE.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing title repetition and blank pages...")
    
    # Issue 1: Remove title page content that appears after TOC
    print("1️⃣ Removing title page content after TOC...")
    
    # Find and remove any content that appears between TOC and the first chapter/preface
    # Look for patterns like author info, version, etc. that shouldn't be there
    html_content = re.sub(
        r'</div>\s*<!-- Content -->\s*<div class="content">\s*<h2[^>]*>A Comprehensive Guide[^<]*</h2>\s*<p><div class="tech-label">Author:</div>\s*<p>Hari Mallepally</p><br />\s*<div class="tech-label">Version:</div>\s*<p>3\.0</p><br />\s*<div class="tech-label">Publication Date:</div>\s*<p>September 2025</p><br />\s*<div class="tech-label">Target Audience:</div>\s*<p>Business Leaders, Operations Professionals, MBA Students</p></p>',
        '</div>\n\n    <!-- Content -->',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 2: Remove blank pages before chapters
    print("2️⃣ Removing blank pages before chapters...")
    
    # Remove the clearpage divs that are causing blank pages
    # We want chapters to start on fresh pages but not have blank pages
    html_content = re.sub(
        r'<div class="clearpage"></div>\n<div id="preface" class="chapter">',
        '<div id="preface" class="chapter">',
        html_content
    )
    
    html_content = re.sub(
        r'<div class="clearpage"></div>\n<div id="chapter\d+" class="chapter">',
        '<div id="chapter\\1" class="chapter">',
        html_content
    )
    
    # Issue 3: Ensure proper page breaks without blank pages
    print("3️⃣ Setting proper page breaks...")
    
    # Update CSS to ensure chapters start on fresh pages without blank pages
    css_fix = """
        /* Chapter styling - no blank pages */
        .chapter {
            page-break-before: always;
        }
        
        /* Remove any extra spacing */
        .chapter-title {
            font-size: 2.8em;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-top: 1.5in;
            margin-bottom: 0.6in;
            line-height: 1.2;
        }
    """
    
    # Replace the existing chapter CSS
    html_content = re.sub(
        r'\.chapter \{[^}]*\}',
        css_fix.strip(),
        html_content,
        flags=re.DOTALL
    )
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_FINAL_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Final fixed HTML created: {output_file}")
    return output_file

def generate_final_fixed_pdf(html_file):
    """Generate PDF from final fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_FINAL_FIXED.pdf'
            await page.pdf(
                path=output_pdf,
                format='A4',
                margin={
                    'top': '0.75in',
                    'right': '0.75in',
                    'bottom': '0.75in',
                    'left': '0.75in'
                },
                print_background=True,
                prefer_css_page_size=True,
                display_header_footer=False,
                tagged=True
            )
            
            await browser.close()
            print(f"🎉 Final fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing title repetition and blank pages...")
    
    # Fix the issues
    html_file = fix_title_repetition_and_blank_pages()
    
    if html_file:
        # Generate final fixed PDF
        print("\n📚 Generating final fixed PDF...")
        generate_final_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Issues fixed:")
        print("1. ✅ Removed title page content after TOC")
        print("2. ✅ Removed blank pages before chapters")
        print("3. ✅ Chapters still start on fresh pages")
        print("\n📖 Please review the final fixed PDF!")
    else:
        print("❌ Failed to fix the issues")
