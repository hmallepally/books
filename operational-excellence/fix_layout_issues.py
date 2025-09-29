#!/usr/bin/env python3
"""
Fix Title Page, Content Repetition, and Chapter Breaks
"""

import re
import subprocess
import os

def fix_all_layout_issues():
    """Fix title page alignment, content repetition, and chapter breaks"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_UPDATED_PAGES.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing layout issues...")
    
    # Issue 1: Fix title page alignment
    print("1️⃣ Fixing title page alignment...")
    
    # Update CSS for title page alignment
    title_page_css = """
        .author-info {
            font-size: 1.2em;
            color: #2c3e50;
            margin-top: 1in;
            text-align: center;
        }

        .author-info p {
            margin: 0.1in 0;
            text-align: center;
        }

        .author-info strong {
            font-weight: bold;
        }
    """
    
    # Replace the existing author-info CSS
    html_content = re.sub(
        r'\.author-info \{[^}]*\}',
        title_page_css.strip(),
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 2: Remove content repetition after TOC
    print("2️⃣ Removing content repetition after TOC...")
    
    # Find and remove any duplicate title content that appears after TOC
    # Look for the pattern: TOC ends, then title content appears
    html_content = re.sub(
        r'</div>\s*<!-- Content will be inserted here from the markdown file -->\s*<div id="content-placeholder">\s*<!-- This will be populated with the complete content -->\s*</div>',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Remove any duplicate title content that might appear in the body
    html_content = re.sub(
        r'<h1[^>]*>Operational Excellence with AI</h1>\s*<h2[^>]*>A Comprehensive Guide[^<]*</h2>\s*<p><strong>Author:</strong>[^<]*</p>.*?(?=<h1[^>]*>Preface</h1>)',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 3: Fix chapter page breaks
    print("3️⃣ Fixing chapter page breaks...")
    
    # Ensure each chapter starts with a clear page break
    # Add clearpage before each chapter heading
    html_content = re.sub(
        r'<h1[^>]*>Chapter \d+:[^<]*</h1>',
        r'<div class="clearpage"></div>\n\g<0>',
        html_content
    )
    
    # Also ensure Preface starts with a clear page break
    html_content = re.sub(
        r'<h1[^>]*>Preface</h1>',
        r'<div class="clearpage"></div>\n\g<0>',
        html_content
    )
    
    # Remove any duplicate clearpage divs
    html_content = re.sub(
        r'<div class="clearpage"></div>\s*<div class="clearpage"></div>',
        '<div class="clearpage"></div>',
        html_content
    )
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_LAYOUT_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Layout fixed HTML created: {output_file}")
    return output_file

def generate_pdf_with_layout_fixes(html_file):
    """Generate PDF with layout fixes"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_LAYOUT_FIXED.pdf'
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
            print(f"🎉 Layout fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing title page alignment, content repetition, and chapter breaks...")
    
    # Fix layout issues
    html_file = fix_all_layout_issues()
    
    if html_file:
        # Generate PDF with layout fixes
        print("\n📚 Generating PDF with layout fixes...")
        generate_pdf_with_layout_fixes(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Fixed issues:")
        print("1. ✅ Title page: Both headings and values are now center-aligned")
        print("2. ✅ Removed content repetition after TOC")
        print("3. ✅ Fixed chapter page breaks - chapters now start on fresh pages")
        print("\n📖 Please review the PDF and provide the correct page numbers!")
    else:
        print("❌ Failed to fix layout issues")
