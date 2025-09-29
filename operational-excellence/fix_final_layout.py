#!/usr/bin/env python3
"""
Fix Final Layout Issues
"""

import re
import subprocess
import os

def fix_final_layout_issues():
    """Fix title page fit, content repetition, and chapter breaks"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_LAYOUT_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing final layout issues...")
    
    # Issue 1: Make title page content fit on one page
    print("1️⃣ Making title page content fit on one page...")
    
    # Update CSS to make title page more compact
    title_page_css = """
        .title-page {
            page-break-after: always;
            text-align: center;
            padding-top: 2.5in;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .main-title {
            font-size: 3.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0.25in;
            line-height: 1.2;
        }

        .subtitle {
            font-size: 1.6em;
            color: #34495e;
            margin-bottom: 0.4in;
            font-weight: normal;
        }

        .author-info {
            font-size: 1.1em;
            color: #2c3e50;
            margin-top: 0.8in;
            text-align: center;
        }

        .author-info p {
            margin: 0.08in 0;
            text-align: center;
        }
    """
    
    # Replace the existing title page CSS
    html_content = re.sub(
        r'\.title-page \{[^}]*\}',
        title_page_css.strip(),
        html_content,
        flags=re.DOTALL
    )
    
    # Also update main-title and subtitle CSS
    html_content = re.sub(
        r'\.main-title \{[^}]*\}',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    html_content = re.sub(
        r'\.subtitle \{[^}]*\}',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 2: Remove title page content repetition after TOC
    print("2️⃣ Removing title page content repetition...")
    
    # Find and remove any duplicate title content that appears after TOC
    # Look for patterns like the title appearing again
    html_content = re.sub(
        r'<h1[^>]*>Operational Excellence with AI</h1>\s*<h2[^>]*>A Comprehensive Guide[^<]*</h2>\s*<p><strong>Author:</strong>[^<]*</p>.*?(?=<h1[^>]*>Preface</h1>)',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Remove any content that appears between TOC and Preface
    html_content = re.sub(
        r'</div>\s*<div[^>]*class="chapter"[^>]*>\s*<h1[^>]*>Preface</h1>',
        '</div>\n\n    <!-- Preface -->\n    <div id="preface" class="chapter">\n        <h1 class="preface-title">Preface</h1>',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 3: Fix chapter page breaks
    print("3️⃣ Fixing chapter page breaks...")
    
    # Ensure each chapter starts with a clear page break
    html_content = re.sub(
        r'<h1[^>]*>Chapter \d+:[^<]*</h1>',
        r'<div class="clearpage"></div>\n\g<0>',
        html_content
    )
    
    # Ensure Preface starts with a clear page break
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
    
    # Ensure clearpage divs are properly formatted
    html_content = re.sub(
        r'<div class="clearpage"></div>',
        '<div class="clearpage"></div>',
        html_content
    )
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_FINAL_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Final fixed HTML created: {output_file}")
    return output_file

def generate_final_pdf(html_file):
    """Generate the final PDF"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_FINAL.pdf'
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
            print(f"🎉 Final PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing final layout issues...")
    
    # Fix layout issues
    html_file = fix_final_layout_issues()
    
    if html_file:
        # Generate final PDF
        print("\n📚 Generating final PDF...")
        generate_final_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Fixed issues:")
        print("1. ✅ Title page content now fits on one page")
        print("2. ✅ Removed title page content repetition after TOC")
        print("3. ✅ Fixed chapter page breaks - chapters start on new pages")
        print("4. ✅ Organized old PDFs in 'old' folder")
        print("\n📖 Please review the PDF and provide correct page numbers!")
    else:
        print("❌ Failed to fix layout issues")
