#!/usr/bin/env python3
"""
Convert Complete Markdown to Beautiful HTML with All Chapters
"""

import re
import subprocess
import os

def convert_markdown_to_html():
    """Convert the complete markdown file to HTML with proper styling"""
    
    # Read the template HTML
    with open('Operational_Excellence_with_AI_FINAL_TEMPLATE.html', 'r', encoding='utf-8') as f:
        template = f.read()
    
    # Convert markdown to HTML using pandoc
    print("🔄 Converting markdown to HTML...")
    result = subprocess.run([
        'pandoc', 
        'book/Operational_Excellence_with_AI_KDP_READY.md',
        '-f', 'markdown',
        '-t', 'html',
        '--wrap=none'
    ], capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"❌ Pandoc error: {result.stderr}")
        return False
    
    html_content = result.stdout
    
    # Clean up the HTML content
    print("🧹 Cleaning up HTML content...")
    
    # Remove the title from the content (we have our own title page)
    html_content = re.sub(r'<h1[^>]*>.*?</h1>', '', html_content, flags=re.DOTALL)
    
    # Convert \clearpage to proper page breaks
    html_content = html_content.replace('\\clearpage', '<div class="clearpage"></div>')
    
    # Convert quote-chapter divs to proper styling
    html_content = re.sub(
        r'<div class="quote-chapter">\s*<blockquote>\s*<p><strong>"([^"]*)"</strong></p>\s*</blockquote>\s*</div>',
        r'<div class="chapter-quote">"\1"</div>',
        html_content,
        flags=re.DOTALL
    )
    
    # Convert regular blockquotes to chapter quotes
    html_content = re.sub(
        r'<blockquote>\s*<p><strong>"([^"]*)"</strong></p>\s*</blockquote>',
        r'<div class="chapter-quote">"\1"</div>',
        html_content,
        flags=re.DOTALL
    )
    
    # Add chapter overview sections
    html_content = re.sub(
        r'<h2>Chapter Overview</h2>\s*<p>([^<]*)</p>',
        r'<h2 class="chapter-overview">Chapter Overview</h2>\n<p>\1</p>',
        html_content,
        flags=re.DOTALL
    )
    
    # Wrap content in proper divs
    html_content = f'<div class="content">\n{html_content}\n</div>'
    
    # Insert content into template
    final_html = template.replace('<div id="content-placeholder">', html_content)
    
    # Write the final HTML file
    output_file = 'Operational_Excellence_with_AI_COMPLETE_FINAL.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"✅ Complete HTML file created: {output_file}")
    return output_file

def generate_pdf_from_html(html_file):
    """Generate PDF from the complete HTML file"""
    import asyncio
    from playwright.async_api import async_playwright
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_COMPLETE_FINAL.pdf'
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
            print(f"🎉 Complete PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Creating the complete book with all chapters and proper formatting...")
    
    # Convert markdown to HTML
    html_file = convert_markdown_to_html()
    
    if html_file:
        # Generate PDF
        print("📚 Generating PDF from complete HTML...")
        generate_pdf_from_html(html_file)
        print("✅ SUCCESS! Your complete book with all chapters is ready!")
    else:
        print("❌ Failed to create HTML file")
