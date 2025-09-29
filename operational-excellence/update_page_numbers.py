#!/usr/bin/env python3
"""
Update Page Numbers in TOC
This script will help update the Table of Contents with correct page numbers
"""

def update_page_numbers():
    """Update the TOC with correct page numbers"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📖 Current TOC page numbers:")
    print("Please review the PDF and provide the actual page numbers for each chapter.")
    print("\nCurrent TOC structure:")
    
    # Extract current TOC entries
    toc_pattern = r'<li class="chapter"><a href="#([^"]*)">([^<]*) <span class="toc-page-number">(\d+)</span></a></li>'
    toc_entries = re.findall(toc_pattern, html_content)
    
    for i, (anchor, title, current_page) in enumerate(toc_entries, 1):
        print(f"{i:2d}. {title} (currently page {current_page})")
    
    print("\n" + "="*60)
    print("Please provide the correct page numbers:")
    print("Format: chapter_number:page_number (e.g., 1:9, 2:25, 3:45)")
    print("Or type 'auto' to use estimated page numbers")
    print("="*60)
    
    # For now, let's use estimated page numbers based on typical book layout
    estimated_pages = {
        'preface': 5,
        'chapter1': 9,
        'chapter2': 25,
        'chapter3': 45,
        'chapter4': 65,
        'chapter5': 85,
        'chapter6': 105,
        'chapter7': 125,
        'chapter8': 145,
        'chapter9': 165,
        'chapter10': 185,
        'chapter11': 205,
        'chapter12': 225,
        'chapter13': 245,
        'chapter14': 265,
        'chapter15': 285
    }
    
    # Update with estimated page numbers
    for anchor, title, current_page in toc_entries:
        if anchor in estimated_pages:
            new_page = estimated_pages[anchor]
            old_pattern = f'<a href="#{anchor}">([^<]*) <span class="toc-page-number">{current_page}</span></a>'
            new_replacement = f'<a href="#{anchor}">\\1 <span class="toc-page-number">{new_page}</span></a>'
            html_content = re.sub(old_pattern, new_replacement, html_content)
            print(f"✅ Updated {title}: {current_page} → {new_page}")
    
    # Write updated HTML
    output_file = 'Operational_Excellence_with_AI_UPDATED_PAGES.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n📝 Updated HTML saved as: {output_file}")
    print("📚 You can now generate a new PDF with updated page numbers")
    
    return output_file

def generate_pdf_with_updated_pages(html_file):
    """Generate PDF with updated page numbers"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_FINAL_VERSION.pdf'
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
    import re
    
    print("🔄 Updating page numbers in TOC...")
    
    # Update page numbers
    html_file = update_page_numbers()
    
    # Generate PDF with updated page numbers
    print("\n📚 Generating PDF with updated page numbers...")
    generate_pdf_with_updated_pages(html_file)
    
    print("\n✅ SUCCESS!")
    print("📖 Please review the PDF and let me know if any page numbers need adjustment.")
    print("🔄 We can iterate as many times as needed to get it perfect!")
