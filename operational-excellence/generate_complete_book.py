#!/usr/bin/env python3
"""
Generate the Complete PDF Book with All Chapters
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def generate_complete_pdf(html_file, output_pdf):
    """Generate the complete PDF with all chapters and proper formatting"""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load HTML file
        await page.goto(f"file://{os.path.abspath(html_file)}")
        
        # Wait for content to load
        await page.wait_for_load_state('networkidle')
        
        # Generate PDF with complete settings
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
            tagged=True  # For accessibility
        )
        
        await browser.close()
        print(f"🎉 COMPLETE PDF generated successfully: {output_pdf}")
        print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
        print(f"📚 This is your complete book with all chapters!")

if __name__ == "__main__":
    html_file = "Operational_Excellence_with_AI_COMPLETE.html"
    output_pdf = "Operational_Excellence_with_AI_COMPLETE_BOOK.pdf"
    
    if os.path.exists(html_file):
        print("🚀 Generating the complete PDF book with all chapters...")
        asyncio.run(generate_complete_pdf(html_file, output_pdf))
        print("✅ SUCCESS! Your complete book is ready!")
    else:
        print(f"❌ HTML file not found: {html_file}")
