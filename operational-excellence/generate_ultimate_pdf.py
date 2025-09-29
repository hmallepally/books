#!/usr/bin/env python3
"""
Generate the Ultimate PDF Book using Playwright
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def generate_ultimate_pdf(html_file, output_pdf):
    """Generate the ultimate PDF with perfect formatting"""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load HTML file
        await page.goto(f"file://{os.path.abspath(html_file)}")
        
        # Wait for content to load
        await page.wait_for_load_state('networkidle')
        
        # Generate PDF with ultimate settings
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
        print(f"🎉 ULTIMATE PDF generated successfully: {output_pdf}")
        print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")

if __name__ == "__main__":
    html_file = "Operational_Excellence_with_AI_BEAUTIFUL.html"
    output_pdf = "Operational_Excellence_with_AI_ULTIMATE_BOOK.pdf"
    
    if os.path.exists(html_file):
        print("🚀 Generating the ultimate PDF book...")
        asyncio.run(generate_ultimate_pdf(html_file, output_pdf))
        print("✅ SUCCESS! Your ultimate book is ready!")
    else:
        print(f"❌ HTML file not found: {html_file}")
