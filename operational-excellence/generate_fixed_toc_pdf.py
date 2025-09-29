#!/usr/bin/env python3
"""
Generate the Ultimate PDF Book using Playwright - Fixed TOC Version
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def generate_ultimate_pdf_fixed(html_file, output_pdf):
    """Generate the ultimate PDF with perfect formatting and fixed TOC"""
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
        print(f"🎉 ULTIMATE PDF (Fixed TOC) generated successfully: {output_pdf}")
        print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")

if __name__ == "__main__":
    html_file = "Operational_Excellence_with_AI_BEAUTIFUL.html"
    output_pdf = "Operational_Excellence_with_AI_ULTIMATE_FIXED_TOC.pdf"
    
    if os.path.exists(html_file):
        print("🚀 Generating the ultimate PDF book with fixed TOC...")
        asyncio.run(generate_ultimate_pdf_fixed(html_file, output_pdf))
        print("✅ SUCCESS! Your ultimate book with fixed TOC is ready!")
    else:
        print(f"❌ HTML file not found: {html_file}")
