#!/usr/bin/env python3
"""
Convert HTML to PDF using Playwright
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def html_to_pdf(html_file, output_pdf):
    """Convert HTML file to PDF using Playwright"""
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Load HTML file
        await page.goto(f"file://{os.path.abspath(html_file)}")
        
        # Wait for content to load
        await page.wait_for_load_state('networkidle')
        
        # Generate PDF with proper settings
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
            prefer_css_page_size=True
        )
        
        await browser.close()
        print(f"PDF generated successfully: {output_pdf}")

if __name__ == "__main__":
    html_file = "Operational_Excellence_with_AI.html"
    output_pdf = "Operational_Excellence_with_AI_PLAYWRIGHT.pdf"
    
    if os.path.exists(html_file):
        asyncio.run(html_to_pdf(html_file, output_pdf))
    else:
        print(f"HTML file not found: {html_file}")
