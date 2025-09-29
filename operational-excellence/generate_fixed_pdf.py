#!/usr/bin/env python3
"""
Generate PDF from Fixed HTML
"""

import asyncio
from playwright.async_api import async_playwright
import os

async def generate_pdf():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        await page.goto(f"file://{os.path.abspath('Operational_Excellence_with_AI_FIXED.html')}")
        await page.wait_for_load_state('networkidle')
        
        output_pdf = 'Operational_Excellence_with_AI_ALL_FIXES.pdf'
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
        print(f"🎉 Final PDF with fixes generated: {output_pdf}")
        print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")

if __name__ == "__main__":
    asyncio.run(generate_pdf())
