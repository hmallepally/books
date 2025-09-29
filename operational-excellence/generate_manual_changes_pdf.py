#!/usr/bin/env python3
"""
Generate PDF from manually updated HTML file
"""

def generate_manual_changes_pdf():
    """Generate PDF from manually updated HTML file"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath('Operational_Excellence_with_AI_FINAL_CLEAN.html')}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_MANUAL_CHANGES.pdf'
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
            print(f"🎉 Manual Changes PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Generating PDF from manually updated HTML file...")
    
    # Generate PDF from manual changes
    generate_manual_changes_pdf()
    
    print("\n✅ PDF GENERATED!")
    print("📋 This PDF includes:")
    print("1. ✅ Your manual changes")
    print("2. ✅ All previous content preserved")
    print("3. ✅ Proper formatting and styling")
    print("\n📖 The PDF should reflect all your manual changes!")
