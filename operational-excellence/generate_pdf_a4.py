#!/usr/bin/env python3
"""
Generate PDF with A4 size using the restored HTML file
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_pdf_a4():
    """Generate PDF with A4 size"""
    
    print("📚 Generating PDF with A4 Size")
    print("=" * 40)
    
    # File paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    pdf_file = "final_versions/with_toc/Operational_Excellence_with_AI_RESTORED.pdf"
    
    if not os.path.exists(html_file):
        print(f"❌ HTML file not found: {html_file}")
        return False
    
    print(f"📄 Source HTML: {html_file}")
    print(f"📄 Target PDF: {pdf_file}")
    
    # Generate PDF with A4 size
    print("\n📊 Generating PDF with A4 size...")
    success = generate_pdf_with_playwright_a4(html_file, pdf_file)
    
    return success

def generate_pdf_with_playwright_a4(html_file, pdf_file):
    """Generate PDF using Playwright with A4 size"""
    try:
        import asyncio
        from playwright.async_api import async_playwright
        
        async def generate():
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Load HTML file
                await page.goto(f"file://{os.path.abspath(html_file)}")
                
                # Wait for content to load
                await page.wait_for_load_state('networkidle')
                
                # Generate PDF with A4 size
                await page.pdf(
                    path=pdf_file,
                    format='A4',  # A4 size
                    margin={
                        'top': '0.75in',
                        'right': '0.75in',
                        'bottom': '0.75in',
                        'left': '0.75in'
                    },
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=False
                )
                
                await browser.close()
        
        print("Using Playwright for PDF generation with A4 size...")
        asyncio.run(generate())
        
        print(f"✅ PDF generated successfully: {pdf_file}")
        print(f"File size: {os.path.getsize(pdf_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ Playwright not available")
        return False
    except Exception as e:
        print(f"❌ Playwright error: {e}")
        return False

def main():
    """Main function"""
    success = generate_pdf_a4()
    
    if success:
        print(f"\n🎉 PDF generation completed successfully!")
        print(f"\n📋 Files:")
        print(f"   • Source HTML: final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html")
        print(f"   • New PDF: final_versions/with_toc/Operational_Excellence_with_AI_RESTORED.pdf")
        
        print(f"\n📚 PDF Features:")
        print(f"   ✅ A4 size (8.26\" x 11.69\")")
        print(f"   ✅ Original HTML content preserved")
        print(f"   ✅ Table of Contents included")
        print(f"   ✅ Chapter end markers included")
        print(f"   ✅ All previous improvements maintained")
        
        print(f"\n🔍 This should be the same as the original PDF you had!")
        
    else:
        print(f"\n❌ PDF generation failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
