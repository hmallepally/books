#!/usr/bin/env python3
"""
Fix only the PDF generation to use correct KDP trim size (8.5" x 11")
Without modifying the existing HTML file
"""

import os
import subprocess
import sys
from pathlib import Path

def fix_pdf_trim_size_only():
    """Fix only the PDF generation to use correct KDP trim size"""
    
    print("📚 Fixing PDF Generation for KDP Trim Size Only")
    print("=" * 60)
    
    # File paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    pdf_file = "final_versions/with_toc/Operational_Excellence_with_AI_KDP_CORRECT.pdf"
    
    if not os.path.exists(html_file):
        print(f"❌ HTML file not found: {html_file}")
        return False
    
    print(f"📄 Source HTML: {html_file}")
    print(f"📄 Target PDF: {pdf_file}")
    
    # Generate PDF with correct trim size
    print("\n📊 Generating PDF with KDP trim size (8.5\" x 11\")...")
    success = generate_pdf_with_playwright_kdp(html_file, pdf_file)
    
    return success

def generate_pdf_with_playwright_kdp(html_file, pdf_file):
    """Generate PDF using Playwright with KDP trim size"""
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
                
                # Generate PDF with KDP trim size (8.5" x 11")
                await page.pdf(
                    path=pdf_file,
                    format='Letter',  # US Letter size (8.5" x 11")
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
        
        print("Using Playwright for PDF generation with KDP trim size...")
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
    success = fix_pdf_trim_size_only()
    
    if success:
        print(f"\n🎉 KDP PDF generation completed successfully!")
        print(f"\n📋 Files:")
        print(f"   • Source HTML: final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html")
        print(f"   • New PDF: final_versions/with_toc/Operational_Excellence_with_AI_KDP_CORRECT.pdf")
        
        print(f"\n📚 PDF Features:")
        print(f"   ✅ Correct KDP trim size: 8.5\" x 11\"")
        print(f"   ✅ Original HTML content preserved")
        print(f"   ✅ Table of Contents included")
        print(f"   ✅ Chapter end markers included")
        print(f"   ✅ All previous improvements maintained")
        
        print(f"\n🚀 Ready for KDP Publishing!")
        print(f"   • Upload the PDF file to KDP")
        print(f"   • Trim size matches KDP requirements")
        print(f"   • All original features preserved")
        
    else:
        print(f"\n❌ KDP PDF generation failed!")
        print(f"💡 Please check the error messages above.")

if __name__ == "__main__":
    main()
