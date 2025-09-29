#!/usr/bin/env python3
"""
Generate final PDF with subtle chapter markers and page numbers
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_pdf_with_markers_and_pages(html_file, output_file):
    """Generate PDF using Playwright with subtle markers and page numbers"""
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
                
                # Generate PDF with page numbers
                await page.pdf(
                    path=output_file,
                    format='A4',
                    margin={
                        'top': '0.75in',
                        'right': '0.75in',
                        'bottom': '0.75in',
                        'left': '0.75in'
                    },
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=True,
                    header_template='<div></div>',
                    footer_template='<div style="font-size: 8px; text-align: center; width: 100%; color: #666;"><span class="pageNumber"></span></div>'
                )
                
                await browser.close()
        
        print("Using Playwright for PDF generation with subtle markers and page numbers...")
        asyncio.run(generate())
        
        print(f"✅ PDF generated successfully: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        return True
        
    except ImportError:
        print("❌ Playwright not available")
        return False
    except Exception as e:
        print(f"❌ Playwright error: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Generating Final PDF with Subtle Chapter Markers")
    print("=" * 60)
    
    # Define paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    output_file = "final_versions/with_toc/Operational_Excellence_with_AI_FINAL_WITH_MARKERS.pdf"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print(f"📄 Input HTML: {html_file}")
    print(f"📄 Output PDF: {output_file}")
    
    # Generate PDF with Playwright
    print("\n📊 Generating PDF with Playwright...")
    success = generate_pdf_with_markers_and_pages(html_file, output_file)
    
    if success:
        print(f"\n🎉 PDF generation completed successfully!")
        print(f"📄 Final PDF: {output_file}")
        print(f"📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        print("\n📋 Key features in this PDF:")
        print("   ✅ Subtle chapter end markers")
        print("   ✅ Single page numbers (no duplicates)")
        print("   ✅ Compact design (minimal page count increase)")
        print("   ✅ Professional layout")
        
        print("\n🎨 Subtle Chapter Markers include:")
        print("   • Thin horizontal line (40% width)")
        print("   • Three small dots")
        print("   • Minimal spacing")
        print("   • Print-optimized sizing")
        
        print("\n🔍 Please review the PDF and check:")
        print("   • Page count (should be close to 195)")
        print("   • Single page numbers (no duplicates)")
        print("   • Subtle chapter markers")
        print("   • Overall visual appeal")
        
    else:
        print("\n❌ PDF generation failed!")
        print("\n💡 To install Playwright:")
        print("   • pip install playwright")
        print("   • playwright install")

if __name__ == "__main__":
    main()
