#!/usr/bin/env python3
"""
Generate final complete PDF with Introduction marker and no page numbers
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_final_complete_pdf(html_file, output_file):
    """Generate PDF using Playwright WITHOUT page numbers"""
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
                
                # Generate PDF WITHOUT any page numbers
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
                    display_header_footer=False  # NO page numbers at all
                )
                
                await browser.close()
        
        print("Using Playwright for PDF generation WITHOUT page numbers...")
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
    print("🚀 Generating Final Complete PDF")
    print("=" * 40)
    
    # Define paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    output_file = "final_versions/with_toc/Operational_Excellence_with_AI_COMPLETE.pdf"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print(f"📄 Input HTML: {html_file}")
    print(f"📄 Output PDF: {output_file}")
    
    # Generate PDF with Playwright
    print("\n📊 Generating PDF with Playwright...")
    success = generate_final_complete_pdf(html_file, output_file)
    
    if success:
        print(f"\n🎉 PDF generation completed successfully!")
        print(f"📄 Final PDF: {output_file}")
        print(f"📊 File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
        
        print("\n📋 Key features in this PDF:")
        print("   ✅ Chapter end markers for ALL sections")
        print("   ✅ Introduction marker added")
        print("   ✅ NO page numbers (no duplicates)")
        print("   ✅ Clean, professional layout")
        
        print("\n🎨 Chapter End Markers include:")
        print("   • Introduction: The AI Transformation Journey")
        print("   • All 16 chapters")
        print("   • Thin horizontal line + three dots")
        print("   • No text, just visual elements")
        
        print("\n🔍 Please review the PDF and check:")
        print("   • Introduction has end marker")
        print("   • All chapters have end markers")
        print("   • No page numbers (clean layout)")
        print("   • Page count close to 195")
        
    else:
        print("\n❌ PDF generation failed!")
        print("\n💡 To install Playwright:")
        print("   • pip install playwright")
        print("   • playwright install")

if __name__ == "__main__":
    main()
