#!/usr/bin/env python3
"""
Generate PDF without any page numbers to avoid duplicates
"""

import os
import subprocess
import sys
from pathlib import Path

def generate_pdf_without_page_numbers(html_file, output_file):
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

def generate_pdf_with_custom_page_numbers(html_file, output_file):
    """Generate PDF with custom page numbering using CSS"""
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
                
                # Add custom CSS for page numbers
                await page.add_style_tag(content="""
                    @page {
                        @bottom-center {
                            content: counter(page);
                            font-size: 10px;
                            color: #666;
                        }
                    }
                """)
                
                # Generate PDF with custom page numbering
                await page.pdf(
                    path=output_file,
                    format='A4',
                    margin={
                        'top': '0.75in',
                        'right': '0.75in',
                        'bottom': '1in',  # Extra space for page numbers
                        'left': '0.75in'
                    },
                    print_background=True,
                    prefer_css_page_size=True,
                    display_header_footer=False
                )
                
                await browser.close()
        
        print("Using Playwright with custom CSS page numbering...")
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
    print("🚀 Generating PDF with Fixed Page Numbering")
    print("=" * 50)
    
    # Define paths
    html_file = "final_versions/with_toc/Operational_Excellence_with_AI_WITH_TOC.html"
    
    # Check if HTML file exists
    if not os.path.exists(html_file):
        print(f"Error: HTML file {html_file} not found")
        return False
    
    print(f"📄 Input HTML: {html_file}")
    
    # Try method 1: No page numbers at all
    print("\n📊 Method 1: Generating PDF without page numbers...")
    output_file1 = "final_versions/with_toc/Operational_Excellence_with_AI_NO_PAGES.pdf"
    success1 = generate_pdf_without_page_numbers(html_file, output_file1)
    
    if success1:
        print(f"✅ Method 1 succeeded: {output_file1}")
    
    # Try method 2: Custom CSS page numbers
    print("\n📊 Method 2: Generating PDF with custom CSS page numbers...")
    output_file2 = "final_versions/with_toc/Operational_Excellence_with_AI_CUSTOM_PAGES.pdf"
    success2 = generate_pdf_with_custom_page_numbers(html_file, output_file2)
    
    if success2:
        print(f"✅ Method 2 succeeded: {output_file2}")
    
    if success1 or success2:
        print(f"\n🎉 PDF generation completed!")
        print(f"\n📋 Generated files:")
        if success1:
            print(f"   • {output_file1} (no page numbers)")
        if success2:
            print(f"   • {output_file2} (custom page numbers)")
        
        print(f"\n🔍 Please review both PDFs and check:")
        print(f"   • Which one has better page numbering?")
        print(f"   • Are there any duplicate page numbers?")
        print(f"   • Which layout do you prefer?")
        
    else:
        print("\n❌ Both PDF generation methods failed!")
        print("\n💡 To install Playwright:")
        print("   • pip install playwright")
        print("   • playwright install")

if __name__ == "__main__":
    main()
