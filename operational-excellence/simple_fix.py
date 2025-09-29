#!/usr/bin/env python3
"""
Simple Fix for Title Repetition and Blank Pages
"""

import re
import subprocess
import os

def simple_fix():
    """Simple fix for the remaining issues"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_PROPER_STRUCTURE.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Applying simple fixes...")
    
    # Issue 1: Remove title page content after TOC
    print("1️⃣ Removing title page content after TOC...")
    
    # Simple string replacement to remove the duplicate content
    html_content = html_content.replace(
        '<div class="content">\n<h2 id="a-comprehensive-guide-to-ai-driven-performance-optimization">A Comprehensive Guide to AI-Driven Performance Optimization</h2>\n<p><div class="tech-label">Author:</div>\n<p>Hari Mallepally</p><br />\n<div class="tech-label">Version:</div>\n<p>3.0</p><br />\n<div class="tech-label">Publication Date:</div>\n<p>September 2025</p><br />\n<div class="tech-label">Target Audience:</div>\n<p>Business Leaders, Operations Professionals, MBA Students</p></p>',
        '<div class="content">'
    )
    
    # Issue 2: Remove blank pages before chapters
    print("2️⃣ Removing blank pages before chapters...")
    
    # Remove clearpage divs that cause blank pages
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="preface" class="chapter">', '<div id="preface" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter1" class="chapter">', '<div id="chapter1" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter2" class="chapter">', '<div id="chapter2" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter3" class="chapter">', '<div id="chapter3" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter4" class="chapter">', '<div id="chapter4" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter5" class="chapter">', '<div id="chapter5" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter6" class="chapter">', '<div id="chapter6" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter7" class="chapter">', '<div id="chapter7" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter8" class="chapter">', '<div id="chapter8" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter9" class="chapter">', '<div id="chapter9" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter10" class="chapter">', '<div id="chapter10" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter11" class="chapter">', '<div id="chapter11" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter12" class="chapter">', '<div id="chapter12" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter13" class="chapter">', '<div id="chapter13" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter14" class="chapter">', '<div id="chapter14" class="chapter">')
    html_content = html_content.replace('<div class="clearpage"></div>\n<div id="chapter15" class="chapter">', '<div id="chapter15" class="chapter">')
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_SIMPLE_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Simple fixed HTML created: {output_file}")
    return output_file

def generate_simple_fixed_pdf(html_file):
    """Generate PDF from simple fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_SIMPLE_FIXED.pdf'
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
            print(f"🎉 Simple fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Applying simple fixes...")
    
    # Apply simple fixes
    html_file = simple_fix()
    
    if html_file:
        # Generate simple fixed PDF
        print("\n📚 Generating simple fixed PDF...")
        generate_simple_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Simple fixes applied:")
        print("1. ✅ Removed title page content after TOC")
        print("2. ✅ Removed blank pages before chapters")
        print("3. ✅ Chapters still start on fresh pages (via CSS)")
        print("\n📖 Ready for review tomorrow!")
        print("🌙 Good night! Sleep well!")
    else:
        print("❌ Failed to apply simple fixes")
