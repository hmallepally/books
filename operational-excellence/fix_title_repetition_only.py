#!/usr/bin/env python3
"""
Fix Issue 1: Remove Title Page Content Repetition
"""

def fix_title_repetition():
    """Fix only the title page content repetition"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_ALL_ISSUES_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing title page content repetition...")
    
    # Remove the specific title page content that appears after TOC
    # This is the exact content we see on lines 382-390
    title_content_to_remove = """A Comprehensive Guide to AI-Driven Performance Optimization</h2>
<p><div class="tech-label">Author:</div>
<p>Hari Mallepally</p><br />
<div class="tech-label">Version:</div>
<p>3.0</p><br />
<div class="tech-label">Publication Date:</div>
<p>September 2025</p><br />
<div class="tech-label">Target Audience:</div>
<p>Business Leaders, Operations Professionals, MBA Students</p></p>

</div>"""
    
    # Remove this content
    html_content = html_content.replace(title_content_to_remove, '')
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_TITLE_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Title repetition fixed HTML created: {output_file}")
    return output_file

def generate_title_fixed_pdf(html_file):
    """Generate PDF from title fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_TITLE_FIXED.pdf'
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
            print(f"🎉 Title fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing title page content repetition...")
    
    # Fix title repetition
    html_file = fix_title_repetition()
    
    if html_file:
        # Generate title fixed PDF
        print("\n📚 Generating title fixed PDF...")
        generate_title_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Title repetition fixed:")
        print("1. ✅ Removed title page content repetition on page 3")
        print("\n📖 Please review this PDF first!")
        print("🔍 Check if the title repetition is gone, then we'll fix the next issue.")
    else:
        print("❌ Failed to fix title repetition")
