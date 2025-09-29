#!/usr/bin/env python3
"""
Fix survey location - move it from Chapter 2 to Chapter 1
"""

def fix_survey_location():
    """Move survey from Chapter 2 to Chapter 1"""
    import re
    
    print("🚀 Fixing survey location - moving from Chapter 2 to Chapter 1...")
    
    # Read the HTML file
    with open('Operational_Excellence_with_AI_COMPLETE_WITH_CONTENT.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Extracting survey content from Chapter 2...")
    
    # Extract the survey content from Chapter 2
    survey_match = re.search(
        r'(<div class="assessment-section">.*?</div>\s*</div>)',
        html_content,
        flags=re.DOTALL
    )
    
    if not survey_match:
        print("❌ Survey not found in Chapter 2")
        return None
    
    survey_content = survey_match.group(1)
    print(f"✅ Survey content extracted ({len(survey_content)} characters)")
    
    print("📝 2. Removing survey from Chapter 2...")
    
    # Remove the survey from Chapter 2
    html_content = re.sub(
        r'<div class="assessment-section">.*?</div>\s*</div>',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    print("📝 3. Adding survey to Chapter 1 before conclusion...")
    
    # Add survey to Chapter 1 before conclusion
    html_content = re.sub(
        r'(<h3 id="conclusion">Conclusion</h3>)',
        survey_content + r'\1',
        html_content
    )
    
    print("📝 4. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_SURVEY_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Survey location fixed HTML created: {output_file}")
    return output_file

def generate_survey_fixed_pdf(html_file):
    """Generate PDF from survey fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_SURVEY_FIXED.pdf'
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
            print(f"🎉 Survey Fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing survey location...")
    
    # Fix survey location
    html_file = fix_survey_location()
    
    if html_file:
        # Generate survey fixed PDF
        print("\n📚 Generating survey fixed PDF...")
        generate_survey_fixed_pdf(html_file)
        
        print("\n✅ SURVEY LOCATION FIXED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Survey moved from Chapter 2 to Chapter 1")
        print("2. ✅ Survey appears at the end of Chapter 1")
        print("3. ✅ All other content preserved")
        print("\n📖 Please check - the survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to fix survey location")
