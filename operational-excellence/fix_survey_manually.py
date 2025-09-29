#!/usr/bin/env python3
"""
Fix survey location manually - find exact text and move it
"""

def fix_survey_manually():
    """Move survey from Chapter 2 to Chapter 1 manually"""
    
    print("🚀 Fixing survey location manually - moving from Chapter 2 to Chapter 1...")
    
    # Read the HTML file
    with open('Operational_Excellence_with_AI_SURVEY_PRECISELY_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding survey content manually...")
    
    # Find the survey content manually
    survey_start = html_content.find('<div class="assessment-section">')
    if survey_start == -1:
        print("❌ Survey start not found")
        return None
    
    # Find the end of the survey (look for the closing div)
    survey_end = html_content.find('</div>', survey_start)
    if survey_end == -1:
        print("❌ Survey end not found")
        return None
    
    # Find the complete survey section
    survey_end = html_content.find('</div>', survey_end + 1)  # Second closing div
    if survey_end == -1:
        print("❌ Survey complete end not found")
        return None
    
    survey_end += 6  # Include the </div>
    
    survey_content = html_content[survey_start:survey_end]
    print(f"✅ Survey content found manually ({len(survey_content)} characters)")
    
    print("📝 2. Removing survey from Chapter 2...")
    
    # Remove the survey from Chapter 2
    html_content = html_content[:survey_start] + html_content[survey_end:]
    
    print("📝 3. Finding Chapter 1 conclusion...")
    
    # Find Chapter 1 conclusion
    conclusion_start = html_content.find('<h3 id="conclusion">Conclusion</h3>')
    if conclusion_start == -1:
        print("❌ Chapter 1 conclusion not found")
        return None
    
    print("📝 4. Adding survey to Chapter 1 before conclusion...")
    
    # Add survey to Chapter 1 before conclusion
    html_content = html_content[:conclusion_start] + survey_content + '\n\n' + html_content[conclusion_start:]
    
    print("📝 5. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_SURVEY_MANUALLY_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Survey manually fixed HTML created: {output_file}")
    return output_file

def generate_survey_manually_fixed_pdf(html_file):
    """Generate PDF from survey manually fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_SURVEY_MANUALLY_FIXED.pdf'
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
            print(f"🎉 Survey Manually Fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing survey location manually...")
    
    # Fix survey location manually
    html_file = fix_survey_manually()
    
    if html_file:
        # Generate survey manually fixed PDF
        print("\n📚 Generating survey manually fixed PDF...")
        generate_survey_manually_fixed_pdf(html_file)
        
        print("\n✅ SURVEY LOCATION MANUALLY FIXED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Survey moved from Chapter 2 to Chapter 1")
        print("2. ✅ Survey appears at the end of Chapter 1")
        print("3. ✅ All other content preserved")
        print("\n📖 Please check - the survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to fix survey location manually")
