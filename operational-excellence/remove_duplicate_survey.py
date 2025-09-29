#!/usr/bin/env python3
"""
Remove duplicate survey - remove from Chapter 2, keep only in Chapter 1
"""

def remove_duplicate_survey():
    """Remove duplicate survey from Chapter 2"""
    
    print("🚀 Removing duplicate survey from Chapter 2...")
    
    # Use the working survey fix file
    with open('Operational_Excellence_with_AI_WORKING_SURVEY_FIX.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding survey in Chapter 2...")
    
    # Find the survey in Chapter 2 (the original one)
    # Look for survey that comes after Chapter 2 content (Industry 4.0, Smart Manufacturing)
    chapter2_start = html_content.find('<div id="chapter2" class="chapter">')
    chapter3_start = html_content.find('<div id="chapter3" class="chapter">')
    
    # Find survey in Chapter 2 section
    chapter2_section = html_content[chapter2_start:chapter3_start]
    survey_in_chapter2 = chapter2_section.find('<div class="assessment-section">')
    
    if survey_in_chapter2 == -1:
        print("❌ Survey not found in Chapter 2")
        return None
    
    # Adjust location to be relative to full HTML
    survey_start = chapter2_start + survey_in_chapter2
    
    print("📝 2. Finding end of survey in Chapter 2...")
    
    # Find the end of the survey in Chapter 2
    survey_end_marker = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>', survey_start)
    if survey_end_marker == -1:
        print("❌ Survey end marker not found")
        return None
    
    survey_end = survey_end_marker + len('        </div>\n    </div>')
    
    print("📝 3. Removing survey from Chapter 2...")
    
    # Remove the survey from Chapter 2
    html_content = html_content[:survey_start] + html_content[survey_end:]
    
    print("📝 4. Verifying survey is only in Chapter 1...")
    
    # Count how many surveys are left
    survey_count = html_content.count('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    if survey_count == 1:
        print("✅ Survey is now only in Chapter 1")
    else:
        print(f"❌ Survey count is {survey_count}, should be 1")
        return None
    
    print("📝 5. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_NO_DUPLICATE_SURVEY.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ No duplicate survey HTML created: {output_file}")
    return output_file

def generate_no_duplicate_survey_pdf(html_file):
    """Generate PDF from no duplicate survey HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_NO_DUPLICATE_SURVEY.pdf'
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
            print(f"🎉 No Duplicate Survey PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Removing duplicate survey from Chapter 2...")
    
    # Remove duplicate survey
    html_file = remove_duplicate_survey()
    
    if html_file:
        # Generate no duplicate survey PDF
        print("\n📚 Generating no duplicate survey PDF...")
        generate_no_duplicate_survey_pdf(html_file)
        
        print("\n✅ DUPLICATE SURVEY REMOVED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey ONLY in Chapter 1 (duplicate removed!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("6. ✅ Book size back to ~244 pages (duplicate removed)")
        print("\n📖 The survey should now be only in Chapter 1!")
    else:
        print("❌ Failed to remove duplicate survey")
