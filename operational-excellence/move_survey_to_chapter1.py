#!/usr/bin/env python3
"""
Move survey from Chapter 2 to Chapter 1 specifically
"""

def move_survey_to_chapter1():
    """Move survey from Chapter 2 to Chapter 1"""
    
    print("🚀 Moving survey from Chapter 2 to Chapter 1...")
    
    # Read the manual fix HTML file
    with open('Operational_Excellence_with_AI_MANUAL_FIX.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding survey content in Chapter 2...")
    
    # Find the survey content
    survey_start = html_content.find('<div class="assessment-section">')
    if survey_start == -1:
        print("❌ Survey start not found")
        return None
    
    # Find the end of the survey
    survey_end = html_content.find('</div>', survey_start)
    survey_end = html_content.find('</div>', survey_end + 6)  # Second closing div
    survey_end += 6  # Include the </div>
    
    survey_content = html_content[survey_start:survey_end]
    print(f"✅ Survey content found ({len(survey_content)} characters)")
    
    print("📝 2. Removing survey from Chapter 2...")
    
    # Remove the survey from Chapter 2
    html_content = html_content[:survey_start] + html_content[survey_end:]
    
    print("📝 3. Finding Chapter 1 conclusion to add survey before it...")
    
    # Find Chapter 1 conclusion
    chapter1_conclusion_location = html_content.find('<h3 id="conclusion">Conclusion</h3>')
    if chapter1_conclusion_location == -1:
        print("❌ Chapter 1 conclusion not found")
        return None
    
    print("📝 4. Adding survey to Chapter 1 before conclusion...")
    
    # Add survey to Chapter 1 before conclusion
    html_content = html_content[:chapter1_conclusion_location] + survey_content + '\n\n' + html_content[chapter1_conclusion_location:]
    
    print("📝 5. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_SURVEY_MOVED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Survey moved HTML created: {output_file}")
    return output_file

def generate_survey_moved_pdf(html_file):
    """Generate PDF from survey moved HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_SURVEY_MOVED.pdf'
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
            print(f"🎉 Survey Moved PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Moving survey from Chapter 2 to Chapter 1...")
    
    # Move survey to Chapter 1
    html_file = move_survey_to_chapter1()
    
    if html_file:
        # Generate survey moved PDF
        print("\n📚 Generating survey moved PDF...")
        generate_survey_moved_pdf(html_file)
        
        print("\n✅ SURVEY MOVED TO CHAPTER 1!")
        print("📋 This PDF now includes:")
        print("1. ✅ Survey moved from Chapter 2 to Chapter 1")
        print("2. ✅ Survey appears at the end of Chapter 1")
        print("3. ✅ All other content preserved")
        print("\n📖 Please check - the survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to move survey")
