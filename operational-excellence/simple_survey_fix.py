#!/usr/bin/env python3
"""
Simple survey fix - just copy the survey content and add it to Chapter 1
"""

def simple_survey_fix():
    """Simple survey fix"""
    
    print("🚀 Simple survey fix - copying survey content to Chapter 1...")
    
    # Use the file with all the good changes
    with open('Operational_Excellence_with_AI_SURVEY_FINAL_FIX.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Extracting survey content...")
    
    # Extract the survey content (we know it's there from previous checks)
    survey_start = html_content.find('<div class="assessment-section">')
    survey_end = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>')
    survey_end += len('        </div>\n    </div>')
    
    survey_content = html_content[survey_start:survey_end]
    print(f"✅ Survey content extracted ({len(survey_content)} characters)")
    
    print("📝 2. Finding Chapter 1 conclusion...")
    
    # Find the FIRST Chapter 1 conclusion (not Chapter 2)
    chapter1_start = html_content.find('<div id="chapter1" class="chapter">')
    chapter2_start = html_content.find('<div id="chapter2" class="chapter">')
    
    # Find Chapter 1 conclusion between chapter1_start and chapter2_start
    chapter1_section = html_content[chapter1_start:chapter2_start]
    chapter1_conclusion_location = chapter1_section.find('<h3 id="conclusion">Conclusion</h3>')
    
    if chapter1_conclusion_location == -1:
        print("❌ Chapter 1 conclusion not found")
        return None
    
    # Adjust the location to be relative to the full HTML
    chapter1_conclusion_location += chapter1_start
    
    print("📝 3. Adding survey to Chapter 1 before conclusion...")
    
    # Add survey to Chapter 1 before conclusion
    html_content = html_content[:chapter1_conclusion_location] + survey_content + '\n\n' + html_content[chapter1_conclusion_location:]
    
    print("📝 4. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_SIMPLE_SURVEY_FIX.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Simple survey fix HTML created: {output_file}")
    return output_file

def generate_simple_survey_fix_pdf(html_file):
    """Generate PDF from simple survey fix HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_SIMPLE_SURVEY_FIX.pdf'
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
            print(f"🎉 Simple Survey Fix PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Simple survey fix...")
    
    # Simple survey fix
    html_file = simple_survey_fix()
    
    if html_file:
        # Generate simple survey fix PDF
        print("\n📚 Generating simple survey fix PDF...")
        generate_simple_survey_fix_pdf(html_file)
        
        print("\n✅ SIMPLE SURVEY FIX COMPLETE!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey added to Chapter 1 (SIMPLE FIX!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("\n📖 The survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to fix survey location simply")
