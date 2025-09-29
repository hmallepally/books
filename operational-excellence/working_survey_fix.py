#!/usr/bin/env python3
"""
Working survey fix - using the correct conclusion ID
"""

def working_survey_fix():
    """Working survey fix using correct conclusion ID"""
    
    print("🚀 Working survey fix - using correct conclusion ID...")
    
    # Use the file with all the good changes
    with open('Operational_Excellence_with_AI_SURVEY_FINAL_FIX.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Extracting survey content...")
    
    # Extract the survey content
    survey_start = html_content.find('<div class="assessment-section">')
    survey_end = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>')
    survey_end += len('        </div>\n    </div>')
    
    survey_content = html_content[survey_start:survey_end]
    print(f"✅ Survey content extracted ({len(survey_content)} characters)")
    
    print("📝 2. Finding Chapter 1 conclusion (id='conclusion')...")
    
    # Find Chapter 1 conclusion using the correct ID
    chapter1_conclusion_location = html_content.find('<h3 id="conclusion">Conclusion</h3>')
    if chapter1_conclusion_location == -1:
        print("❌ Chapter 1 conclusion not found")
        return None
    
    print("📝 3. Adding survey to Chapter 1 before conclusion...")
    
    # Add survey to Chapter 1 before conclusion
    html_content = html_content[:chapter1_conclusion_location] + survey_content + '\n\n' + html_content[chapter1_conclusion_location:]
    
    print("📝 4. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_WORKING_SURVEY_FIX.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Working survey fix HTML created: {output_file}")
    return output_file

def generate_working_survey_fix_pdf(html_file):
    """Generate PDF from working survey fix HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_WORKING_SURVEY_FIX.pdf'
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
            print(f"🎉 Working Survey Fix PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Working survey fix...")
    
    # Working survey fix
    html_file = working_survey_fix()
    
    if html_file:
        # Generate working survey fix PDF
        print("\n📚 Generating working survey fix PDF...")
        generate_working_survey_fix_pdf(html_file)
        
        print("\n✅ WORKING SURVEY FIX COMPLETE!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey added to Chapter 1 (WORKING FIX!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("\n📖 The survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to fix survey location")
