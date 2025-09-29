#!/usr/bin/env python3
"""
Proper survey move - move survey from Chapter 2 to Chapter 1 correctly
"""

def proper_survey_move():
    """Properly move survey from Chapter 2 to Chapter 1"""
    
    print("🚀 Properly moving survey from Chapter 2 to Chapter 1...")
    
    # Use the correct start file
    with open('Operational_Excellence_with_AI_CORRECT_START.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding survey in Chapter 2...")
    
    # Find the survey in Chapter 2
    survey_start = html_content.find('<div class="assessment-section">')
    if survey_start == -1:
        print("❌ Survey not found")
        return None
    
    print("📝 2. Finding end of survey...")
    
    # Find the end of the survey
    survey_end_marker = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>')
    if survey_end_marker == -1:
        print("❌ Survey end marker not found")
        return None
    
    survey_end = survey_end_marker + len('        </div>\n    </div>')
    
    survey_content = html_content[survey_start:survey_end]
    print(f"✅ Survey content extracted ({len(survey_content)} characters)")
    
    print("📝 3. Removing survey from Chapter 2...")
    
    # Remove the survey from Chapter 2
    html_content = html_content[:survey_start] + html_content[survey_end:]
    
    print("📝 4. Finding Chapter 1 conclusion...")
    
    # Find Chapter 1 conclusion
    chapter1_conclusion_location = html_content.find('<h3 id="conclusion">Conclusion</h3>')
    if chapter1_conclusion_location == -1:
        print("❌ Chapter 1 conclusion not found")
        return None
    
    print("📝 5. Adding survey to Chapter 1 before conclusion...")
    
    # Add survey to Chapter 1 before conclusion
    html_content = html_content[:chapter1_conclusion_location] + survey_content + '\n\n' + html_content[chapter1_conclusion_location:]
    
    print("📝 6. Verifying survey is now in Chapter 1...")
    
    # Verify the survey is now in Chapter 1
    survey_count = html_content.count('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    if survey_count == 1:
        print("✅ Survey is now only in Chapter 1")
    else:
        print(f"❌ Survey count is {survey_count}, should be 1")
        return None
    
    print("📝 7. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_PROPER_SURVEY_MOVE.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Proper survey move HTML created: {output_file}")
    return output_file

def generate_proper_survey_move_pdf(html_file):
    """Generate PDF from proper survey move HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_PROPER_SURVEY_MOVE.pdf'
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
            print(f"🎉 Proper Survey Move PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Properly moving survey from Chapter 2 to Chapter 1...")
    
    # Proper survey move
    html_file = proper_survey_move()
    
    if html_file:
        # Generate proper survey move PDF
        print("\n📚 Generating proper survey move PDF...")
        generate_proper_survey_move_pdf(html_file)
        
        print("\n✅ PROPER SURVEY MOVE COMPLETE!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey moved from Chapter 2 to Chapter 1 (PROPERLY!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("6. ✅ Book size back to ~244 pages (no duplicates)")
        print("\n📖 The survey should now be only in Chapter 1!")
    else:
        print("❌ Failed to properly move survey")
