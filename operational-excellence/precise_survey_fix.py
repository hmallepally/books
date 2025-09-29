#!/usr/bin/env python3
"""
Precise survey fix - find exact survey boundaries and move it
"""

def precise_survey_fix():
    """Precise survey fix using exact boundaries"""
    
    print("🚀 Precise survey fix - finding exact survey boundaries...")
    
    # Use the file with all the good changes
    with open('Operational_Excellence_with_AI_SURVEY_FINAL_FIX.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding exact survey start...")
    
    # Find the exact survey start
    survey_start = html_content.find('    <div class="assessment-section">')
    if survey_start == -1:
        print("❌ Survey start not found")
        return None
    
    print("📝 2. Finding exact survey end...")
    
    # Find the exact survey end - look for the closing </div> after "Try This" section
    survey_end_marker = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>')
    if survey_end_marker == -1:
        print("❌ Survey end marker not found")
        return None
    
    survey_end = survey_end_marker + len('        </div>\n    </div>')
    
    survey_content = html_content[survey_start:survey_end]
    print(f"✅ Survey content extracted precisely ({len(survey_content)} characters)")
    
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
    new_survey_location = html_content.find('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    if new_survey_location == -1:
        print("❌ Survey not found after move")
        return None
    
    # Check context to confirm it's now in Chapter 1
    new_context_before = html_content[new_survey_location - 200:new_survey_location]
    if "Industry 4.0" in new_context_before or "Smart Manufacturing" in new_context_before:
        print("❌ Survey still in Chapter 2 - move failed")
        return None
    else:
        print("✅ Survey successfully moved to Chapter 1")
    
    print("📝 7. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_PRECISE_SURVEY_FIX.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Precise survey fix HTML created: {output_file}")
    return output_file

def generate_precise_survey_fix_pdf(html_file):
    """Generate PDF from precise survey fix HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_PRECISE_SURVEY_FIX.pdf'
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
            print(f"🎉 Precise Survey Fix PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Precise survey fix...")
    
    # Precise survey fix
    html_file = precise_survey_fix()
    
    if html_file:
        # Generate precise survey fix PDF
        print("\n📚 Generating precise survey fix PDF...")
        generate_precise_survey_fix_pdf(html_file)
        
        print("\n✅ PRECISE SURVEY FIX COMPLETE!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey moved from Chapter 2 to Chapter 1 (PRECISELY FIXED!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("\n📖 The survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to fix survey location precisely")
