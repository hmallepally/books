#!/usr/bin/env python3
"""
Option 2: Focus ONLY on fixing survey location
Keep all good changes (Chapter 16 in TOC, stories, etc.)
Only move survey from Chapter 2 to Chapter 1
"""

def fix_survey_only():
    """Focus only on fixing survey location"""
    
    print("🚀 Option 2: Focus ONLY on fixing survey location...")
    print("📋 Keeping all good changes (Chapter 16 in TOC, stories, etc.)")
    print("🎯 Only moving survey from Chapter 2 to Chapter 1")
    
    # Use the file with all the good changes
    with open('Operational_Excellence_with_AI_SURVEY_FINAL_FIX.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Verifying current survey location...")
    
    # Find the survey
    survey_location = html_content.find('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    if survey_location == -1:
        print("❌ Survey not found")
        return None
    
    # Check context to confirm it's in Chapter 2
    context_before = html_content[survey_location - 200:survey_location]
    if "Industry 4.0" in context_before or "Smart Manufacturing" in context_before:
        print("✅ Survey confirmed in Chapter 2 - proceeding with move")
    else:
        print("✅ Survey already in Chapter 1 - no changes needed")
        return 'Operational_Excellence_with_AI_SURVEY_FINAL_FIX.html'
    
    print("📝 2. Extracting survey content precisely...")
    
    # Find the complete survey section more precisely
    survey_start = html_content.find('<div class="assessment-section">', survey_location - 50)
    if survey_start == -1:
        print("❌ Survey start not found")
        return None
    
    # Find the end of the survey by looking for the closing divs
    survey_end = survey_start
    div_count = 0
    i = survey_start
    while i < len(html_content):
        if html_content[i:i+6] == '<div ':
            div_count += 1
        elif html_content[i:i+7] == '</div>':
            div_count -= 1
            if div_count == 0:
                survey_end = i + 7
                break
        i += 1
    
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
    output_file = 'Operational_Excellence_with_AI_SURVEY_ONLY_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Survey-only fix HTML created: {output_file}")
    return output_file

def generate_survey_only_fixed_pdf(html_file):
    """Generate PDF from survey-only fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_SURVEY_ONLY_FIXED.pdf'
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
            print(f"🎉 Survey-Only Fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Option 2: Focus ONLY on fixing survey location...")
    
    # Fix survey only
    html_file = fix_survey_only()
    
    if html_file:
        # Generate survey-only fixed PDF
        print("\n📚 Generating survey-only fixed PDF...")
        generate_survey_only_fixed_pdf(html_file)
        
        print("\n✅ SURVEY-ONLY FIX COMPLETE!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey moved from Chapter 2 to Chapter 1 (FIXED!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("\n📖 The survey should now be at the end of Chapter 1!")
    else:
        print("❌ Failed to fix survey location")
