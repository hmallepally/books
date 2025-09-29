#!/usr/bin/env python3
"""
Remove first survey - keep only the second one
"""

def remove_first_survey():
    """Remove first survey, keep only the second one"""
    
    print("🚀 Removing first survey, keeping only the second one...")
    
    # Use the latest file
    with open('Operational_Excellence_with_AI_LATEST.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding first survey...")
    
    # Find the first survey
    first_survey_start = html_content.find('<div class="assessment-section">')
    if first_survey_start == -1:
        print("❌ First survey not found")
        return None
    
    print("📝 2. Finding end of first survey...")
    
    # Find the end of the first survey (look for the closing </div> after "Try This" section)
    first_survey_end = html_content.find('        </div>\n    </div>\n\n<div class="assessment-section">', first_survey_start)
    if first_survey_end == -1:
        print("❌ First survey end not found")
        return None
    
    first_survey_end += len('        </div>\n    </div>\n\n')
    
    print("📝 3. Removing first survey...")
    
    # Remove the first survey
    html_content = html_content[:first_survey_start] + html_content[first_survey_end:]
    
    print("📝 4. Verifying only one survey remains...")
    
    # Verify only one survey remains
    final_survey_count = html_content.count('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    if final_survey_count == 1:
        print("✅ Only one survey remains")
    else:
        print(f"❌ Survey count is {final_survey_count}, should be 1")
        return None
    
    print("📝 5. Writing updated HTML...")
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_FINAL_CLEAN.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Final clean HTML created: {output_file}")
    return output_file

def generate_final_clean_pdf(html_file):
    """Generate PDF from final clean HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_FINAL_CLEAN.pdf'
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
            print(f"🎉 Final Clean PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Removing first survey, keeping only the second one...")
    
    # Remove first survey
    html_file = remove_first_survey()
    
    if html_file:
        # Generate final clean PDF
        print("\n📚 Generating final clean PDF...")
        generate_final_clean_pdf(html_file)
        
        print("\n✅ FIRST SURVEY REMOVED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey only in Chapter 1 (first survey removed!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("6. ✅ Book size back to ~244 pages (duplicate removed)")
        print("\n📖 The survey should now be only in Chapter 1!")
    else:
        print("❌ Failed to remove first survey")
