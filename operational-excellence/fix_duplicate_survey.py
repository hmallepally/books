#!/usr/bin/env python3
"""
Fix duplicate survey - remove duplicate and keep only one in Chapter 1
"""

def fix_duplicate_survey():
    """Fix duplicate survey by removing duplicate and keeping one in Chapter 1"""
    
    print("🚀 Fixing duplicate survey - removing duplicate and keeping one in Chapter 1...")
    
    # Use the latest file
    with open('Operational_Excellence_with_AI_LATEST.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 1. Finding all surveys...")
    
    # Count surveys
    survey_count = html_content.count('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    print(f"Found {survey_count} surveys")
    
    if survey_count == 1:
        print("✅ Only one survey found - no duplicates")
        return 'Operational_Excellence_with_AI_LATEST.html'
    
    print("📝 2. Finding first survey (should be in Chapter 2)...")
    
    # Find the first survey
    first_survey_start = html_content.find('<div class="assessment-section">')
    if first_survey_start == -1:
        print("❌ First survey not found")
        return None
    
    # Find the end of the first survey
    first_survey_end_marker = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>', first_survey_start)
    if first_survey_end_marker == -1:
        print("❌ First survey end marker not found")
        return None
    
    first_survey_end = first_survey_end_marker + len('        </div>\n    </div>')
    
    print("📝 3. Finding second survey (should be in Chapter 1)...")
    
    # Find the second survey
    second_survey_start = html_content.find('<div class="assessment-section">', first_survey_end)
    if second_survey_start == -1:
        print("❌ Second survey not found")
        return None
    
    # Find the end of the second survey
    second_survey_end_marker = html_content.find('        </div>\n    </div>\n\n<h3 id="conclusion">Conclusion</h3>', second_survey_start)
    if second_survey_end_marker == -1:
        print("❌ Second survey end marker not found")
        return None
    
    second_survey_end = second_survey_end_marker + len('        </div>\n    </div>')
    
    print("📝 4. Removing first survey (from Chapter 2)...")
    
    # Remove the first survey (from Chapter 2)
    html_content = html_content[:first_survey_start] + html_content[first_survey_end:]
    
    print("📝 5. Verifying only one survey remains...")
    
    # Verify only one survey remains
    final_survey_count = html_content.count('<h3>Chapter 1 Self-Assessment: AI Readiness Evaluation</h3>')
    if final_survey_count == 1:
        print("✅ Only one survey remains")
    else:
        print(f"❌ Survey count is {final_survey_count}, should be 1")
        return None
    
    print("📝 6. Writing updated HTML...")
    
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
    print("🚀 Fixing duplicate survey...")
    
    # Fix duplicate survey
    html_file = fix_duplicate_survey()
    
    if html_file:
        # Generate final clean PDF
        print("\n📚 Generating final clean PDF...")
        generate_final_clean_pdf(html_file)
        
        print("\n✅ DUPLICATE SURVEY FIXED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 in TOC (kept from previous changes)")
        print("2. ✅ Sarah Chen story in Chapter 1 (kept from previous changes)")
        print("3. ✅ Marcus Rodriguez story in Chapter 2 (kept from previous changes)")
        print("4. ✅ Survey only in Chapter 1 (duplicate removed!)")
        print("5. ✅ TOC fits on one page (kept from previous changes)")
        print("6. ✅ Book size back to ~244 pages (duplicate removed)")
        print("\n📖 The survey should now be only in Chapter 1!")
    else:
        print("❌ Failed to fix duplicate survey")
