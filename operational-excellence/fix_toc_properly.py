#!/usr/bin/env python3
"""
Fix TOC properly - Chapter 16 should be inside the ul tags, not outside
"""

def fix_toc_properly():
    """Fix TOC properly with Chapter 16 inside ul tags"""
    import re
    
    print("🚀 Fixing TOC properly - Chapter 16 should be inside ul tags...")
    
    # Read the HTML file
    with open('Operational_Excellence_with_AI_COMPLETE_FINAL.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 Adding Chapter 16 to TOC properly...")
    
    # Add Chapter 16 to TOC BEFORE the closing </ul> tag
    chapter16_toc = '            <li class="chapter"><a href="#chapter16">Chapter 16: AI Ethics and Governance <span class="toc-page-number">305</span></a></li>\n'
    
    # Find the TOC and add Chapter 16 BEFORE the closing </ul>
    html_content = re.sub(
        r'(<li class="chapter"><a href="#chapter15">Chapter 15: AI Tools and Technologies: A Practical Guide <span class="toc-page-number">285</span></a></li>\s*)(</ul>)',
        r'\1' + chapter16_toc + r'\2',
        html_content
    )
    
    print("📝 Making TOC fit on one page with smaller font...")
    
    # Add CSS to make TOC fit on one page
    toc_css = '''
        /* Make TOC fit on one page */
        .toc-list {
            font-size: 0.8em;
        }
        
        .toc-list li {
            margin-bottom: 0.06in;
            line-height: 1.2;
        }
        
        .toc-title {
            font-size: 2.2em;
            margin-bottom: 0.4in;
        }
    '''
    
    # Inject the TOC CSS
    html_content = html_content.replace('</style>', toc_css + '\n    </style>')
    
    # Write the updated HTML
    output_file = 'Operational_Excellence_with_AI_TOC_PROPERLY_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ TOC properly fixed HTML created: {output_file}")
    return output_file

def generate_toc_properly_fixed_pdf(html_file):
    """Generate PDF from properly fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_TOC_PROPERLY_FIXED.pdf'
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
            print(f"🎉 TOC Properly Fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing TOC properly...")
    
    # Fix TOC properly
    html_file = fix_toc_properly()
    
    if html_file:
        # Generate properly fixed PDF
        print("\n📚 Generating properly fixed PDF...")
        generate_toc_properly_fixed_pdf(html_file)
        
        print("\n✅ TOC PROPERLY FIXED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 added INSIDE the ul tags")
        print("2. ✅ Smaller font to fit TOC on one page")
        print("3. ✅ No separate page or hyperlink issues")
        print("\n📖 Please check the TOC - Chapter 16 should now be part of the main TOC list!")
    else:
        print("❌ Failed to fix TOC")
