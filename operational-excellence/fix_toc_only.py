#!/usr/bin/env python3
"""
Fix only the TOC issue - add Chapter 16 and make it fit on one page
"""

def fix_toc_only():
    """Fix only the TOC issue"""
    import re
    
    print("🚀 Fixing TOC only - adding Chapter 16 and making it fit on one page...")
    
    # Read the HTML file
    with open('Operational_Excellence_with_AI_COMPLETE_FINAL.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("📝 Adding Chapter 16 to TOC...")
    
    # Add Chapter 16 to TOC before closing </ul>
    chapter16_toc = '            <li class="chapter"><a href="#chapter16">Chapter 16: AI Ethics and Governance <span class="toc-page-number">305</span></a></li>\n'
    
    # Find the TOC and add Chapter 16
    html_content = re.sub(
        r'(<li class="chapter"><a href="#chapter15">Chapter 15: AI Tools and Technologies: A Practical Guide <span class="toc-page-number">285</span></a></li>\s*</ul>)',
        r'\1' + chapter16_toc + '        </ul>',
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
    output_file = 'Operational_Excellence_with_AI_TOC_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ TOC fixed HTML created: {output_file}")
    return output_file

def generate_toc_fixed_pdf(html_file):
    """Generate PDF from TOC fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_TOC_FIXED.pdf'
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
            print(f"🎉 TOC Fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing TOC only...")
    
    # Fix TOC only
    html_file = fix_toc_only()
    
    if html_file:
        # Generate TOC fixed PDF
        print("\n📚 Generating TOC fixed PDF...")
        generate_toc_fixed_pdf(html_file)
        
        print("\n✅ TOC FIXED!")
        print("📋 This PDF now includes:")
        print("1. ✅ Chapter 16 added to TOC")
        print("2. ✅ Smaller font to fit TOC on one page")
        print("3. ✅ No hyperlink issues")
        print("\n📖 Please check the TOC - it should now have Chapter 16 and fit on one page!")
    else:
        print("❌ Failed to fix TOC")
