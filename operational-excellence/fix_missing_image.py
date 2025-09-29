#!/usr/bin/env python3
"""
Fix Missing AI Success Framework Image
"""

def fix_missing_image():
    """Fix the missing AI Success Framework image"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_FRAMEWORK_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing missing AI Success Framework image...")
    
    # Issue 1: Revert to the original image file
    print("1️⃣ Reverting to original image file...")
    
    # Change back to the original image file
    html_content = html_content.replace(
        '<img src="images/ai_success_framework_simple.png" alt="AI Success Framework - Simplified" />',
        '<img src="images/ai_success_framework.png" alt="AI Success Framework" />'
    )
    
    # Issue 2: Create much better CSS for the existing image
    print("2️⃣ Creating better CSS for the existing image...")
    
    # Remove the old CSS and add much better CSS
    html_content = re.sub(
        r'/\* Clean image styling[^}]*\}',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Add much better CSS for the AI Success Framework image
    better_css = """
        /* Much better styling for AI Success Framework */
        img[src*="ai_success_framework"] {
            max-width: 100% !important;
            width: 100% !important;
            height: auto !important;
            margin: 0.2in auto !important;
            display: block !important;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            object-fit: contain !important;
            page-break-inside: avoid;
        }
        
        /* Ensure the figure container is properly sized */
        figure {
            page-break-inside: avoid;
            margin: 0.2in 0;
            text-align: center;
            width: 100%;
        }
        
        /* Make sure the image doesn't get compressed */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.3in auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }
    """
    
    # Insert the better CSS before closing </style> tag
    html_content = html_content.replace('</style>', better_css + '\n    </style>')
    
    # Issue 3: Add some explanatory text to make the image more useful
    print("3️⃣ Adding explanatory text...")
    
    # Add text before the image to explain what it shows
    html_content = html_content.replace(
        '<figure>\n<img src="images/ai_success_framework.png" alt="AI Success Framework" />\n<figcaption aria-hidden="true">AI Success Framework</figcaption>\n</figure>',
        '<p>The framework below illustrates how successful AI implementations balance three critical components:</p>\n\n<figure>\n<img src="images/ai_success_framework.png" alt="AI Success Framework" />\n<figcaption aria-hidden="true">AI Success Framework</figcaption>\n</figure>'
    )
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_IMAGE_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Image fixed HTML created: {output_file}")
    return output_file

def generate_image_fixed_pdf(html_file):
    """Generate PDF from image fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_IMAGE_FIXED.pdf'
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
            print(f"🎉 Image fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    import re
    
    print("🚀 Fixing missing AI Success Framework image...")
    
    # Fix the missing image
    html_file = fix_missing_image()
    
    if html_file:
        # Generate image fixed PDF
        print("\n📚 Generating image fixed PDF...")
        generate_image_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 AI Success Framework image fixed:")
        print("1. ✅ Reverted to original image file")
        print("2. ✅ Added much better CSS with !important rules")
        print("3. ✅ Added explanatory text before the image")
        print("4. ✅ Ensured image takes full width and is readable")
        print("\n📖 Please review this PDF!")
        print("🔍 Check if the AI Success Framework image now shows up and is readable.")
    else:
        print("❌ Failed to fix missing image")
