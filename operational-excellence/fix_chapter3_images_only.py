#!/usr/bin/env python3
"""
Fix Issue 3: Images in Chapter 3 (after the pie chart)
"""

def fix_chapter3_images():
    """Fix images in Chapter 3 - improve sizing and layout"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_CHAPTER_TITLES_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing images in Chapter 3...")
    
    # Issue 1: Remove duplicate ai_success_framework image
    print("1️⃣ Removing duplicate ai_success_framework image...")
    
    # Remove the second duplicate image
    duplicate_image = """<figure>
<img src="images/ai_success_framework.png" alt="AI Success Framework" />
<figcaption aria-hidden="true">AI Success Framework</figcaption>
</figure>"""
    
    html_content = html_content.replace(duplicate_image, '', 1)  # Remove only the first occurrence (the duplicate)
    
    # Issue 2: Improve image sizing and layout
    print("2️⃣ Improving image sizing and layout...")
    
    # Update CSS for better image handling in Chapter 3
    image_css = """
        /* Images - improved sizing for Chapter 3 */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.3in auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }
        
        /* Specific styling for Chapter 3 diagrams */
        img[src*="ai_success_framework"] {
            max-width: 95%;
            margin: 0.4in auto;
            min-height: 400px;
        }
        
        img[src*="ai_success_distribution"] {
            max-width: 80%;
            margin: 0.4in auto;
        }
        
        /* Figure captions */
        .figure-caption {
            text-align: center;
            font-style: italic;
            color: #7f8c8d;
            margin-top: 0.1in;
            font-size: 0.9em;
        }
        
        /* Ensure images don't break across pages */
        figure {
            page-break-inside: avoid;
            margin: 0.4in 0;
        }
    """
    
    # Replace the existing image CSS
    html_content = re.sub(
        r'/\* Images[^}]*\}',
        image_css.strip(),
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 3: Improve the ai_success_framework image specifically
    print("3️⃣ Improving ai_success_framework image layout...")
    
    # Add some text before the ai_success_framework image to provide context
    html_content = html_content.replace(
        '<figure>\n<img src="images/ai_success_framework.png" alt="AI Success Framework" />\n<figcaption aria-hidden="true">AI Success Framework</figcaption>\n</figure>',
        '<p>The framework below provides a detailed breakdown of how organizations can structure their AI initiatives across the three critical areas:</p>\n\n<figure>\n<img src="images/ai_success_framework.png" alt="AI Success Framework" />\n<figcaption aria-hidden="true">AI Success Framework</figcaption>\n</figure>'
    )
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_CHAPTER3_IMAGES_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Chapter 3 images fixed HTML created: {output_file}")
    return output_file

def generate_chapter3_images_fixed_pdf(html_file):
    """Generate PDF from Chapter 3 images fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_CHAPTER3_IMAGES_FIXED.pdf'
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
            print(f"🎉 Chapter 3 images fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    import re
    
    print("🚀 Fixing images in Chapter 3...")
    
    # Fix Chapter 3 images
    html_file = fix_chapter3_images()
    
    if html_file:
        # Generate Chapter 3 images fixed PDF
        print("\n📚 Generating Chapter 3 images fixed PDF...")
        generate_chapter3_images_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Chapter 3 images fixed:")
        print("1. ✅ Removed duplicate ai_success_framework image")
        print("2. ✅ Improved image sizing and layout")
        print("3. ✅ Added context text before the framework image")
        print("4. ✅ Better spacing and readability")
        print("\n📖 Please review this PDF!")
        print("🔍 Check if the images in Chapter 3 are now more readable and properly sized.")
    else:
        print("❌ Failed to fix Chapter 3 images")
