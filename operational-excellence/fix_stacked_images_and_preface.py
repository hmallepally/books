#!/usr/bin/env python3
"""
Fix Image Stacking and Preface Title
"""

def fix_image_stacking_and_preface():
    """Fix image stacking and preface title issues"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_CHAPTER3_IMAGES_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing image stacking and preface title...")
    
    # Issue 1: Fix image stacking - make them stack vertically instead of scaling
    print("1️⃣ Fixing image stacking...")
    
    # Update CSS for proper image stacking
    image_css = """
        /* Images - proper stacking instead of scaling */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.3in auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }
        
        /* Specific styling for Chapter 3 diagrams - stack them */
        img[src*="ai_success_framework"] {
            max-width: 100%;
            margin: 0.4in auto;
            width: 100%;
        }
        
        img[src*="ai_success_distribution"] {
            max-width: 100%;
            margin: 0.4in auto;
            width: 100%;
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
        
        /* Stack images vertically with proper spacing */
        .image-stack {
            display: flex;
            flex-direction: column;
            gap: 0.5in;
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
    
    # Issue 2: Fix duplicate Preface title and rename it
    print("2️⃣ Fixing duplicate Preface title and renaming...")
    
    # Remove the duplicate "Preface" text
    html_content = html_content.replace('Preface</h2>', '')
    
    # Change the preface title to something more engaging
    html_content = html_content.replace(
        '<h1 class="preface-title">Preface</h1>',
        '<h1 class="preface-title">Introduction: The AI Transformation Journey</h1>'
    )
    
    # Update the TOC to reflect the new title
    html_content = html_content.replace(
        '<a href="#preface">Preface <span class="toc-page-number">5</span></a>',
        '<a href="#preface">Introduction: The AI Transformation Journey <span class="toc-page-number">5</span></a>'
    )
    
    # Issue 3: Improve the image layout by wrapping them in a stack container
    print("3️⃣ Improving image layout with proper stacking...")
    
    # Find the section with the two images and wrap them in a stack container
    image_section = """<figure>
<img src="images/ai_success_distribution.png" alt="AI Success Distribution" />
<figcaption aria-hidden="true">AI Success Distribution</figcaption>
</figure>
<p><em>Figure 3.1: AI Success Distribution - The 10-20-70 principle showing the relative importance of different factors in AI success</em></p>
<figure>
<img src="images/ai_success_framework.png" alt="AI Success Framework" />
<figcaption aria-hidden="true">AI Success Framework</figcaption>
</figure>
<p><em>Figure 3.2: AI Success Framework - Detailed breakdown of the 10-20-70 principle components</em></p>"""
    
    improved_image_section = """<div class="image-stack">
<figure>
<img src="images/ai_success_distribution.png" alt="AI Success Distribution" />
<figcaption aria-hidden="true">AI Success Distribution</figcaption>
</figure>
<p><em>Figure 3.1: AI Success Distribution - The 10-20-70 principle showing the relative importance of different factors in AI success</em></p>

<figure>
<img src="images/ai_success_framework.png" alt="AI Success Framework" />
<figcaption aria-hidden="true">AI Success Framework</figcaption>
</figure>
<p><em>Figure 3.2: AI Success Framework - Detailed breakdown of the 10-20-70 principle components</em></p>
</div>"""
    
    html_content = html_content.replace(image_section, improved_image_section)
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_STACKED_IMAGES_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Stacked images and preface fixed HTML created: {output_file}")
    return output_file

def generate_stacked_images_fixed_pdf(html_file):
    """Generate PDF from stacked images fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_STACKED_IMAGES_FIXED.pdf'
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
            print(f"🎉 Stacked images fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    import re
    
    print("🚀 Fixing image stacking and preface title...")
    
    # Fix image stacking and preface
    html_file = fix_image_stacking_and_preface()
    
    if html_file:
        # Generate stacked images fixed PDF
        print("\n📚 Generating stacked images fixed PDF...")
        generate_stacked_images_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Image stacking and preface fixed:")
        print("1. ✅ Images now stack vertically instead of scaling")
        print("2. ✅ Removed duplicate Preface title")
        print("3. ✅ Renamed Preface to 'Introduction: The AI Transformation Journey'")
        print("4. ✅ Updated TOC with new title")
        print("5. ✅ Better spacing between stacked images")
        print("\n📖 Please review this PDF!")
        print("🔍 Check if the images are now properly stacked and readable.")
    else:
        print("❌ Failed to fix image stacking and preface")
