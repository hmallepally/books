#!/usr/bin/env python3
"""
Fix AI Success Framework Image Readability
"""

def fix_ai_success_framework_image():
    """Fix the AI Success Framework image to be more readable"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_STACKED_IMAGES_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing AI Success Framework image readability...")
    
    # Issue 1: Clean up duplicate CSS rules
    print("1️⃣ Cleaning up duplicate CSS rules...")
    
    # Remove all the duplicate CSS rules and create one clean set
    html_content = re.sub(
        r'/\* Specific styling for Chapter 3 diagrams[^}]*\}',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    html_content = re.sub(
        r'/\* Specific styling for diagrams[^}]*\}',
        '',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 2: Create a simpler, more readable AI Success Framework
    print("2️⃣ Creating a simpler AI Success Framework...")
    
    # Create a new, simpler mermaid diagram
    simple_framework = """
graph TD
    A["🎯 AI Success Framework"] --> B["📊 10% Algorithms"]
    A --> C["🔧 20% Data & Infrastructure"] 
    A --> D["👥 70% People & Process"]
    
    B --> E["Machine Learning<br/>Deep Learning<br/>NLP"]
    C --> F["Data Quality<br/>Cloud Infrastructure<br/>Security"]
    D --> G["Change Management<br/>Training<br/>Culture"]
    
    style A fill:#2c3e50,stroke:#34495e,stroke-width:3px,color:#fff
    style B fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    style C fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff
    style D fill:#27ae60,stroke:#229954,stroke-width:2px,color:#fff
    style E fill:#ecf0f1,stroke:#bdc3c7,stroke-width:1px,color:#2c3e50
    style F fill:#ecf0f1,stroke:#bdc3c7,stroke-width:1px,color:#2c3e50
    style G fill:#ecf0f1,stroke:#bdc3c7,stroke-width:1px,color:#2c3e50
    """
    
    # Write the simpler mermaid code
    with open('diagrams/ai_success_framework_simple.mmd', 'w', encoding='utf-8') as f:
        f.write(simple_framework)
    
    # Issue 3: Replace the complex image with a simpler version
    print("3️⃣ Replacing with simpler image...")
    
    # Replace the complex image with a simpler one
    html_content = html_content.replace(
        '<img src="images/ai_success_framework.png" alt="AI Success Framework" />',
        '<img src="images/ai_success_framework_simple.png" alt="AI Success Framework - Simplified" />'
    )
    
    # Issue 4: Add clean CSS for the image
    print("4️⃣ Adding clean CSS...")
    
    # Add clean CSS before the closing </style> tag
    clean_css = """
        /* Clean image styling */
        img[src*="ai_success_framework"] {
            max-width: 100%;
            width: 100%;
            height: auto;
            margin: 0.4in auto;
            display: block;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Ensure the image is readable */
        figure {
            page-break-inside: avoid;
            margin: 0.4in 0;
            text-align: center;
        }
    """
    
    html_content = html_content.replace('</style>', clean_css + '\n    </style>')
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_FRAMEWORK_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ AI Success Framework fixed HTML created: {output_file}")
    return output_file

def generate_framework_fixed_pdf(html_file):
    """Generate PDF from framework fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_FRAMEWORK_FIXED.pdf'
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
            print(f"🎉 Framework fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    import re
    
    print("🚀 Fixing AI Success Framework image readability...")
    
    # Fix the framework image
    html_file = fix_ai_success_framework_image()
    
    if html_file:
        # Generate framework fixed PDF
        print("\n📚 Generating framework fixed PDF...")
        generate_framework_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 AI Success Framework fixed:")
        print("1. ✅ Cleaned up duplicate CSS rules")
        print("2. ✅ Created simpler framework diagram")
        print("3. ✅ Replaced complex image with readable version")
        print("4. ✅ Added clean CSS for proper sizing")
        print("\n📖 Please review this PDF!")
        print("🔍 Check if the AI Success Framework image on page 39 is now readable.")
    else:
        print("❌ Failed to fix AI Success Framework image")
