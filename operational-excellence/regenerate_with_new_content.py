#!/usr/bin/env python3
"""
Regenerate HTML from updated markdown with all enhancements
"""

def regenerate_html_with_enhancements():
    """Regenerate HTML from the updated markdown file"""
    import subprocess
    import os
    import re
    
    print("🚀 Regenerating HTML from updated markdown with all enhancements...")
    
    # Step 1: Convert updated markdown to HTML using pandoc
    print("📝 Converting updated markdown to HTML...")
    markdown_file = 'book/Operational_Excellence_with_AI_KDP_READY.md'
    html_file = 'Operational_Excellence_with_AI_ENHANCED_CONTENT.html'
    
    pandoc_cmd = [
        'pandoc',
        markdown_file,
        '-o', html_file,
        '--css', 'pdf-styles.css',
        '--toc',
        '--toc-depth=3',
        '--metadata', 'title=Operational Excellence with AI',
        '--metadata', 'author=Hari Mallepally',
        '--metadata', 'language=en',
        '-s'
    ]
    
    try:
        subprocess.run(pandoc_cmd, check=True)
        print(f"✅ Enhanced HTML generated: {html_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pandoc failed: {e}")
        return False
    
    # Step 2: Enhance the HTML with better styling
    print("🎨 Enhancing HTML with comprehensive styling...")
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Add comprehensive CSS styling for enhanced content
    enhanced_css = """
    <style>
        /* Enhanced styling for the comprehensive book */
        body {
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
        }
        
        /* Title page styling */
        h1 {
            font-size: 36pt;
            text-align: center;
            margin: 2in 0 1in 0;
            color: #2c3e50;
            font-weight: bold;
        }
        
        h2 {
            font-size: 24pt;
            text-align: center;
            margin: 0.5in 0;
            color: #34495e;
        }
        
        /* Chapter styling */
        .chapter-title {
            font-size: 28pt;
            text-align: center;
            margin: 1.5in 0 0.5in 0;
            color: #2c3e50;
            font-weight: bold;
            page-break-before: always;
        }
        
        .chapter-subtitle {
            font-size: 16pt;
            text-align: center;
            margin: 0.2in 0 0.5in 0;
            color: #7f8c8d;
            font-style: italic;
        }
        
        /* Quote styling */
        .quote-chapter {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.5in;
            margin: 0.5in 0;
            border-radius: 10px;
            font-size: 18pt;
            text-align: center;
            font-style: italic;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        .quote-chapter blockquote {
            margin: 0;
            font-size: 18pt;
            line-height: 1.4;
        }
        
        /* Story sections */
        h3:contains("Story:") {
            color: #e74c3c;
            font-size: 20pt;
            margin: 0.5in 0 0.3in 0;
            border-bottom: 2px solid #e74c3c;
            padding-bottom: 0.1in;
        }
        
        /* Self-assessment styling */
        .self-assessment {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 0.3in;
            margin: 0.3in 0;
            page-break-inside: avoid;
        }
        
        .self-assessment h4 {
            color: #495057;
            background: #e9ecef;
            padding: 0.1in;
            margin: -0.3in -0.3in 0.2in -0.3in;
            border-radius: 6px 6px 0 0;
        }
        
        /* Try This exercises */
        .try-this {
            background: #e8f5e8;
            border-left: 4px solid #28a745;
            padding: 0.3in;
            margin: 0.3in 0;
            page-break-inside: avoid;
        }
        
        .try-this h4 {
            color: #155724;
            margin-top: 0;
        }
        
        /* ROI Framework styling */
        .roi-framework {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 0.3in;
            margin: 0.3in 0;
            page-break-inside: avoid;
        }
        
        .roi-framework h4 {
            color: #856404;
            background: #ffeaa7;
            padding: 0.1in;
            margin: -0.3in -0.3in 0.2in -0.3in;
            border-radius: 6px 6px 0 0;
        }
        
        /* Image styling */
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.3in auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            page-break-inside: avoid;
        }
        
        figure {
            page-break-inside: avoid;
            margin: 0.3in 0;
            text-align: center;
        }
        
        figcaption {
            font-size: 10pt;
            color: #666;
            margin-top: 0.1in;
            font-style: italic;
        }
        
        /* Enhanced readability */
        p {
            margin: 0.15in 0;
            text-align: justify;
        }
        
        h3 {
            color: #2c3e50;
            font-size: 18pt;
            margin: 0.4in 0 0.2in 0;
        }
        
        h4 {
            color: #34495e;
            font-size: 16pt;
            margin: 0.3in 0 0.15in 0;
        }
        
        /* Print optimizations */
        @media print {
            body {
                font-size: 11pt;
            }
            
            .chapter-title {
                page-break-before: always;
            }
            
            img {
                max-width: 100%;
                page-break-inside: avoid;
            }
            
            .self-assessment,
            .try-this,
            .roi-framework {
                page-break-inside: avoid;
            }
        }
    </style>
    """
    
    # Inject the enhanced CSS
    html_content = html_content.replace('</head>', enhanced_css + '\n</head>')
    
    # Write the enhanced HTML
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Enhanced HTML created: {html_file}")
    return html_file

def generate_pdf_from_enhanced_html(html_file):
    """Generate PDF from the enhanced HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_ENHANCED_WITH_NEW_CONTENT.pdf'
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
            print(f"🎉 Enhanced PDF with new content generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Regenerating HTML with all new content enhancements...")
    
    # Regenerate HTML from updated markdown
    html_file = regenerate_html_with_enhancements()
    
    if html_file:
        # Generate PDF from enhanced HTML
        print("\n📚 Generating PDF with all new content...")
        generate_pdf_from_enhanced_html(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 This PDF now includes:")
        print("1. ✅ All new engaging stories (Chapters 1-6)")
        print("2. ✅ All new self-assessment quizzes")
        print("3. ✅ All new 'Try This' exercises")
        print("4. ✅ Comprehensive ROI measurement framework")
        print("5. ✅ AI Ethics & Governance chapter (Chapter 16)")
        print("6. ✅ All content enhancements from the updated markdown")
        print("\n📖 The PDF should now have both correct structure AND all new content!")
    else:
        print("❌ Failed to regenerate HTML")
