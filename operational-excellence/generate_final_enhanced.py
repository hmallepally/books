#!/usr/bin/env python3
"""
Generate Final Enhanced PDF with All Improvements
"""

def generate_enhanced_pdf():
    """Generate the final enhanced PDF with all improvements"""
    import asyncio
    from playwright.async_api import async_playwright
    import subprocess
    import os
    import re
    
    print("🚀 Generating Final Enhanced PDF with All Improvements...")
    
    # Step 1: Convert Markdown to HTML using pandoc
    print("📝 Converting markdown to HTML...")
    markdown_file = 'book/Operational_Excellence_with_AI_KDP_READY.md'
    html_file = 'Operational_Excellence_with_AI_ENHANCED.html'
    
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
        print(f"✅ HTML generated: {html_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pandoc failed: {e}")
        return False
    
    # Step 2: Read and enhance the HTML
    print("🎨 Enhancing HTML with custom styling...")
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Add comprehensive CSS styling
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
        }
        
        .try-this h4 {
            color: #155724;
            margin-top: 0;
        }
        
        /* Pitfall and Solution styling */
        .pitfall-section {
            background: #ffeaea;
            border-left: 4px solid #dc3545;
            padding: 0.2in;
            margin: 0.2in 0;
        }
        
        .solution-section {
            background: #eafaf1;
            border-left: 4px solid #28a745;
            padding: 0.2in;
            margin: 0.2in 0;
        }
        
        .pitfall-section strong {
            color: #dc3545;
        }
        
        .solution-section strong {
            color: #28a745;
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
        
        /* Table styling */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 0.3in 0;
            page-break-inside: avoid;
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 0.1in;
            text-align: left;
        }
        
        th {
            background: #f8f9fa;
            font-weight: bold;
        }
        
        /* List styling */
        ul, ol {
            margin: 0.2in 0;
            padding-left: 0.3in;
        }
        
        li {
            margin: 0.1in 0;
        }
        
        /* Page breaks */
        .page-break {
            page-break-before: always;
        }
        
        /* TOC styling */
        #TOC {
            page-break-after: always;
        }
        
        #TOC ul {
            list-style: none;
            padding-left: 0;
        }
        
        #TOC li {
            margin: 0.1in 0;
        }
        
        #TOC a {
            text-decoration: none;
            color: #2c3e50;
        }
        
        /* Code styling */
        code {
            background: #f8f9fa;
            padding: 0.05in;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        
        pre {
            background: #f8f9fa;
            padding: 0.2in;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
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
        
        h5 {
            color: #7f8c8d;
            font-size: 14pt;
            margin: 0.2in 0 0.1in 0;
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
            .try-this {
                page-break-inside: avoid;
            }
        }
    </style>
    """
    
    # Inject the enhanced CSS
    html_content = html_content.replace('</head>', enhanced_css + '\n</head>')
    
    # Add page breaks before chapters
    html_content = re.sub(
        r'<h1 class="chapter-title">(Chapter \d+:.*?)</h1>',
        r'<div class="page-break"></div>\n<h1 class="chapter-title">\1</h1>',
        html_content
    )
    
    # Write the enhanced HTML
    enhanced_html_file = 'Operational_Excellence_with_AI_ENHANCED.html'
    with open(enhanced_html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Enhanced HTML created: {enhanced_html_file}")
    
    # Step 3: Generate PDF using Playwright
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(enhanced_html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_FINAL_ENHANCED.pdf'
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
            print(f"🎉 Final Enhanced PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
            return output_pdf
    
    # Generate the PDF
    pdf_file = asyncio.run(generate_pdf())
    
    if pdf_file:
        print("\n✅ SUCCESS! Final Enhanced Book Generated!")
        print("📋 Enhancements included:")
        print("1. ✅ Engaging opening stories for each chapter")
        print("2. ✅ Interactive self-assessment quizzes")
        print("3. ✅ 'Try This' hands-on exercises")
        print("4. ✅ New AI Ethics & Governance chapter (Chapter 16)")
        print("5. ✅ Enhanced storytelling and narrative flow")
        print("6. ✅ Professional styling and formatting")
        print("7. ✅ Comprehensive case studies and examples")
        print("8. ✅ Practical implementation frameworks")
        print("\n📖 The book is now ready for final review!")
        return True
    else:
        print("❌ Failed to generate final enhanced PDF")
        return False

if __name__ == "__main__":
    generate_enhanced_pdf()
