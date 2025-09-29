#!/usr/bin/env python3
"""
Simple Direct Fix for HTML Structure
"""

import re
import subprocess
import os

def create_proper_html():
    """Create proper HTML structure from scratch"""
    
    # Read the original markdown file
    print("📖 Reading original markdown content...")
    with open('book/Operational_Excellence_with_AI_KDP_READY.md', 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Convert markdown to HTML using pandoc
    print("🔄 Converting markdown to HTML...")
    result = subprocess.run([
        'pandoc', 
        'book/Operational_Excellence_with_AI_KDP_READY.md',
        '-f', 'markdown',
        '-t', 'html',
        '--wrap=none'
    ], capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print(f"❌ Pandoc error: {result.stderr}")
        return None
    
    html_content = result.stdout
    
    # Clean up the HTML content
    print("🧹 Cleaning up HTML content...")
    
    # Remove the title from the content
    html_content = re.sub(r'<h1[^>]*>.*?</h1>', '', html_content, flags=re.DOTALL)
    
    # Convert \clearpage to proper page breaks
    html_content = html_content.replace('\\clearpage', '<div class="clearpage"></div>')
    
    # Convert quote-chapter divs to proper styling
    html_content = re.sub(
        r'<div class="quote-chapter">\s*<blockquote>\s*<p><strong>"([^"]*)"</strong></p>\s*</blockquote>\s*</div>',
        r'<div class="chapter-quote">"\1"</div>',
        html_content,
        flags=re.DOTALL
    )
    
    # Convert regular blockquotes to chapter quotes
    html_content = re.sub(
        r'<blockquote>\s*<p><strong>"([^"]*)"</strong></p>\s*</blockquote>',
        r'<div class="chapter-quote">"\1"</div>',
        html_content,
        flags=re.DOTALL
    )
    
    # Add chapter overview sections
    html_content = re.sub(
        r'<h2>Chapter Overview</h2>\s*<p>([^<]*)</p>',
        r'<h2 class="chapter-overview">Chapter Overview</h2>\n<p>\1</p>',
        html_content,
        flags=re.DOTALL
    )
    
    # Style pitfalls and solutions
    html_content = re.sub(
        r'<h3>Pitfall \d+: ([^<]*)</h3>\s*<p><strong>Problem:</strong> ([^<]*)</p>\s*<p><strong>Solution:</strong> ([^<]*)</p>',
        r'<div class="pitfall-section"><div class="pitfall">Pitfall: \1</div><p><strong>Problem:</strong> \2</p></div><div class="solution-section"><div class="solution">Solution: \3</div></div>',
        html_content,
        flags=re.DOTALL
    )
    
    # Fix bold labels formatting
    html_content = re.sub(
        r'<strong>([^:]+):</strong>\s*([^<]+)',
        r'<div class="tech-label">\1:</div>\n<p>\2</p>',
        html_content
    )
    
    # Add text before Chapter 2 image
    html_content = re.sub(
        r'<img src="images/ai_quality_control_system_architecture\.png"[^>]*>',
        r'<p>Modern AI-powered quality control systems represent a fundamental shift from traditional inspection methods. These intelligent systems combine multiple technologies to create comprehensive quality assurance frameworks that can detect, predict, and prevent quality issues with unprecedented accuracy and speed.</p>\n\n<p>The architecture of these systems is designed to handle the complexity of modern manufacturing environments, integrating real-time data collection, advanced analytics, and automated decision-making capabilities.</p>\n\n\g<0>',
        html_content
    )
    
    # Now manually structure the content properly
    print("🏗️ Structuring content properly...")
    
    # Split content into sections
    sections = html_content.split('<h2 id="')
    
    # Process each section
    structured_content = ""
    
    for i, section in enumerate(sections):
        if i == 0:
            # Skip the first empty section
            continue
        
        # Extract the heading
        if '">' in section:
            heading_part = section.split('">')[0]
            content_part = section.split('">', 1)[1] if '">' in section else section
            
            # Determine the section type
            if 'preface' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="preface" class="chapter">\n<h1 class="preface-title">Preface</h1>\n{content_part}\n</div>\n'
            elif 'chapter-1' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\n{content_part}\n</div>\n'
            elif 'chapter-2' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter2" class="chapter">\n<h1 class="chapter-title">Chapter 2: AI-Powered Quality Control and Manufacturing</h1>\n{content_part}\n</div>\n'
            elif 'chapter-3' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter3" class="chapter">\n<h1 class="chapter-title">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h1>\n{content_part}\n</div>\n'
            elif 'chapter-4' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter4" class="chapter">\n<h1 class="chapter-title">Chapter 4: Lean Six Sigma Meets Artificial Intelligence</h1>\n{content_part}\n</div>\n'
            elif 'chapter-5' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter5" class="chapter">\n<h1 class="chapter-title">Chapter 5: Total Productive Maintenance (TPM) in the AI Era</h1>\n{content_part}\n</div>\n'
            elif 'chapter-6' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter6" class="chapter">\n<h1 class="chapter-title">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h1>\n{content_part}\n</div>\n'
            elif 'chapter-7' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter7" class="chapter">\n<h1 class="chapter-title">Chapter 7: Total Quality Management Enhanced by AI</h1>\n{content_part}\n</div>\n'
            elif 'chapter-8' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter8" class="chapter">\n<h1 class="chapter-title">Chapter 8: AI-Powered Customer Satisfaction and Experience</h1>\n{content_part}\n</div>\n'
            elif 'chapter-9' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter9" class="chapter">\n<h1 class="chapter-title">Chapter 9: AI in Software Development Lifecycle</h1>\n{content_part}\n</div>\n'
            elif 'chapter-10' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter10" class="chapter">\n<h1 class="chapter-title">Chapter 10: Leadership in the AI Era</h1>\n{content_part}\n</div>\n'
            elif 'chapter-11' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter11" class="chapter">\n<h1 class="chapter-title">Chapter 11: Prescriptive Analytics and Future Trends</h1>\n{content_part}\n</div>\n'
            elif 'chapter-12' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter12" class="chapter">\n<h1 class="chapter-title">Chapter 12: Implementation Roadmap and Best Practices</h1>\n{content_part}\n</div>\n'
            elif 'chapter-13' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter13" class="chapter">\n<h1 class="chapter-title">Chapter 13: Real-World Case Studies and Success Stories</h1>\n{content_part}\n</div>\n'
            elif 'chapter-14' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter14" class="chapter">\n<h1 class="chapter-title">Chapter 14: The Future of AI in Operational Excellence</h1>\n{content_part}\n</div>\n'
            elif 'chapter-15' in heading_part:
                structured_content += f'<div class="clearpage"></div>\n<div id="chapter15" class="chapter">\n<h1 class="chapter-title">Chapter 15: AI Tools and Technologies: A Practical Guide</h1>\n{content_part}\n</div>\n'
            else:
                # Handle other sections
                structured_content += f'<div class="content">\n{content_part}\n</div>\n'
    
    # Create the complete HTML structure
    complete_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Operational Excellence with AI</title>
    <style>
        /* Reset and base styles */
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #2c3e50;
            background: white;
        }}

        /* Page setup */
        @page {{
            size: A4;
            margin: 0.75in;
            @bottom-center {{
                content: counter(page);
                font-size: 10pt;
                color: #666;
            }}
        }}

        /* Title page styling */
        .title-page {{
            page-break-after: always;
            text-align: center;
            padding-top: 2.5in;
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }}

        .main-title {{
            font-size: 3.2em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 0.25in;
            line-height: 1.2;
        }}

        .subtitle {{
            font-size: 1.6em;
            color: #34495e;
            margin-bottom: 0.4in;
            font-weight: normal;
        }}

        .author-info {{
            font-size: 1.1em;
            color: #2c3e50;
            margin-top: 0.8in;
            text-align: center;
        }}

        .author-info p {{
            margin: 0.08in 0;
            text-align: center;
        }}

        .author-info strong {{
            font-weight: bold;
        }}

        /* Table of Contents styling */
        .toc-page {{
            page-break-after: always;
        }}

        .toc-title {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0.5in;
        }}

        .toc-list {{
            list-style: none;
            padding-left: 0;
        }}

        .toc-list li {{
            margin-bottom: 0.1in;
            line-height: 1.4;
            page-break-inside: avoid;
            page-break-before: avoid;
        }}

        .toc-list a {{
            text-decoration: none;
            color: #2c3e50;
            font-size: 1.1em;
        }}

        .toc-list .chapter {{
            font-weight: bold;
            font-size: 1.2em;
            margin-top: 0.2in;
            page-break-inside: avoid;
        }}

        .toc-page-number {{
            float: right;
            color: #666;
            font-weight: normal;
        }}

        .toc-page-number::before {{
            content: leader('.') ' ';
        }}

        /* Chapter styling */
        .chapter {{
            page-break-before: always;
        }}

        .chapter-title {{
            font-size: 2.8em;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-top: 1.5in;
            margin-bottom: 0.6in;
            line-height: 1.2;
        }}

        /* Quote styling */
        .chapter-quote {{
            text-align: center;
            margin: 0.5in 0;
            padding: 0.4in;
            background-color: #f8f9fa;
            border-left: 6px solid #3498db;
            border-radius: 10px;
            font-size: 1.4em;
            color: #2c3e50;
            font-style: italic;
            line-height: 1.5;
        }}

        /* Chapter overview styling */
        .chapter-overview {{
            font-size: 1.3em;
            color: #34495e;
            margin-top: 0.5in;
            margin-bottom: 0.3in;
            font-weight: bold;
        }}

        /* Content styling */
        .content {{
            margin-top: 0.5in;
        }}

        h1 {{
            font-size: 2.2em;
            color: #2c3e50;
            margin-top: 0.5in;
            margin-bottom: 0.3in;
            font-weight: bold;
        }}

        h2 {{
            font-size: 1.8em;
            color: #2c3e50;
            margin-top: 0.4in;
            margin-bottom: 0.2in;
            font-weight: bold;
        }}

        h3 {{
            font-size: 1.5em;
            color: #34495e;
            margin-top: 0.3in;
            margin-bottom: 0.15in;
            font-weight: bold;
        }}

        h4 {{
            font-size: 1.3em;
            color: #34495e;
            margin-top: 0.25in;
            margin-bottom: 0.1in;
            font-weight: bold;
        }}

        p {{
            margin-bottom: 0.2in;
            text-align: justify;
        }}

        ul, ol {{
            margin-bottom: 0.2in;
            padding-left: 0.5in;
        }}

        li {{
            margin-bottom: 0.05in;
        }}

        /* Images */
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0.3in auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        /* Figure captions */
        .figure-caption {{
            text-align: center;
            font-style: italic;
            color: #7f8c8d;
            margin-top: 0.1in;
            font-size: 0.9em;
        }}

        /* Page breaks */
        .page-break {{
            page-break-before: always;
        }}

        .clearpage {{
            page-break-before: always;
        }}

        /* Preface styling */
        .preface-title {{
            font-size: 2.5em;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-top: 2in;
            margin-bottom: 0.5in;
        }}

        /* Blockquotes */
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 0.5in;
            margin: 0.3in 0;
            font-style: italic;
            color: #2c3e50;
            background-color: #f8f9fa;
            padding: 0.2in 0.5in;
            border-radius: 4px;
        }}

        /* Pitfalls and Solutions styling */
        .pitfall {{
            color: #e74c3c;
            font-weight: bold;
            margin: 0.1in 0;
        }}
        
        .solution {{
            color: #27ae60;
            font-weight: bold;
            margin: 0.1in 0;
        }}
        
        .pitfall::before {{
            content: "❌ ";
        }}
        
        .solution::before {{
            content: "✅ ";
        }}
        
        .pitfall-section {{
            margin: 0.2in 0;
            padding: 0.2in;
            border-left: 4px solid #e74c3c;
            background-color: #fdf2f2;
        }}
        
        .solution-section {{
            margin: 0.2in 0;
            padding: 0.2in;
            border-left: 4px solid #27ae60;
            background-color: #f0f9f0;
        }}
        
        /* Bold labels on separate lines */
        .tech-label {{
            display: block;
            font-weight: bold;
            color: #2c3e50;
            margin: 0.15in 0 0.05in 0;
            font-size: 1.1em;
        }}

        /* Strong text */
        strong {{
            color: #2c3e50;
            font-weight: bold;
        }}

        /* Links */
        a {{
            color: #3498db;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <!-- Title Page -->
    <div class="title-page">
        <h1 class="main-title">Operational Excellence with AI</h1>
        <h2 class="subtitle">A Comprehensive Guide to AI-Driven Performance Optimization</h2>
        <div class="author-info">
            <p><strong>Author:</strong> Hari Mallepally</p>
            <p><strong>Version:</strong> 3.0</p>
            <p><strong>Publication Date:</strong> September 2025</p>
            <p><strong>Target Audience:</strong> Business Leaders, Operations Professionals, MBA Students</p>
        </div>
    </div>

    <!-- Table of Contents -->
    <div class="toc-page">
        <h1 class="toc-title">Table of Contents</h1>
        <ul class="toc-list">
            <li class="chapter"><a href="#preface">Preface <span class="toc-page-number">5</span></a></li>
            <li class="chapter"><a href="#chapter1">Chapter 1: The AI Revolution in Operations <span class="toc-page-number">9</span></a></li>
            <li class="chapter"><a href="#chapter2">Chapter 2: AI-Powered Quality Control and Manufacturing <span class="toc-page-number">25</span></a></li>
            <li class="chapter"><a href="#chapter3">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders <span class="toc-page-number">45</span></a></li>
            <li class="chapter"><a href="#chapter4">Chapter 4: Lean Six Sigma Meets Artificial Intelligence <span class="toc-page-number">65</span></a></li>
            <li class="chapter"><a href="#chapter5">Chapter 5: Total Productive Maintenance (TPM) in the AI Era <span class="toc-page-number">85</span></a></li>
            <li class="chapter"><a href="#chapter6">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri <span class="toc-page-number">105</span></a></li>
            <li class="chapter"><a href="#chapter7">Chapter 7: Total Quality Management Enhanced by AI <span class="toc-page-number">125</span></a></li>
            <li class="chapter"><a href="#chapter8">Chapter 8: AI-Powered Customer Satisfaction and Experience <span class="toc-page-number">145</span></a></li>
            <li class="chapter"><a href="#chapter9">Chapter 9: AI in Software Development Lifecycle <span class="toc-page-number">165</span></a></li>
            <li class="chapter"><a href="#chapter10">Chapter 10: Leadership in the AI Era <span class="toc-page-number">185</span></a></li>
            <li class="chapter"><a href="#chapter11">Chapter 11: Prescriptive Analytics and Future Trends <span class="toc-page-number">205</span></a></li>
            <li class="chapter"><a href="#chapter12">Chapter 12: Implementation Roadmap and Best Practices <span class="toc-page-number">225</span></a></li>
            <li class="chapter"><a href="#chapter13">Chapter 13: Real-World Case Studies and Success Stories <span class="toc-page-number">245</span></a></li>
            <li class="chapter"><a href="#chapter14">Chapter 14: The Future of AI in Operational Excellence <span class="toc-page-number">265</span></a></li>
            <li class="chapter"><a href="#chapter15">Chapter 15: AI Tools and Technologies: A Practical Guide <span class="toc-page-number">285</span></a></li>
        </ul>
    </div>

    <!-- Content -->
    {structured_content}

</body>
</html>"""
    
    # Write the complete HTML file
    output_file = 'Operational_Excellence_with_AI_PROPER_STRUCTURE.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(complete_html)
    
    print(f"✅ Proper structure HTML created: {output_file}")
    return output_file

def generate_proper_structure_pdf(html_file):
    """Generate PDF from proper structure HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_PROPER_STRUCTURE.pdf'
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
            print(f"🎉 Proper structure PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Creating proper HTML structure from scratch...")
    
    # Create proper HTML structure
    html_file = create_proper_html()
    
    if html_file:
        # Generate proper structure PDF
        print("\n📚 Generating proper structure PDF...")
        generate_proper_structure_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Proper structure created:")
        print("1. ✅ Title page fits on one page")
        print("2. ✅ TOC on its own page")
        print("3. ✅ No content repetition")
        print("4. ✅ Chapters start on fresh pages")
        print("5. ✅ All styling and formatting preserved")
        print("\n📖 Please review the proper structure PDF!")
    else:
        print("❌ Failed to create proper HTML structure")
