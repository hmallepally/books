#!/usr/bin/env python3
"""
Fix Issue 2: Chapter Titles
"""

def fix_chapter_titles():
    """Fix chapter titles - remove duplicates and fix IDs"""
    
    # Read the current HTML file
    with open('Operational_Excellence_with_AI_TITLE_FIXED.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing chapter titles...")
    
    # Fix Chapter 1
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 1: The AI Revolution in Operations</h2>',
        '<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\n<p class="chapter-subtitle">Transforming Operations Through Artificial Intelligence</p>'
    )
    
    # Fix Chapter 2
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 2: AI-Powered Quality Control and Manufacturing</h1>\nChapter 2: AI-Powered Quality Control and Manufacturing</h2>',
        '<h1 class="chapter-title">Chapter 2: AI-Powered Quality Control and Manufacturing</h1>\n<p class="chapter-subtitle">Revolutionizing Quality Assurance with Intelligent Systems</p>'
    )
    
    # Fix Chapter 3
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h1>\nChapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h2>',
        '<h1 class="chapter-title">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h1>\n<p class="chapter-subtitle">Learning from Successful AI Transformations</p>'
    )
    
    # Fix Chapter 4
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 4: Lean Six Sigma Meets Artificial Intelligence</h1>\nChapter 4: Lean Six Sigma Meets Artificial Intelligence</h2>',
        '<h1 class="chapter-title">Chapter 4: Lean Six Sigma Meets Artificial Intelligence</h1>\n<p class="chapter-subtitle">Enhancing Traditional Methodologies with AI</p>'
    )
    
    # Fix Chapter 5
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 5: Total Productive Maintenance (TPM) in the AI Era</h1>\nChapter 5: Total Productive Maintenance (TPM) in the AI Era</h2>',
        '<h1 class="chapter-title">Chapter 5: Total Productive Maintenance (TPM) in the AI Era</h1>\n<p class="chapter-subtitle">Predictive Maintenance and Equipment Optimization</p>'
    )
    
    # Fix Chapter 6
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h1>\nChapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h2>',
        '<h1 class="chapter-title">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h1>\n<p class="chapter-subtitle">Strategic Planning in the Digital Age</p>'
    )
    
    # Fix Chapter 7
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 7: Total Quality Management Enhanced by AI</h1>\nChapter 7: Total Quality Management Enhanced by AI</h2>',
        '<h1 class="chapter-title">Chapter 7: Total Quality Management Enhanced by AI</h1>\n<p class="chapter-subtitle">Intelligent Quality Management Systems</p>'
    )
    
    # Fix Chapter 8
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 8: AI-Powered Customer Satisfaction and Experience</h1>\nChapter 8: AI-Powered Customer Satisfaction and Experience</h2>',
        '<h1 class="chapter-title">Chapter 8: AI-Powered Customer Satisfaction and Experience</h1>\n<p class="chapter-subtitle">Delivering Exceptional Customer Experiences</p>'
    )
    
    # Fix Chapter 9
    html_content = html_content.replace(
        '<h1 class="chapter-title">Chapter 9: AI in Software Development Lifecycle</h1>\nChapter 9: AI in Software Development Lifecycle</h2>',
        '<h1 class="chapter-title">Chapter 9: AI in Software Development Lifecycle</h1>\n<p class="chapter-subtitle">Accelerating Development with Intelligent Automation</p>'
    )
    
    # Fix Chapter 10 - also fix the wrong ID
    html_content = html_content.replace(
        '<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 10: Leadership in the AI Era</h2>',
        '<div id="chapter10" class="chapter">\n<h1 class="chapter-title">Chapter 10: Leadership in the AI Era</h1>\n<p class="chapter-subtitle">Leading Organizations Through AI Transformation</p>'
    )
    
    # Fix Chapter 11 - also fix the wrong ID
    html_content = html_content.replace(
        '<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 11: Prescriptive Analytics and Future Trends</h2>',
        '<div id="chapter11" class="chapter">\n<h1 class="chapter-title">Chapter 11: Prescriptive Analytics and Future Trends</h1>\n<p class="chapter-subtitle">Predicting and Shaping the Future of Operations</p>'
    )
    
    # Fix Chapter 12 - also fix the wrong ID
    html_content = html_content.replace(
        '<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 12: Implementation Roadmap and Best Practices</h2>',
        '<div id="chapter12" class="chapter">\n<h1 class="chapter-title">Chapter 12: Implementation Roadmap and Best Practices</h1>\n<p class="chapter-subtitle">Your Guide to Successful AI Implementation</p>'
    )
    
    # Fix Chapter 13 - also fix the wrong ID
    html_content = html_content.replace(
        '<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 13: Real-World Case Studies and Success Stories</h2>',
        '<div id="chapter13" class="chapter">\n<h1 class="chapter-title">Chapter 13: Real-World Case Studies and Success Stories</h1>\n<p class="chapter-subtitle">Learning from Industry Success Stories</p>'
    )
    
    # Fix Chapter 14 - also fix the wrong ID
    html_content = html_content.replace(
        '<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 14: The Future of AI in Operational Excellence</h2>',
        '<div id="chapter14" class="chapter">\n<h1 class="chapter-title">Chapter 14: The Future of AI in Operational Excellence</h1>\n<p class="chapter-subtitle">Emerging Trends and Future Possibilities</p>'
    )
    
    # Fix Chapter 15 - also fix the wrong ID
    html_content = html_content.replace(
        '<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>\nChapter 15: AI Tools and Technologies: A Practical Guide</h2>',
        '<div id="chapter15" class="chapter">\n<h1 class="chapter-title">Chapter 15: AI Tools and Technologies: A Practical Guide</h1>\n<p class="chapter-subtitle">Essential Tools for AI Implementation</p>'
    )
    
    # Add CSS for chapter subtitles
    css_addition = """
        /* Chapter subtitle styling */
        .chapter-subtitle {
            font-size: 1.2em;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 0.5in;
            font-style: italic;
        }
    """
    
    # Insert CSS before closing </style> tag
    html_content = html_content.replace('</style>', css_addition + '\n    </style>')
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_CHAPTER_TITLES_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Chapter titles fixed HTML created: {output_file}")
    return output_file

def generate_chapter_titles_fixed_pdf(html_file):
    """Generate PDF from chapter titles fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_CHAPTER_TITLES_FIXED.pdf'
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
            print(f"🎉 Chapter titles fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing chapter titles...")
    
    # Fix chapter titles
    html_file = fix_chapter_titles()
    
    if html_file:
        # Generate chapter titles fixed PDF
        print("\n📚 Generating chapter titles fixed PDF...")
        generate_chapter_titles_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Chapter titles fixed:")
        print("1. ✅ Removed duplicate chapter titles")
        print("2. ✅ Added meaningful subtitles for each chapter")
        print("3. ✅ Fixed wrong chapter IDs (chapter10-15)")
        print("\n📖 Please review this PDF!")
        print("🔍 Check if chapter titles are now clean and properly formatted.")
    else:
        print("❌ Failed to fix chapter titles")
