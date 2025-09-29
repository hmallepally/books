#!/usr/bin/env python3
"""
Fix HTML Structure Properly
"""

import re
import subprocess
import os

def fix_html_structure():
    """Fix the HTML structure properly"""
    
    # Read the current clean HTML file
    with open('Operational_Excellence_with_AI_CLEAN.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    print("🔧 Fixing HTML structure...")
    
    # Issue 1: Remove the duplicate title content that appears after TOC
    print("1️⃣ Removing duplicate title content...")
    
    # Find the content section and remove the duplicate title info
    html_content = re.sub(
        r'<div class="content">\s*<h2 id="a-comprehensive-guide-to-ai-driven-performance-optimization">A Comprehensive Guide to AI-Driven Performance Optimization</h2>\s*<p><div class="tech-label">Author:</div>\s*<p>Hari Mallepally</p><br />\s*<div class="tech-label">Version:</div>\s*<p>3\.0</p><br />\s*<div class="tech-label">Publication Date:</div>\s*<p>September 2025</p><br />\s*<div class="tech-label">Target Audience:</div>\s*<p>Business Leaders, Operations Professionals, MBA Students</p></p>',
        '<div class="content">',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 2: Fix the Preface structure
    print("2️⃣ Fixing Preface structure...")
    
    # Replace the Preface heading with proper structure
    html_content = re.sub(
        r'<h2 id="preface">Preface</h2>',
        '<div class="clearpage"></div>\n<div id="preface" class="chapter">\n<h1 class="preface-title">Preface</h1>',
        html_content
    )
    
    # Issue 3: Fix chapter structures
    print("3️⃣ Fixing chapter structures...")
    
    # Fix Chapter 1
    html_content = re.sub(
        r'<h2 id="chapter-1-the-ai-revolution-in-operations">Chapter 1: The AI Revolution in Operations</h2>',
        '<div class="clearpage"></div>\n<div id="chapter1" class="chapter">\n<h1 class="chapter-title">Chapter 1: The AI Revolution in Operations</h1>',
        html_content
    )
    
    # Fix Chapter 2
    html_content = re.sub(
        r'<h2 id="chapter-2-ai-powered-quality-control-and-manufacturing">Chapter 2: AI-Powered Quality Control and Manufacturing</h2>',
        '<div class="clearpage"></div>\n<div id="chapter2" class="chapter">\n<h1 class="chapter-title">Chapter 2: AI-Powered Quality Control and Manufacturing</h1>',
        html_content
    )
    
    # Fix Chapter 3
    html_content = re.sub(
        r'<h2 id="chapter-3-strategic-ai-implementation-lessons-from-industry-leaders">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h2>',
        '<div class="clearpage"></div>\n<div id="chapter3" class="chapter">\n<h1 class="chapter-title">Chapter 3: Strategic AI Implementation: Lessons from Industry Leaders</h1>',
        html_content
    )
    
    # Fix Chapter 4
    html_content = re.sub(
        r'<h2 id="chapter-4-lean-six-sigma-meets-artificial-intelligence">Chapter 4: Lean Six Sigma Meets Artificial Intelligence</h2>',
        '<div class="clearpage"></div>\n<div id="chapter4" class="chapter">\n<h1 class="chapter-title">Chapter 4: Lean Six Sigma Meets Artificial Intelligence</h1>',
        html_content
    )
    
    # Fix Chapter 5
    html_content = re.sub(
        r'<h2 id="chapter-5-total-productive-maintenance-tpm-in-the-ai-era">Chapter 5: Total Productive Maintenance \(TPM\) in the AI Era</h2>',
        '<div class="clearpage"></div>\n<div id="chapter5" class="chapter">\n<h1 class="chapter-title">Chapter 5: Total Productive Maintenance (TPM) in the AI Era</h1>',
        html_content
    )
    
    # Fix Chapter 6
    html_content = re.sub(
        r'<h2 id="chapter-6-the-ai-powered-strategic-compass-hoshin-kanri">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h2>',
        '<div class="clearpage"></div>\n<div id="chapter6" class="chapter">\n<h1 class="chapter-title">Chapter 6: The AI-Powered Strategic Compass: Hoshin Kanri</h1>',
        html_content
    )
    
    # Fix Chapter 7
    html_content = re.sub(
        r'<h2 id="chapter-7-total-quality-management-enhanced-by-ai">Chapter 7: Total Quality Management Enhanced by AI</h2>',
        '<div class="clearpage"></div>\n<div id="chapter7" class="chapter">\n<h1 class="chapter-title">Chapter 7: Total Quality Management Enhanced by AI</h1>',
        html_content
    )
    
    # Fix Chapter 8
    html_content = re.sub(
        r'<h2 id="chapter-8-ai-powered-customer-satisfaction-and-experience">Chapter 8: AI-Powered Customer Satisfaction and Experience</h2>',
        '<div class="clearpage"></div>\n<div id="chapter8" class="chapter">\n<h1 class="chapter-title">Chapter 8: AI-Powered Customer Satisfaction and Experience</h1>',
        html_content
    )
    
    # Fix Chapter 9
    html_content = re.sub(
        r'<h2 id="chapter-9-ai-in-software-development-lifecycle">Chapter 9: AI in Software Development Lifecycle</h2>',
        '<div class="clearpage"></div>\n<div id="chapter9" class="chapter">\n<h1 class="chapter-title">Chapter 9: AI in Software Development Lifecycle</h1>',
        html_content
    )
    
    # Fix Chapter 10
    html_content = re.sub(
        r'<h2 id="chapter-10-leadership-in-the-ai-era">Chapter 10: Leadership in the AI Era</h2>',
        '<div class="clearpage"></div>\n<div id="chapter10" class="chapter">\n<h1 class="chapter-title">Chapter 10: Leadership in the AI Era</h1>',
        html_content
    )
    
    # Fix Chapter 11
    html_content = re.sub(
        r'<h2 id="chapter-11-prescriptive-analytics-and-future-trends">Chapter 11: Prescriptive Analytics and Future Trends</h2>',
        '<div class="clearpage"></div>\n<div id="chapter11" class="chapter">\n<h1 class="chapter-title">Chapter 11: Prescriptive Analytics and Future Trends</h1>',
        html_content
    )
    
    # Fix Chapter 12
    html_content = re.sub(
        r'<h2 id="chapter-12-implementation-roadmap-and-best-practices">Chapter 12: Implementation Roadmap and Best Practices</h2>',
        '<div class="clearpage"></div>\n<div id="chapter12" class="chapter">\n<h1 class="chapter-title">Chapter 12: Implementation Roadmap and Best Practices</h1>',
        html_content
    )
    
    # Fix Chapter 13
    html_content = re.sub(
        r'<h2 id="chapter-13-real-world-case-studies-and-success-stories">Chapter 13: Real-World Case Studies and Success Stories</h2>',
        '<div class="clearpage"></div>\n<div id="chapter13" class="chapter">\n<h1 class="chapter-title">Chapter 13: Real-World Case Studies and Success Stories</h1>',
        html_content
    )
    
    # Fix Chapter 14
    html_content = re.sub(
        r'<h2 id="chapter-14-the-future-of-ai-in-operational-excellence">Chapter 14: The Future of AI in Operational Excellence</h2>',
        '<div class="clearpage"></div>\n<div id="chapter14" class="chapter">\n<h1 class="chapter-title">Chapter 14: The Future of AI in Operational Excellence</h1>',
        html_content
    )
    
    # Fix Chapter 15
    html_content = re.sub(
        r'<h2 id="chapter-15-ai-tools-and-technologies-a-practical-guide">Chapter 15: AI Tools and Technologies: A Practical Guide</h2>',
        '<div class="clearpage"></div>\n<div id="chapter15" class="chapter">\n<h1 class="chapter-title">Chapter 15: AI Tools and Technologies: A Practical Guide</h1>',
        html_content
    )
    
    # Issue 4: Fix quote structures
    print("4️⃣ Fixing quote structures...")
    
    # Fix the quote-chapter divs
    html_content = re.sub(
        r'<div class="quote-chapter">\s*<blockquote>\s*<p><strong>"([^"]*)"</strong></p>\s*</blockquote>\s*</div>',
        r'<div class="chapter-quote">"\1"</div>',
        html_content,
        flags=re.DOTALL
    )
    
    # Issue 5: Close all chapter divs properly
    print("5️⃣ Closing chapter divs...")
    
    # Add closing divs before each new chapter
    html_content = re.sub(
        r'<div class="clearpage"></div>\n<div id="chapter\d+" class="chapter">',
        '</div>\n<div class="clearpage"></div>\n<div id="chapter\\1" class="chapter">',
        html_content
    )
    
    # Close the last chapter
    html_content = html_content.replace('</body>', '</div>\n</body>')
    
    # Write the fixed HTML
    output_file = 'Operational_Excellence_with_AI_STRUCTURE_FIXED.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Structure fixed HTML created: {output_file}")
    return output_file

def generate_structure_fixed_pdf(html_file):
    """Generate PDF from structure fixed HTML"""
    import asyncio
    from playwright.async_api import async_playwright
    import os
    
    async def generate_pdf():
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            await page.goto(f"file://{os.path.abspath(html_file)}")
            await page.wait_for_load_state('networkidle')
            
            output_pdf = 'Operational_Excellence_with_AI_STRUCTURE_FIXED.pdf'
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
            print(f"🎉 Structure fixed PDF generated: {output_pdf}")
            print(f"📄 File size: {os.path.getsize(output_pdf) / 1024:.1f} KB")
    
    asyncio.run(generate_pdf())

if __name__ == "__main__":
    print("🚀 Fixing HTML structure properly...")
    
    # Fix HTML structure
    html_file = fix_html_structure()
    
    if html_file:
        # Generate structure fixed PDF
        print("\n📚 Generating structure fixed PDF...")
        generate_structure_fixed_pdf(html_file)
        
        print("\n✅ SUCCESS!")
        print("📋 Structure fixes applied:")
        print("1. ✅ Removed duplicate title content after TOC")
        print("2. ✅ Fixed Preface structure with proper page break")
        print("3. ✅ Fixed all chapter structures with proper page breaks")
        print("4. ✅ Fixed quote structures")
        print("5. ✅ Properly closed all chapter divs")
        print("\n📖 Please review the structure fixed PDF!")
    else:
        print("❌ Failed to fix HTML structure")
